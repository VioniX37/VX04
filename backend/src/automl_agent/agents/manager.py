"""Agent Manager: orchestrates the full pipeline (paper Fig. 2, plus grounded verification).

Stages::

    parse -> verify_request -> prepare -> retrieve ->
    [ plan -> execute_plans -> (ground) -> select -> implement -> verify_impl ] x (1 + revisions)

With ``VERIFICATION_MODE=pseudo`` the ``ground`` stage is skipped and plans are
ranked on the agents' predicted scores, exactly as in the paper. With
``grounded`` (default) plans are ranked on real multi-fidelity runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from automl_agent.execution.model_registry import normalize_model, supported_models
from automl_agent.execution.renderer import render_template
from automl_agent.execution.sandbox import run_script
from automl_agent.extensions import PipelineHooks, registered_hooks, run_hook
from automl_agent.memory import MemoryHooks, MemoryRetriever
from automl_agent.planning.decomposition import decompose
from automl_agent.planning.gemini_search import GeminiSearchRetriever
from automl_agent.planning.retrieval import KnowledgeItem, LocalKnowledgeRetriever, Retriever, retrieve_all
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import Observation, Plan, PlanEvaluation, PlanSet
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.tools.ingest import ingest_to_parquet
from automl_agent.tools.splits import ensure_split
from automl_agent.verification import (
    RungReport,
    fidelity_schedule,
    rank_plans,
    successive_halving,
    verify_implementation,
    verify_request,
)

from .base import BaseAgent
from .context import RunContext
from .data_agent import DataAgent
from .model_agent import ModelAgent
from .operation_agent import OperationAgent
from .plan_analyst import PlanAnalyst
from .prompt_agent import PromptAgent


@dataclass
class PipelineResult:
    """Outcome of one pipeline run.

    Attributes:
        metrics: Contents of the best implementation's ``metrics.json`` (scored on the test split).
        observations: One row per (plan, fidelity) real run and per final run, for calibration analysis.
        fixes: Error->fix pairs from the Operation Agent's debug loop (fed to experience memory).
        stop_reason: Why the revision loop ended (``target_met``, ``max_revisions`` or a budget name).
    """

    success: bool
    task_spec: TaskSpec | None = None
    plan: Plan | None = None
    code: str | None = None
    metrics: dict[str, Any] | None = None
    target_met: bool = False
    error: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    fixes: list[dict[str, Any]] = field(default_factory=list)
    knowledge_sources: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    budget: dict[str, Any] | None = None


class AgentManager(BaseAgent):
    """Plans, delegates to the specialist agents, verifies and revises."""

    name = "manager"
    prompt_name = "manager"
    model_role = "smart"

    def __init__(
        self,
        ctx: RunContext,
        retrievers: list[Retriever] | None = None,
        hooks: list[PipelineHooks] | None = None,
    ) -> None:
        super().__init__(ctx)
        self.hooks = hooks if hooks is not None else self.default_hooks(ctx)
        self.retrievers = retrievers if retrievers is not None else self.default_retrievers(ctx)
        self.prompt_agent = PromptAgent(ctx)
        self.data_agent = DataAgent(ctx)
        self.model_agent = ModelAgent(ctx)
        self.operation_agent = OperationAgent(ctx)
        self.plan_analyst = PlanAnalyst(ctx)

    @staticmethod
    def default_hooks(ctx: RunContext) -> list[PipelineHooks]:
        """Extension hooks for this run: experience memory (when enabled) plus user-registered hooks."""
        builtin: list[PipelineHooks] = [MemoryHooks()] if ctx.settings.memory_enabled else []
        return [*builtin, *registered_hooks()]

    @staticmethod
    def default_retrievers(ctx: RunContext) -> list[Retriever]:
        """Local knowledge base, experience memory and Google Search grounding (when enabled)."""
        retrievers: list[Retriever] = [LocalKnowledgeRetriever()]
        if ctx.settings.memory_enabled:
            retrievers.append(MemoryRetriever(ctx.settings))
        if ctx.settings.search_grounding and ctx.llm.supports_search():
            retrievers.append(GeminiSearchRetriever(ctx.llm))
        return retrievers

    # ------------------------------------------------------------------ stages

    async def parse_and_verify(self) -> TaskSpec | None:
        """Prompt Agent parsing + request verification (one self-repair round)."""
        ctx = self.ctx
        await ctx.emit(Stage.parse, self.name, "Parsing the request into a task specification", kind="status")
        spec = await self.prompt_agent.parse()

        await ctx.emit(Stage.verify_request, self.name, "Verifying the task specification", kind="status")
        check = verify_request(spec, ctx.profile)
        if not check.ok:
            await ctx.emit(
                Stage.verify_request, self.name, f"Specification rejected: {check.feedback}", kind="warning"
            )
            spec = await self.prompt_agent.parse(feedback=check.feedback)
            check = verify_request(spec, ctx.profile)
            if not check.ok:
                await ctx.emit(
                    Stage.verify_request,
                    self.name,
                    f"Could not build a valid task: {check.feedback}",
                    kind="error",
                )
                return None
        spec = await run_hook("on_task_parsed", ctx, spec, hooks=self.hooks)
        for assumption in spec.assumptions:
            await ctx.emit(Stage.verify_request, self.name, f"Assumption: {assumption}", kind="info")
        await ctx.emit(
            Stage.verify_request,
            self.name,
            "Task specification verified",
            kind="artifact",
            payload={"task_spec": spec.model_dump(mode="json")},
        )
        return spec

    async def prepare_data(self, spec: TaskSpec) -> None:
        """Make sure the data is Parquet and create the fixed train/valid/test split for the target."""
        ctx = self.ctx
        await ctx.emit(Stage.prepare, self.name, "Preparing data splits", kind="status")
        if ctx.dataset_path.suffix.lower() not in {".parquet", ".pq"}:
            ctx.dataset_path = await asyncio.to_thread(
                ingest_to_parquet, ctx.dataset_path, ctx.workdir / "data" / "data.parquet"
            )
        ctx.split = await asyncio.to_thread(
            ensure_split,
            ctx.dataset_path,
            spec.target_column,
            stratify=spec.task_type != TaskType.tabular_regression,
            valid_fraction=ctx.settings.split_valid_fraction,
            test_fraction=ctx.settings.split_test_fraction,
            seed=ctx.settings.split_seed,
        )
        split = ctx.split
        await ctx.emit(
            Stage.prepare,
            self.name,
            f"Split {split.n_train:,} train / {split.n_valid:,} valid / {split.n_test:,} test rows "
            "(test rows are held out until the final evaluation)",
            kind="artifact",
            payload={"split": {"train": split.n_train, "valid": split.n_valid, "test": split.n_test}},
        )

    async def retrieve_knowledge(self, spec: TaskSpec) -> list[KnowledgeItem]:
        """Retrieval-augmented planning: gather knowledge from every configured retriever."""
        ctx = self.ctx
        await ctx.emit(Stage.retrieve, self.name, "Retrieving relevant ML knowledge", kind="status")
        items = await retrieve_all(self.retrievers, spec, ctx.profile, ctx.prompt)
        items = await run_hook("on_knowledge_retrieved", ctx, items, hooks=self.hooks)
        await ctx.emit(
            Stage.retrieve,
            self.name,
            f"Retrieved {len(items)} knowledge items",
            kind="artifact",
            payload={
                "knowledge": [
                    {"id": k.id, "title": k.title, "source": k.source, "urls": k.urls} for k in items
                ]
            },
        )
        return items

    async def generate_plans(
        self, spec: TaskSpec, knowledge: list[KnowledgeItem], revision: int, history: list[dict]
    ) -> list[Plan]:
        """Ask the planner for N diverse plans (with feedback from earlier rounds)."""
        ctx = self.ctx
        n = ctx.settings.n_plans
        await ctx.emit(
            Stage.plan, self.name, f"Generating {n} candidate plans (round {revision + 1})", kind="status"
        )
        context: dict[str, Any] = {
            "task_spec": spec.model_dump(mode="json"),
            "dataset_profile": ctx.profile.compact(),
            "knowledge": [
                {
                    "title": k.title,
                    "content": k.content,
                    "source": k.source,
                    **({"data": k.data} if k.data else {}),
                }
                for k in knowledge
            ],
            "allowed_models": supported_models(spec.task_type, ctx.train_rows),
            "n_plans": n,
            "revision": revision,
            "budget": ctx.budget.snapshot() if ctx.budget else None,
        }
        if history:
            context["feedback"] = history
        plan_set = await self.ask_json(Stage.plan, f"Devise {n} distinct end-to-end plans.", context, PlanSet)
        plans = [
            p.model_copy(
                update={
                    "id": f"r{revision + 1}p{i + 1}",
                    "model_family": normalize_model(spec.task_type, p.model_family, ctx.train_rows),
                }
            )
            for i, p in enumerate(plan_set.plans[:n])
        ]
        return await run_hook("on_plans_generated", ctx, plans, hooks=self.hooks)

    async def evaluate_plan(self, spec: TaskSpec, plan: Plan) -> PlanEvaluation:
        """Decompose a plan and pseudo-execute it with the Data and Model agents."""
        data_tasks, model_tasks = decompose(plan)
        if self.ctx.settings.agent_fusion:
            fused = await self.plan_analyst.execute(spec, plan, data_tasks, model_tasks)
            return PlanEvaluation(plan=plan, data=fused.data, model=fused.model)
        data, model = await asyncio.gather(
            self.data_agent.execute(spec, plan, data_tasks),
            self.model_agent.execute(spec, plan, model_tasks),
        )
        return PlanEvaluation(plan=plan, data=data, model=model)

    async def ground(
        self, spec: TaskSpec, evaluations: list[PlanEvaluation], revision: int
    ) -> list[PlanEvaluation]:
        """Grounded execution verification: successive halving on nested data subsamples."""
        ctx = self.ctx
        assert ctx.split is not None
        s = ctx.settings
        schedule = fidelity_schedule(
            ctx.split.n_train,
            len(evaluations),
            min_rows=s.grounding_min_rows,
            growth=s.grounding_growth,
            eta=s.grounding_eta,
        )
        rung_labels = [f"{r:,}" if r else "all" for r in schedule.rungs]
        await ctx.emit(
            Stage.ground,
            self.name,
            f"Grounding {len(evaluations)} plans with real runs at {' -> '.join(rung_labels)} training rows",
            kind="status",
            payload={
                "schedule": schedule.rungs,
                "n_train": ctx.split.n_train,
                "predicted": {ev.plan.id: ev.model.predicted_score for ev in evaluations},
            },
        )

        async def run(ev: PlanEvaluation, rows: int | None) -> Observation:
            code = render_template(
                spec,
                ev.plan.model_copy(update={"model_family": ev.model.model_family}),
                ctx.dataset_path,
                ctx.split.path,
                hyperparameters=ev.model.hyperparameters or ev.plan.hyperparameters,
                fidelity_rows=rows,
                eval_split="valid",
                eval_max_rows=s.grounding_valid_rows,
                save_model=False,
                n_jobs=s.exec_n_jobs,
                n_rows=ctx.train_rows,
            )
            remaining = ctx.budget.wall_remaining() if ctx.budget else None
            timeout = int(min(s.exec_timeout_s, remaining)) if remaining else s.exec_timeout_s
            workdir = ctx.workdir / f"ground_r{revision + 1}" / f"{ev.plan.id}_{rows or 'all'}"
            job = {"plan_id": ev.plan.id, "rows": rows, "job": "ground", "round": revision + 1}

            async def on_progress(item: dict[str, Any]) -> None:
                key = "telemetry" if item.get("telemetry") else "progress"
                message = (
                    "resources" if key == "telemetry" else f"{ev.plan.id}: {item.get('stage', 'progress')}"
                )
                await ctx.emit(Stage.ground, self.name, message, kind="telemetry", payload={key: item, **job})

            result = await run_script(
                code,
                workdir,
                timeout_s=max(timeout, 30),
                max_mem_mb=s.exec_max_mem_mb or None,
                on_progress=on_progress,
            )
            metrics = result.metrics or {}
            return Observation(
                fidelity_rows=int(metrics.get("n_train") or rows or ctx.split.n_train),
                score=metrics.get("score") if result.ok else None,
                ok=result.ok,
                duration_s=result.duration_s,
                error=None if result.ok else result.stderr.strip()[-300:],
            )

        async def on_rung(report: RungReport) -> None:
            if report.stopped_for_budget:
                await ctx.emit(
                    Stage.ground,
                    self.name,
                    "Stopping grounding early: next rung would exceed the time budget",
                    kind="warning",
                )
                return
            label = f"{report.rows:,}" if report.rows else "all"
            best = [r for r in report.results if r["ok"]]
            await ctx.emit(
                Stage.ground,
                self.name,
                f"Rung at {label} rows: "
                + ", ".join(
                    f"{r['plan_id']}={r['score']:.4f}" if r["ok"] else f"{r['plan_id']}=failed"
                    for r in report.results
                ),
                kind="artifact" if best else "warning",
                payload={"rung": {"rows": report.rows, "results": report.results}},
            )

        return await successive_halving(
            evaluations,
            spec,
            schedule,
            run,
            n_train=ctx.split.n_train,
            wall_remaining=ctx.budget.wall_remaining if ctx.budget else (lambda: None),
            on_rung=on_rung,
        )

    # ------------------------------------------------------------------ full pipeline

    def _record_observations(
        self, result: PipelineResult, ranked: list[PlanEvaluation], spec: TaskSpec, revision: int
    ) -> None:
        mode = self.ctx.settings.verification_mode
        for ev in ranked:
            base = {
                "revision": revision + 1,
                "plan_id": ev.plan.id,
                "model_family": ev.model.model_family,
                "metric": spec.metric,
                "higher_is_better": spec.higher_is_better,
                "predicted_score": ev.model.predicted_score,
                "predicted_train_time_s": ev.model.predicted_train_time_s,
                "rank": ev.rank,
                "verification_mode": mode,
            }
            if not ev.observations:
                result.observations.append({**base, "split": None, "final": False})
            for obs in ev.observations:
                result.observations.append(
                    {
                        **base,
                        "split": "valid",
                        "final": False,
                        "fidelity_rows": obs.fidelity_rows,
                        "observed_score": obs.score,
                        "ok": obs.ok,
                        "duration_s": obs.duration_s,
                    }
                )

    async def run(self) -> PipelineResult:
        """Execute the pipeline end to end and return its result."""
        ctx = self.ctx
        spec = await self.parse_and_verify()
        if spec is None:
            return PipelineResult(success=False, error="request verification failed")
        await self.prepare_data(spec)
        knowledge = await self.retrieve_knowledge(spec)

        result = PipelineResult(
            success=False, task_spec=spec, knowledge_sources=[k.source for k in knowledge]
        )
        best_score: float | None = None
        history: list[dict] = []
        grounded = ctx.settings.verification_mode == "grounded"

        for revision in range(ctx.settings.max_revisions + 1):
            if revision > 0 and ctx.budget and (spent := ctx.budget.exhausted()):
                result.stop_reason = f"{spent}_budget"
                await ctx.emit(
                    Stage.verify_impl, self.name, f"Stopping: {spent} budget exhausted", kind="warning"
                )
                break
            plans = await self.generate_plans(spec, knowledge, revision, history)

            await ctx.emit(
                Stage.execute_plans,
                self.name,
                f"{'Plan analyst' if ctx.settings.agent_fusion else 'Data & Model agents'} evaluating "
                f"{len(plans)} plans in parallel",
                kind="status",
            )
            evaluations = list(await asyncio.gather(*(self.evaluate_plan(spec, p) for p in plans)))
            ranked = (
                await self.ground(spec, evaluations, revision) if grounded else rank_plans(evaluations, spec)
            )
            ranked = await run_hook("on_plans_ranked", ctx, ranked, hooks=self.hooks)
            self._record_observations(result, ranked, spec, revision)
            chosen = ranked[0]
            why = "best observed validation score" if grounded else "best predicted score"
            await ctx.emit(
                Stage.select,
                self.name,
                f"Selected plan {chosen.plan.id}: {chosen.plan.title} ({why})",
                kind="artifact",
                payload={
                    "ranked": [ev.model_dump(mode="json") for ev in ranked],
                    "selected": chosen.plan.id,
                    "verification_mode": ctx.settings.verification_mode,
                },
            )

            await ctx.emit(
                Stage.implement, self.name, "Operation agent implementing the selected plan", kind="status"
            )
            outcome = await self.operation_agent.implement(
                spec, chosen, ctx.workdir / f"attempt_{revision + 1}", past_fixes=ctx.state.get("past_fixes")
            )
            code, exec_result = outcome.code, outcome.result
            result.fixes.extend(outcome.fixes)

            verdict = verify_implementation(spec, exec_result)
            score = (exec_result.metrics or {}).get("score") if exec_result.ok else None
            valid_score = ((exec_result.metrics or {}).get("metrics_valid") or {}).get(spec.metric)
            result.observations.append(
                {
                    "revision": revision + 1,
                    "plan_id": chosen.plan.id,
                    "model_family": chosen.model.model_family,
                    "metric": spec.metric,
                    "higher_is_better": spec.higher_is_better,
                    "predicted_score": chosen.model.predicted_score,
                    "predicted_train_time_s": chosen.model.predicted_train_time_s,
                    "rank": chosen.rank,
                    "verification_mode": ctx.settings.verification_mode,
                    "split": "test",
                    "final": True,
                    "fidelity_rows": (exec_result.metrics or {}).get("n_train"),
                    "observed_score": score,
                    "observed_valid_score": valid_score,
                    "ok": exec_result.ok,
                    "duration_s": exec_result.duration_s,
                    "train_time_s": (exec_result.metrics or {}).get("train_time_s"),
                }
            )
            history.append(
                {
                    "plan": chosen.plan.title,
                    "model_family": chosen.model.model_family,
                    "score": score,
                    "predicted_score": chosen.model.predicted_score,
                    "issues": verdict.issues,
                }
            )
            result.attempts.append(
                {
                    "revision": revision + 1,
                    "plan_id": chosen.plan.id,
                    "score": score,
                    "ok": exec_result.ok,
                    "issues": verdict.issues,
                    "debug_attempts": outcome.attempts,
                    "template_fallback": outcome.used_template_fallback,
                }
            )

            if isinstance(score, int | float) and (
                best_score is None or (score > best_score if spec.higher_is_better else score < best_score)
            ):
                best_score = score
                result.success = True
                result.plan = chosen.plan.model_copy(
                    update={
                        "model_family": chosen.model.model_family,
                        "hyperparameters": chosen.model.hyperparameters,
                    }
                )
                result.code = code
                result.metrics = {
                    **(exec_result.metrics or {}),
                    "artifact_dir": str(ctx.workdir / f"attempt_{revision + 1}"),
                }
                result.target_met = verdict.ok

            await ctx.emit(
                Stage.verify_impl,
                self.name,
                "Requirements met" if verdict.ok else f"Not yet: {verdict.feedback}",
                kind="artifact" if verdict.ok else "warning",
                payload={
                    "issues": verdict.issues,
                    "metrics": exec_result.metrics,
                    "budget": ctx.budget.snapshot() if ctx.budget else None,
                },
            )
            if verdict.ok:
                result.stop_reason = "target_met"
                break
            if revision < ctx.settings.max_revisions:
                await ctx.emit(Stage.verify_impl, self.name, "Revising plans with feedback", kind="status")
        else:
            result.stop_reason = result.stop_reason or "max_revisions"

        if not result.success:
            result.error = "no plan produced a working implementation"
        result.budget = ctx.budget.snapshot() if ctx.budget else None
        for hooks in self.hooks:
            await hooks.on_run_finished(ctx, result)
        return result
