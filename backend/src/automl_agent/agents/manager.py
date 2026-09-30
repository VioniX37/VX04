"""Agent Manager: orchestrates the full AutoML-Agent pipeline (paper Fig. 2)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from automl_agent.execution.model_registry import normalize_model, supported_models
from automl_agent.extensions import registered_hooks, run_hook
from automl_agent.planning.decomposition import decompose
from automl_agent.planning.gemini_search import GeminiSearchRetriever
from automl_agent.planning.retrieval import KnowledgeItem, LocalKnowledgeRetriever, Retriever, retrieve_all
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import Plan, PlanEvaluation, PlanSet
from automl_agent.schemas.task_spec import TaskSpec
from automl_agent.verification import rank_plans, verify_implementation, verify_request

from .base import BaseAgent
from .context import RunContext
from .data_agent import DataAgent
from .model_agent import ModelAgent
from .operation_agent import OperationAgent
from .plan_analyst import PlanAnalyst
from .prompt_agent import PromptAgent


@dataclass
class PipelineResult:
    success: bool
    task_spec: TaskSpec | None = None
    plan: Plan | None = None
    code: str | None = None
    metrics: dict[str, Any] | None = None
    target_met: bool = False
    error: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)


class AgentManager(BaseAgent):
    name = "manager"
    prompt_name = "manager"
    model_role = "smart"

    def __init__(self, ctx: RunContext, retrievers: list[Retriever] | None = None) -> None:
        super().__init__(ctx)
        self.retrievers = retrievers if retrievers is not None else self.default_retrievers(ctx)
        self.prompt_agent = PromptAgent(ctx)
        self.data_agent = DataAgent(ctx)
        self.model_agent = ModelAgent(ctx)
        self.operation_agent = OperationAgent(ctx)
        self.plan_analyst = PlanAnalyst(ctx)

    @staticmethod
    def default_retrievers(ctx: RunContext) -> list[Retriever]:
        """Local knowledge base, plus Google Search grounding when enabled and supported."""
        retrievers: list[Retriever] = [LocalKnowledgeRetriever()]
        if ctx.settings.search_grounding and ctx.llm.supports_search():
            retrievers.append(GeminiSearchRetriever(ctx.llm))
        return retrievers

    # ------------------------------------------------------------------ stages

    async def parse_and_verify(self) -> TaskSpec | None:
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
        spec = await run_hook("on_task_parsed", ctx, spec)
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

    async def retrieve_knowledge(self, spec: TaskSpec) -> list[KnowledgeItem]:
        ctx = self.ctx
        await ctx.emit(Stage.retrieve, self.name, "Retrieving relevant ML knowledge", kind="status")
        items = await retrieve_all(self.retrievers, spec, ctx.profile, ctx.prompt)
        items = await run_hook("on_knowledge_retrieved", ctx, items)
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
        ctx = self.ctx
        n = ctx.settings.n_plans
        await ctx.emit(
            Stage.plan, self.name, f"Generating {n} candidate plans (round {revision + 1})", kind="status"
        )
        context: dict[str, Any] = {
            "task_spec": spec.model_dump(mode="json"),
            "dataset_profile": ctx.profile.compact(),
            "knowledge": [{"title": k.title, "content": k.content} for k in knowledge],
            "allowed_models": supported_models(spec.task_type),
            "n_plans": n,
            "revision": revision,
        }
        if history:
            context["feedback"] = history
        plan_set = await self.ask_json(Stage.plan, f"Devise {n} distinct end-to-end plans.", context, PlanSet)
        plans = [
            p.model_copy(
                update={
                    "id": f"r{revision + 1}p{i + 1}",
                    "model_family": normalize_model(spec.task_type, p.model_family),
                }
            )
            for i, p in enumerate(plan_set.plans[:n])
        ]
        return await run_hook("on_plans_generated", ctx, plans)

    async def evaluate_plan(self, spec: TaskSpec, plan: Plan) -> PlanEvaluation:
        data_tasks, model_tasks = decompose(plan)
        if self.ctx.settings.agent_fusion:
            fused = await self.plan_analyst.execute(spec, plan, data_tasks, model_tasks)
            return PlanEvaluation(plan=plan, data=fused.data, model=fused.model)
        data, model = await asyncio.gather(
            self.data_agent.execute(spec, plan, data_tasks),
            self.model_agent.execute(spec, plan, model_tasks),
        )
        return PlanEvaluation(plan=plan, data=data, model=model)

    # ------------------------------------------------------------------ full pipeline

    async def run(self) -> PipelineResult:
        ctx = self.ctx
        spec = await self.parse_and_verify()
        if spec is None:
            return PipelineResult(success=False, error="request verification failed")
        knowledge = await self.retrieve_knowledge(spec)

        result = PipelineResult(success=False, task_spec=spec)
        best_score: float | None = None
        history: list[dict] = []

        for revision in range(ctx.settings.max_revisions + 1):
            plans = await self.generate_plans(spec, knowledge, revision, history)

            await ctx.emit(
                Stage.execute_plans,
                self.name,
                f"Data & Model agents evaluating {len(plans)} plans in parallel",
                kind="status",
            )
            evaluations = await asyncio.gather(*(self.evaluate_plan(spec, p) for p in plans))
            ranked = await run_hook("on_plans_ranked", ctx, rank_plans(list(evaluations), spec))
            chosen = ranked[0]
            await ctx.emit(
                Stage.select,
                self.name,
                f"Selected plan {chosen.plan.id}: {chosen.plan.title}",
                kind="artifact",
                payload={"ranked": [ev.model_dump(mode="json") for ev in ranked], "selected": chosen.plan.id},
            )

            await ctx.emit(
                Stage.implement, self.name, "Operation agent implementing the selected plan", kind="status"
            )
            code, exec_result = await self.operation_agent.implement(
                spec, chosen, ctx.workdir / f"attempt_{revision + 1}"
            )

            verdict = verify_implementation(spec, exec_result)
            score = (exec_result.metrics or {}).get("score") if exec_result.ok else None
            history.append(
                {
                    "plan": chosen.plan.title,
                    "model_family": chosen.model.model_family,
                    "score": score,
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
                payload={"issues": verdict.issues, "metrics": exec_result.metrics},
            )
            if verdict.ok:
                break
            if revision < ctx.settings.max_revisions:
                await ctx.emit(Stage.verify_impl, self.name, "Revising plans with feedback", kind="status")

        if not result.success:
            result.error = "no plan produced a working implementation"
        for hooks in registered_hooks():
            await hooks.on_run_finished(ctx, result)
        return result
