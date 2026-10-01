"""Pipeline hooks that feed and use experience memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from automl_agent.extensions import PipelineHooks
from automl_agent.schemas.task_spec import TaskSpec
from automl_agent.storage.db import ExperienceRecord

from .meta_features import dataset_fingerprint, meta_features
from .store import ExperienceStore

if TYPE_CHECKING:
    from automl_agent.agents.context import RunContext
    from automl_agent.agents.manager import PipelineResult


class MemoryHooks(PipelineHooks):
    """Loads relevant past fixes before a run and stores the run's experience afterwards."""

    async def on_task_parsed(self, ctx: RunContext, spec: TaskSpec) -> TaskSpec:
        """Give the Operation Agent the error->fix pairs learned on other runs."""
        store = ExperienceStore(ctx.settings)
        exclude = dataset_fingerprint(ctx.profile) if ctx.settings.memory_exclude_same_dataset else None
        ctx.state["past_fixes"] = store.fixes(exclude_fingerprint=exclude)
        return spec

    async def on_run_finished(self, ctx: RunContext, result: PipelineResult) -> None:
        """Store the run's best plan, all observed plans and learned fixes."""
        spec = result.task_spec
        if spec is None:
            return
        latest: dict[str, dict] = {}
        for row in result.observations:
            if row.get("final"):
                continue
            latest[row["plan_id"]] = row  # rows are in fidelity order, so the last one is the highest rung
        plans = [
            {
                "plan_id": pid,
                "model_family": row["model_family"],
                "predicted_score": row.get("predicted_score"),
                "observed_score": row.get("observed_score"),
                "fidelity_rows": row.get("fidelity_rows"),
                "ok": row.get("ok"),
            }
            for pid, row in latest.items()
        ]
        record = ExperienceRecord(
            run_id=ctx.run_id,
            dataset_fingerprint=dataset_fingerprint(ctx.profile),
            dataset_name=ctx.dataset_id,
            task_type=spec.task_type.value,
            metric=spec.metric,
            higher_is_better=spec.higher_is_better,
            n_rows=ctx.profile.n_rows,
            success=result.success,
            target_met=result.target_met,
            best_score=(result.metrics or {}).get("score"),
            meta=meta_features(ctx.profile, spec),
            best_plan=result.plan.model_dump(mode="json") if result.plan else None,
            plans=plans,
            fixes=result.fixes,
        )
        ExperienceStore(ctx.settings).add(record)
