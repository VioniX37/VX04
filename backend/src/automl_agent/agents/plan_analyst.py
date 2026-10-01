"""Fused Data + Model analysis for one plan (``AGENT_FUSION=true``).

The paper runs the Data and Model agents as two calls per plan. On the
Gemini free tier that is the dominant cost, so this agent performs both
roles in one structured call. It is an explicit experimental dimension:
compare runs with fusion on and off to measure the quality/cost trade-off.
"""

from __future__ import annotations

from automl_agent.execution.model_registry import normalize_model, supported_models
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import Plan, PlanAnalysis, SubTask
from automl_agent.schemas.task_spec import TaskSpec

from .base import BaseAgent


class PlanAnalyst(BaseAgent):
    """Pseudo-executes both the data and the model sub-tasks of a plan in one call."""

    name = "plan_analyst"
    prompt_name = "plan_analyst"
    model_role = "fast"

    async def execute(
        self, spec: TaskSpec, plan: Plan, data_tasks: list[SubTask], model_tasks: list[SubTask]
    ) -> PlanAnalysis:
        """Return the fused data/model analysis for `plan`."""
        result = await self.ask_json(
            Stage.execute_plans,
            f"Carry out the data and model sub-tasks for plan {plan.id}.",
            {
                "task_spec": spec.model_dump(mode="json"),
                "dataset_profile": self.ctx.profile.compact(),
                "plan": plan.model_dump(mode="json"),
                "data_subtasks": [s.instruction for s in data_tasks],
                "model_subtasks": [s.instruction for s in model_tasks],
                "allowed_models": supported_models(spec.task_type, self.ctx.train_rows),
            },
            PlanAnalysis,
        )
        result.model.model_family = normalize_model(
            spec.task_type, result.model.model_family, self.ctx.train_rows
        )
        return result
