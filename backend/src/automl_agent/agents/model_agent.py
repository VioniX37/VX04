from __future__ import annotations

from automl_agent.execution.model_registry import normalize_model, supported_models
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import ModelAgentResult, Plan, SubTask
from automl_agent.schemas.task_spec import TaskSpec

from .base import BaseAgent


class ModelAgent(BaseAgent):
    """Pseudo-executes the model-related sub-tasks of a plan and predicts its performance."""

    name = "model_agent"
    prompt_name = "model_agent"

    async def execute(self, spec: TaskSpec, plan: Plan, subtasks: list[SubTask]) -> ModelAgentResult:
        result = await self.ask_json(
            Stage.execute_plans,
            f"Carry out the model sub-tasks for plan {plan.id}.",
            {
                "task_spec": spec.model_dump(mode="json"),
                "dataset_profile": self.ctx.profile.compact(),
                "plan": plan.model_dump(mode="json"),
                "subtasks": [s.instruction for s in subtasks],
                "allowed_models": supported_models(spec.task_type, self.ctx.train_rows),
            },
            ModelAgentResult,
        )
        result.model_family = normalize_model(spec.task_type, result.model_family, self.ctx.train_rows)
        return result
