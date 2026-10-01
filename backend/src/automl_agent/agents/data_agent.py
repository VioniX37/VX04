"""Data Agent: pseudo-executes the data-related sub-tasks of a plan (paper section 3.3)."""

from __future__ import annotations

from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import DataAgentResult, Plan, SubTask
from automl_agent.schemas.task_spec import TaskSpec

from .base import BaseAgent


class DataAgent(BaseAgent):
    """Pseudo-executes the data-related sub-tasks of a plan."""

    name = "data_agent"
    prompt_name = "data_agent"

    async def execute(self, spec: TaskSpec, plan: Plan, subtasks: list[SubTask]) -> DataAgentResult:
        """Return the refined data steps and risks for `plan`."""
        return await self.ask_json(
            Stage.execute_plans,
            f"Carry out the data sub-tasks for plan {plan.id}.",
            {
                "task_spec": spec.model_dump(mode="json"),
                "dataset_profile": self.ctx.profile.compact(),
                "plan": plan.model_dump(mode="json"),
                "subtasks": [s.instruction for s in subtasks],
            },
            DataAgentResult,
        )
