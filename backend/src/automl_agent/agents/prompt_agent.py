from __future__ import annotations

from automl_agent.schemas.events import Stage
from automl_agent.schemas.task_spec import TaskSpec

from .base import BaseAgent


class PromptAgent(BaseAgent):
    """Parses the user's natural-language request into a structured TaskSpec."""

    name = "prompt_agent"
    prompt_name = "prompt_agent"

    async def parse(self, feedback: str | None = None) -> TaskSpec:
        context = {"user_prompt": self.ctx.prompt, "dataset_profile": self.ctx.profile.compact()}
        task = "Convert the user's request into a task specification."
        if feedback:
            task += f"\nYour previous specification was rejected: {feedback}. Fix these issues."
            context["previous_issues"] = feedback
        return await self.ask_json(Stage.parse, task, context, TaskSpec)
