from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from automl_agent.schemas.events import Stage

from .context import RunContext

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def render_context(data: dict[str, Any]) -> str:
    """Serialize agent inputs into a tagged JSON block (also parsed by the offline FakeLLM)."""
    return f"<context>\n{json.dumps(data, indent=2, default=str)}\n</context>"


class BaseAgent:
    """An LLM-backed role. Subclasses set `name` and `prompt_name` (file in prompts/)."""

    name: str = "agent"
    prompt_name: str = ""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    @property
    def system_prompt(self) -> str:
        return load_prompt(self.prompt_name)

    async def ask_json(self, stage: Stage, task: str, context: dict[str, Any], schema: type[T]) -> T:
        """Send role prompt + task + context, get back a validated `schema` instance."""
        user = f"{task}\n\n{render_context({'expected': schema.__name__, **context})}"
        result = await self.ctx.llm.complete_json(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}],
            schema,
        )
        await self.ctx.emit(
            stage,
            self.name,
            f"{self.name} produced {schema.__name__}",
            kind="llm",
            payload={"output": result.model_dump(mode="json")},
        )
        return result
