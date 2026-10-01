"""Base class shared by all agents, and helpers for prompts and context blocks."""

from __future__ import annotations

import json
import re
import time
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from automl_agent.config import ModelRole
from automl_agent.schemas.events import Stage

from .context import RunContext

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    """Load a role prompt, stripping its leading HTML comment header (documentation only)."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return re.sub(r"\A\s*<!--.*?-->", "", text, flags=re.DOTALL).strip()


def render_context(data: dict[str, Any]) -> str:
    """Serialize agent inputs into a tagged JSON block (also parsed by the offline FakeLLM)."""
    return f"<context>\n{json.dumps(data, indent=2, default=str)}\n</context>"


class BaseAgent:
    """An LLM-backed role.

    Subclasses set ``name`` (shown in events), ``prompt_name`` (a file in ``prompts/``)
    and ``model_role`` (``smart`` for reasoning-heavy roles, ``fast`` otherwise).
    """

    name: str = "agent"
    prompt_name: str = ""
    model_role: ModelRole = "fast"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    @property
    def system_prompt(self) -> str:
        """The role prompt sent as the system message."""
        return load_prompt(self.prompt_name)

    async def ask_json(self, stage: Stage, task: str, context: dict[str, Any], schema: type[T]) -> T:
        """Send role prompt + task + context, get back a validated `schema` instance."""
        user = f"{task}\n\n{render_context({'expected': schema.__name__, **context})}"
        client = self.ctx.llm.for_role(self.model_role)
        usage = client.usage
        before_models = dict(usage.by_model)
        before = (usage.calls, usage.cache_hits, usage.input_tokens, usage.output_tokens)
        started = time.perf_counter()
        result = await client.complete_json(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}],
            schema,
        )
        # Per-call metadata powers the UI's agent timeline and LLM activity charts.
        meta = {
            "schema": schema.__name__,
            "role": self.model_role,
            "requested_model": client.model,
            "models": [m for m, n in usage.by_model.items() if n > before_models.get(m, 0)],
            "calls": usage.calls - before[0],
            "cache_hits": usage.cache_hits - before[1],
            "input_tokens": usage.input_tokens - before[2],
            "output_tokens": usage.output_tokens - before[3],
            "duration_s": round(time.perf_counter() - started, 3),
        }
        await self.ctx.emit(
            stage,
            self.name,
            f"{self.name} produced {schema.__name__}",
            kind="llm",
            payload={"output": result.model_dump(mode="json"), "meta": meta},
        )
        return result
