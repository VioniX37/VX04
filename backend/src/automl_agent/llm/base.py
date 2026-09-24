"""Provider-agnostic LLM interface.

Every agent talks to an `LLMClient`. Concrete adapters (OpenAI-compatible,
Anthropic, Gemini, Fake) only implement `_complete`; JSON-mode parsing,
schema validation with self-repair, and token accounting live here.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.by_model[resp.model] = self.by_model.get(resp.model, 0) + 1

    def to_dict(self) -> dict:
        return asdict(self)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | list:
    """Parse JSON from a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    candidates = [text, *(m.strip() for m in _JSON_FENCE.findall(text))]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON object found in model output")


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


class LLMClient(ABC):
    provider: str = "base"

    def __init__(self, model: str, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature
        self.usage = LLMUsage()

    @abstractmethod
    async def _complete(
        self, messages: list[Message], *, temperature: float, json_mode: bool
    ) -> LLMResponse: ...

    async def complete(
        self, messages: list[Message], *, temperature: float | None = None, json_mode: bool = False
    ) -> LLMResponse:
        t = self.temperature if temperature is None else temperature
        resp = await self._complete(messages, temperature=t, json_mode=json_mode)
        self.usage.record(resp)
        return resp

    async def complete_json(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_repairs: int = 2,
    ) -> T:
        """Ask for JSON matching `schema`; on parse/validation failure, feed the error back and retry."""
        instruction = (
            "Respond with a single JSON object only (no prose, no code fences) that conforms to this "
            f"JSON schema:\n{json.dumps(schema.model_json_schema())}"
        )
        msgs: list[Message] = [{"role": "system", "content": instruction}, *messages]
        last_err: Exception | None = None
        for _ in range(max_repairs + 1):
            resp = await self.complete(msgs, temperature=temperature, json_mode=True)
            try:
                return schema.model_validate(extract_json(resp.text))
            except (ValueError, ValidationError) as e:
                last_err = e
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": resp.text},
                    {
                        "role": "user",
                        "content": f"That output was invalid: {e}. Return the corrected JSON object only.",
                    },
                ]
        raise LLMError(f"{self.provider}: could not obtain valid {schema.__name__}: {last_err}")
