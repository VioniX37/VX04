"""Model-agnostic LLM interface used by every agent.

Concrete clients (:class:`~automl_agent.llm.gemini.GeminiClient`,
:class:`~automl_agent.llm.fake.FakeLLM`) implement only :meth:`LLMClient._complete`.
Structured output, schema validation with self-repair, response caching and
token accounting are shared here.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .cache import ResponseCache

T = TypeVar("T", bound=BaseModel)


class Message(TypedDict):
    """A chat message in the provider-neutral format used by the agents."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """Raised when a model call fails permanently (after retries) or returns unusable output.

    Attributes:
        code: HTTP-style status code from the provider, when known.
    """

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class LLMResponse:
    """Text returned by a model plus the token counts used for accounting."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMUsage:
    """Running totals of model usage for one pipeline run."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens of all non-cached calls."""
        return self.input_tokens + self.output_tokens

    def record(self, resp: LLMResponse, *, cached: bool = False) -> None:
        """Add one response to the totals; cached responses cost no tokens."""
        if cached:
            self.cache_hits += 1
            return
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.by_model[resp.model] = self.by_model.get(resp.model, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage and the API."""
        return {**asdict(self), "total_tokens": self.total_tokens}


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | list:
    """Parse JSON from a model reply, tolerating code fences and surrounding prose.

    Raises:
        ValueError: If no valid JSON object or array can be found.
    """
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
    """Separate system messages (joined) from the chat turns."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


class LLMClient(ABC):
    """Base class for model clients.

    Attributes:
        provider: Short provider name reported in events and results.
        native_json_schema: Whether the backend enforces a JSON schema itself; if not,
            the schema is described in the prompt instead.
    """

    provider: str = "base"
    native_json_schema: bool = False

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        *,
        usage: LLMUsage | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.usage = usage or LLMUsage()
        self.cache = cache

    @abstractmethod
    async def _complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        json_mode: bool,
        schema: dict[str, Any] | None,
    ) -> LLMResponse:
        """Perform one model call. `schema` is a JSON schema when structured output is requested."""

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the model, serving from and filling the response cache when enabled."""
        t = self.temperature if temperature is None else temperature
        key = None
        if self.cache is not None:
            key = self.cache.key(self.model, messages, t, json_mode, schema)
            if (hit := self.cache.get(key)) is not None:
                self.usage.record(hit, cached=True)
                return hit
        resp = await self._complete(messages, temperature=t, json_mode=json_mode, schema=schema)
        self.usage.record(resp)
        if key is not None:
            self.cache.put(key, resp)
        return resp

    async def complete_json(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_repairs: int = 2,
    ) -> T:
        """Return a validated instance of `schema`.

        Uses the backend's native structured output when available, otherwise
        describes the schema in the prompt. Invalid output is fed back to the
        model for repair up to `max_repairs` times.

        Raises:
            LLMError: If no valid object is obtained.
        """
        json_schema = schema.model_json_schema()
        msgs: list[Message] = list(messages)
        if not self.native_json_schema:
            instruction = (
                "Respond with a single JSON object only (no prose, no code fences) that conforms to "
                f"this JSON schema:\n{json.dumps(json_schema)}"
            )
            msgs = [{"role": "system", "content": instruction}, *msgs]
        last_err: Exception | None = None
        for _ in range(max_repairs + 1):
            resp = await self.complete(msgs, temperature=temperature, json_mode=True, schema=json_schema)
            try:
                return schema.model_validate(extract_json(resp.text))
            except (ValueError, ValidationError) as e:
                last_err = e
                if self.cache is not None:  # never serve a known-bad answer again
                    t = self.temperature if temperature is None else temperature
                    self.cache.delete(self.cache.key(self.model, msgs, t, True, json_schema))
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": resp.text},
                    {
                        "role": "user",
                        "content": f"That output was invalid: {e}. Return the corrected JSON object only.",
                    },
                ]
        raise LLMError(f"{self.provider}: could not obtain valid {schema.__name__}: {last_err}")
