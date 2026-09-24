"""Adapter for any OpenAI-compatible Chat Completions endpoint.

Covers OpenAI itself plus Groq, Ollama (`/v1`), vLLM, LM Studio, etc. via `base_url`.
"""

from __future__ import annotations

from .base import LLMClient, LLMError, LLMResponse, Message


class OpenAICompatClient(LLMClient):
    provider = "openai_compatible"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None = None,
        temperature: float = 0.2,
        provider: str = "openai_compatible",
    ) -> None:
        super().__init__(model, temperature)
        from openai import AsyncOpenAI

        self.provider = provider
        self._client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)

    async def _complete(self, messages: list[Message], *, temperature: float, json_mode: bool) -> LLMResponse:
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:  # surface provider errors uniformly to the agents
            raise LLMError(f"{self.provider} request failed: {e}") from e
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model or self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
