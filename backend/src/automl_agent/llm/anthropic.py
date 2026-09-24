"""Adapter for Claude via the official Anthropic SDK."""

from __future__ import annotations

from .base import LLMClient, LLMError, LLMResponse, Message, split_system


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, *, api_key: str | None, temperature: float = 0.2) -> None:
        super().__init__(model, temperature)
        import anthropic

        self._anthropic = anthropic
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an `ant auth login` profile.
        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    async def _complete(self, messages: list[Message], *, temperature: float, json_mode: bool) -> LLMResponse:
        # Current Claude models don't accept sampling params (temperature), so it is not sent.
        # JSON mode is enforced by the instruction added in LLMClient.complete_json.
        system, chat = split_system(messages)
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system or self._anthropic.NOT_GIVEN,
                messages=[{"role": m["role"], "content": m["content"]} for m in chat],
            )
        except self._anthropic.APIError as e:
            raise LLMError(f"anthropic request failed: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("anthropic: the model declined this request")
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
