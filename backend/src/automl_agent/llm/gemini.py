"""Adapter for Google Gemini via the `google-genai` SDK."""

from __future__ import annotations

from .base import LLMClient, LLMError, LLMResponse, Message, split_system


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, *, api_key: str | None, temperature: float = 0.2) -> None:
        super().__init__(model, temperature)
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)

    async def _complete(self, messages: list[Message], *, temperature: float, json_mode: bool) -> LLMResponse:
        types = self._types
        system, chat = split_system(messages)
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])]
            )
            for m in chat
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as e:
            raise LLMError(f"gemini request failed: {e}") from e
        meta = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            model=self.model,
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
        )
