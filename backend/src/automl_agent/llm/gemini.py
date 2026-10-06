"""Gemini client built on the official `google-genai` SDK.

Adds what the free tier needs on top of plain calls: per-model rate limiting,
a global concurrency cap, retries with exponential backoff (honouring the
server's ``retryDelay``), native JSON-schema output, and Google Search
grounding for knowledge retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .base import LLMClient, LLMError, LLMResponse, Message, split_system
from .cache import ResponseCache
from .rate_limit import concurrency_slot, get_limiter

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
# Errors that are specific to one model (overload, per-model quota): worth trying another model.
FALLBACK_CODES = {408, 429, 500, 503, 504}
# Attempts on a model before moving to the next one in the fallback chain.
ATTEMPTS_BEFORE_FALLBACK = 3
# Waits between rounds through the whole chain while every model is busy (capped by overload_wait_s).
ROUND_WAIT_S = (15.0, 30.0, 60.0)
# After a search-grounding quota error, skip web search for this long (it is optional knowledge).
SEARCH_COOLDOWN_S = 600.0
log = logging.getLogger(__name__)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


@dataclass
class SearchResult:
    """Answer text from a grounded search plus the web sources it cites."""

    text: str
    sources: list[dict[str, str]] = field(default_factory=list)


def retry_delay_from_error(err: Exception) -> float | None:
    """Extract the server-suggested retry delay (seconds) from an API error, if present."""
    details = getattr(err, "details", None)
    match = _RETRY_DELAY.search(json.dumps(details) if details is not None else str(err))
    return float(match.group(1)) if match else None


class GeminiClient(LLMClient):
    """One Gemini model with throttling, retries and structured output.

    Args:
        model: Gemini model id, e.g. ``gemini-3.8-flash``.
        api_key: AI Studio key (ignored in Vertex AI mode).
        rpm: Requests-per-minute budget for this model.
        max_concurrency: Global cap on simultaneous requests.
        max_retries: Upper bound on attempts per model in one round (``ATTEMPTS_BEFORE_FALLBACK`` is used
            when it is smaller).
        fallback_models: Models tried in order when this one is overloaded (503) or out of quota (429).
        overload_wait_s: How long to keep cycling through the chain, waiting between rounds, while
            every model is overloaded, before giving up. Demand spikes are usually short.
        vertexai: Use Vertex AI (requires `project` and `location`).
    """

    provider = "gemini"
    native_json_schema = True
    _search_disabled_until: float = 0.0  # shared by all clients in the process (quota is per key)

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        temperature: float = 0.2,
        rpm: int = 10,
        max_concurrency: int = 2,
        max_retries: int = 6,
        fallback_models: list[str] | None = None,
        overload_wait_s: float = 600.0,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        cache: ResponseCache | None = None,
        usage=None,
        client: Any = None,
    ) -> None:
        super().__init__(model, temperature, usage=usage, cache=cache)
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        if client is not None:  # injected in tests
            self._client = client
        else:
            from google import genai

            self._client = (
                genai.Client(vertexai=True, project=project, location=location)
                if vertexai
                else genai.Client(api_key=api_key)
            )
        self.rpm = rpm
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.overload_wait_s = overload_wait_s
        self._schema_supported = True

    # ------------------------------------------------------------------ transport

    async def _generate_with(self, model: str, contents: list[Any], config: Any, attempts: int) -> Any:
        """Call one model with throttling and up to `attempts` tries on retryable errors."""
        limiter = get_limiter(model, self.rpm)
        for attempt in range(attempts):
            await limiter.acquire()
            try:
                async with concurrency_slot(self.max_concurrency):
                    return await asyncio.wait_for(
                        self._client.aio.models.generate_content(
                            model=model, contents=contents, config=config
                        ),
                        timeout=180.0,
                    )
            except TimeoutError as timeout_err:
                if attempt < attempts - 1:
                    await asyncio.sleep(min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1))
                    continue
                raise LLMError(f"gemini {model}: request timed out after 180s", code=408) from timeout_err
            except (httpx.TransportError, ConnectionError) as net_err:
                # Dropped connections and read errors are transient: retry, then let the caller
                # treat the model as unavailable (503) so it falls back or waits.
                if attempt < attempts - 1:
                    await asyncio.sleep(min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1))
                    continue
                raise LLMError(
                    f"gemini {model}: network error {type(net_err).__name__}", code=503
                ) from net_err
            except self._errors.APIError as e:
                server_delay = retry_delay_from_error(e)
                # If server requests a delay > 60s (e.g. daily quota reset in hours),
                # do not sleep for hours. Immediately raise LLMError to trigger fallback.
                if server_delay is not None and server_delay > 60.0:
                    err_msg = (
                        f"gemini {model}: {e.code} {e.message or e} "
                        f"(quota retryDelay {server_delay}s exceeds 60s cap)"
                    )
                    raise LLMError(err_msg, code=e.code) from e

                if e.code in RETRYABLE_CODES and attempt < attempts - 1:
                    delay = min(60.0, server_delay or (2.0 * 2**attempt))
                    await asyncio.sleep(delay + random.uniform(0, 1))
                    continue
                raise LLMError(f"gemini {model}: {e.code} {e.message or e}", code=e.code) from e
        raise LLMError(f"gemini {model}: retries exhausted")  # pragma: no cover

    async def _generate(self, contents: list[Any], config: Any, *, quick: bool = False) -> tuple[Any, str]:
        """Call `generate_content`, falling back along the model chain; returns (response, model used).

        With ``quick=True`` (optional calls such as web search) only one attempt is made on the
        configured model, so a quota error costs seconds instead of minutes.
        """
        if quick:
            return await self._generate_with(self.model, contents, config, 1), self.model
        chain = [self.model, *self.fallback_models]
        attempts = min(ATTEMPTS_BEFORE_FALLBACK, self.max_retries + 1)
        deadline = time.monotonic() + self.overload_wait_s
        for round_no in range(10_000):
            last_error: LLMError | None = None
            for i, model in enumerate(chain):
                try:
                    return await self._generate_with(model, contents, config, attempts), model
                except LLMError as e:
                    if e.code not in FALLBACK_CODES:
                        raise
                    last_error = e
                    if i < len(chain) - 1:
                        log.warning("%s unavailable (%s); falling back to %s", model, e.code, chain[i + 1])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                assert last_error is not None
                raise last_error
            wait = min(ROUND_WAIT_S[min(round_no, len(ROUND_WAIT_S) - 1)], remaining)
            log.warning(
                "all %d Gemini models busy (last error %s); retrying in %.0fs (giving up in %.0fs)",
                len(chain), last_error.code if last_error else "?", wait, remaining,
            )  # fmt: skip
            await asyncio.sleep(wait)
        raise LLMError("gemini: no model available")  # pragma: no cover

    def _contents(self, chat: list[Message]) -> list[Any]:
        types = self._types
        return [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in chat
        ]

    @staticmethod
    def _text(resp: Any) -> str:
        text = resp.text
        if text:
            return text
        reason = None
        if getattr(resp, "candidates", None):
            reason = resp.candidates[0].finish_reason
        raise LLMError(f"gemini returned no text (finish_reason={reason})")

    # ------------------------------------------------------------------ LLMClient API

    async def _complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        json_mode: bool,
        schema: dict[str, Any] | None,
    ) -> LLMResponse:
        system, chat = split_system(messages)
        use_schema = bool(schema) and self._schema_supported
        config = self._types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            response_json_schema=schema if use_schema else None,
            automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            resp, used = await self._generate(self._contents(chat), config)
        except LLMError as e:
            # Some JSON-schema constructs are rejected by some models: fall back to prompt-described schema.
            if use_schema and e.code == 400:
                self._schema_supported = False
                described = [
                    {
                        "role": "system",
                        "content": f"Reply with JSON matching this schema:\n{json.dumps(schema)}",
                    },
                    *messages,
                ]
                return await self._complete(
                    described, temperature=temperature, json_mode=json_mode, schema=None
                )
            raise
        meta = resp.usage_metadata
        return LLMResponse(
            text=self._text(resp),
            model=used,
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
        )

    # ------------------------------------------------------------------ extras

    async def search(self, query: str) -> SearchResult:
        """Answer `query` with Google Search grounding and return the cited web sources."""
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.key(
                self.model, [{"role": "user", "content": query}], 0.0, False, None, extra="google_search"
            )
            if (hit := self.cache.get(cache_key)) is not None:
                self.usage.record(hit, cached=True)
                payload = json.loads(hit.text)
                return SearchResult(payload["text"], payload["sources"])

        if time.monotonic() < GeminiClient._search_disabled_until:
            raise LLMError("google search grounding paused after a quota error", code=429)
        types = self._types
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            resp, used = await self._generate([query], config, quick=True)
        except LLMError as e:
            if e.code == 429:
                GeminiClient._search_disabled_until = time.monotonic() + SEARCH_COOLDOWN_S
            raise
        sources: list[dict[str, str]] = []
        candidates = getattr(resp, "candidates", None) or []
        meta = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        for chunk in (getattr(meta, "grounding_chunks", None) or []) if meta else []:
            web = getattr(chunk, "web", None)
            if web is not None and getattr(web, "uri", None):
                sources.append({"title": web.title or web.uri, "url": web.uri})
        result = SearchResult(self._text(resp), sources)

        usage_meta = resp.usage_metadata
        record = LLMResponse(
            text=json.dumps({"text": result.text, "sources": result.sources}),
            model=used,
            input_tokens=(usage_meta.prompt_token_count or 0) if usage_meta else 0,
            output_tokens=(usage_meta.candidates_token_count or 0) if usage_meta else 0,
        )
        self.usage.record(record)
        if cache_key is not None:
            self.cache.put(cache_key, record)
        return result

    async def list_models(self) -> list[str]:
        """Return the model ids visible to this API key (used to validate configuration)."""
        names: list[str] = []
        pager = await self._client.aio.models.list()
        async for m in pager:
            names.append(m.name.removeprefix("models/"))
        return names
