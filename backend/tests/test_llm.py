import asyncio
import time
from types import SimpleNamespace

import pytest
from google.genai import errors

from automl_agent.config import Settings
from automl_agent.llm import LLMError, create_llm, extract_json
from automl_agent.llm.base import LLMClient, LLMResponse
from automl_agent.llm.cache import ResponseCache
from automl_agent.llm.gemini import GeminiClient, retry_delay_from_error
from automl_agent.llm.rate_limit import RateLimiter
from automl_agent.schemas.plan import CodeDraft


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('noise {"a": 3} trailing') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")


class ScriptedLLM(LLMClient):
    provider = "scripted"

    def __init__(self, replies, cache=None):
        super().__init__("scripted", cache=cache)
        self.replies = list(replies)
        self.seen = []

    async def _complete(self, messages, *, temperature, json_mode, schema):
        self.seen.append(messages)
        return LLMResponse(text=self.replies.pop(0), model=self.model, input_tokens=10, output_tokens=5)


def test_complete_json_repairs_invalid_output():
    llm = ScriptedLLM(["not json", '{"code": "print(1)"}'])
    draft = asyncio.run(llm.complete_json([{"role": "user", "content": "go"}], CodeDraft))
    assert draft.code == "print(1)"
    assert llm.usage.calls == 2 and llm.usage.input_tokens == 20
    assert "invalid" in llm.seen[1][-1]["content"]


def test_complete_json_gives_up():
    llm = ScriptedLLM(["x", "y", "z"])
    with pytest.raises(LLMError):
        asyncio.run(llm.complete_json([{"role": "user", "content": "go"}], CodeDraft, max_repairs=2))


def test_cache_serves_repeat_requests(tmp_path):
    cache = ResponseCache(tmp_path)
    llm = ScriptedLLM(['{"code": "a"}'], cache=cache)
    msgs = [{"role": "user", "content": "same"}]
    first = asyncio.run(llm.complete_json(msgs, CodeDraft))
    second = asyncio.run(llm.complete_json(msgs, CodeDraft))  # no scripted reply left: must be cached
    assert first == second
    assert llm.usage.calls == 1 and llm.usage.cache_hits == 1


def test_cache_drops_invalid_answers(tmp_path):
    cache = ResponseCache(tmp_path)
    llm = ScriptedLLM(["garbage", '{"code": "ok"}'], cache=cache)
    msgs = [{"role": "user", "content": "q"}]
    asyncio.run(llm.complete_json(msgs, CodeDraft))
    # The invalid first answer must not be replayed on the next identical request.
    retry = ScriptedLLM(['{"code": "fresh"}'], cache=cache)
    assert asyncio.run(retry.complete_json(msgs, CodeDraft)).code == "fresh"


def test_rate_limiter_enforces_window():
    async def burst():
        limiter = RateLimiter(rpm=3, window_s=0.5)
        start = time.monotonic()
        for _ in range(4):
            await limiter.acquire()
        return time.monotonic() - start

    assert asyncio.run(burst()) >= 0.45


def test_factory_builds_fake():
    llm = create_llm(Settings(llm_provider="fake"))
    assert llm.provider == "fake"
    assert llm.for_role("smart") is llm.for_role("fast")


def test_factory_requires_gemini_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_llm(Settings(llm_provider="gemini", gemini_api_key=None))


def test_factory_routes_roles_to_models():
    s = Settings(
        llm_provider="gemini",
        gemini_api_key="k",
        gemini_model_smart="big",
        gemini_model_fast="small",
        llm_cache=False,
    )
    router = create_llm(s)
    assert router.for_role("smart").model == "big" and router.for_role("fast").model == "small"
    assert router.for_role("smart").usage is router.for_role("fast").usage is router.usage


class _FakeModels:
    """Stands in for `client.aio.models` and replays a scripted sequence of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models_called = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append(config)
        self.models_called.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            text=outcome,
            candidates=[],
            usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=3),
        )


def _gemini(outcomes, **kw):
    models = _FakeModels(outcomes)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GeminiClient("m", api_key="k", client=client, rpm=100, max_retries=2, **kw), models


def test_gemini_retries_rate_limit_then_succeeds(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("automl_agent.llm.gemini.asyncio.sleep", fake_sleep)
    busy = errors.APIError(429, {"error": {"message": "quota", "details": [{"retryDelay": "3s"}]}})
    llm, models = _gemini([busy, '{"code": "x"}'])
    draft = asyncio.run(llm.complete_json([{"role": "user", "content": "hi"}], CodeDraft))
    assert draft.code == "x" and len(models.calls) == 2
    assert 3 <= sleeps[0] <= 4  # honoured retryDelay
    assert models.calls[1].response_json_schema is not None  # native structured output


def test_gemini_non_retryable_error_raises():
    llm, _ = _gemini([errors.APIError(403, {"error": {"message": "denied"}})])
    with pytest.raises(LLMError) as exc:
        asyncio.run(llm.complete([{"role": "user", "content": "hi"}]))
    assert exc.value.code == 403


def test_gemini_falls_back_when_schema_rejected():
    bad_schema = errors.APIError(400, {"error": {"message": "invalid schema"}})
    llm, models = _gemini([bad_schema, '{"code": "y"}'])
    assert asyncio.run(llm.complete_json([{"role": "user", "content": "hi"}], CodeDraft)).code == "y"
    assert models.calls[1].response_json_schema is None


def test_retry_delay_parsing():
    assert retry_delay_from_error(Exception("retryDelay: '12s'")) == 12.0
    assert retry_delay_from_error(Exception("nothing")) is None


def test_gemini_falls_back_to_next_model_when_overloaded(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("automl_agent.llm.gemini.asyncio.sleep", no_sleep)
    busy = errors.APIError(503, {"error": {"message": "high demand"}})
    llm, models = _gemini([busy, busy, busy, '{"code": "ok"}'], fallback_models=["fb-model"])
    llm.max_retries = 6
    resp = asyncio.run(llm.complete([{"role": "user", "content": "hi"}]))
    assert models.models_called == ["m", "m", "m", "fb-model"]
    assert resp.model == "fb-model" and llm.usage.by_model == {"fb-model": 1}


def test_gemini_does_not_fall_back_on_client_errors():
    llm, models = _gemini([errors.APIError(403, {"error": {"message": "denied"}})], fallback_models=["fb"])
    with pytest.raises(LLMError):
        asyncio.run(llm.complete([{"role": "user", "content": "hi"}]))
    assert models.models_called == ["m"]


def test_fallback_models_setting_parsing():
    assert Settings(gemini_fallback_models="a, b").gemini_fallback_models == ["a", "b"]
    assert Settings(gemini_fallback_models='["x"]').gemini_fallback_models == ["x"]
    assert Settings(gemini_fallback_models="").gemini_fallback_models == []


def test_search_fails_fast_and_pauses_after_quota_error():
    from automl_agent.llm.gemini import GeminiClient

    GeminiClient._search_disabled_until = 0.0
    quota = errors.APIError(429, {"error": {"message": "You exceeded your current quota"}})
    llm, models = _gemini([quota], fallback_models=["fb"])
    with pytest.raises(LLMError):
        asyncio.run(llm.search("best model?"))
    assert models.models_called == ["m"]  # one attempt, no retries, no fallback chain
    with pytest.raises(LLMError, match="paused"):
        asyncio.run(llm.search("another question"))
    assert models.models_called == ["m"]  # skipped during the cooldown
    GeminiClient._search_disabled_until = 0.0


def test_automatic_function_calling_is_disabled():
    llm, models = _gemini(['{"code": "x"}'])
    asyncio.run(llm.complete_json([{"role": "user", "content": "hi"}], CodeDraft))
    assert models.calls[0].automatic_function_calling.disable is True
