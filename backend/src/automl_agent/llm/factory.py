"""Builds the per-run :class:`~automl_agent.llm.router.LLMRouter` from settings."""

from __future__ import annotations

from automl_agent.config import Settings

from .cache import ResponseCache
from .router import LLMRouter


def create_llm(settings: Settings) -> LLMRouter:
    """Create a fresh router (one per run, so usage is tracked per run).

    Raises:
        ValueError: If Gemini is selected but no API key / Vertex project is configured.
    """
    if settings.llm_provider == "fake":
        from .fake import FakeLLM

        return LLMRouter(FakeLLM())

    if not settings.gemini_use_vertexai and not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to backend/.env or set LLM_PROVIDER=fake.")
    if settings.gemini_use_vertexai and not settings.google_cloud_project:
        raise ValueError("Vertex AI mode requires GOOGLE_CLOUD_PROJECT.")

    from .gemini import GeminiClient

    cache = ResponseCache(settings.llm_cache_dir) if settings.llm_cache else None

    def build(model: str, rpm: int) -> GeminiClient:
        return GeminiClient(
            model,
            api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            rpm=rpm,
            max_concurrency=settings.gemini_max_concurrency,
            max_retries=settings.gemini_max_retries,
            fallback_models=settings.gemini_fallback_models,
            vertexai=settings.gemini_use_vertexai,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            cache=cache,
        )

    smart = build(settings.gemini_model_smart, settings.gemini_rpm_smart)
    if settings.gemini_model_fast == settings.gemini_model_smart:
        return LLMRouter(smart)
    return LLMRouter(smart, build(settings.gemini_model_fast, settings.gemini_rpm_fast))


async def validate_models(settings: Settings) -> dict[str, bool]:
    """Check that the configured Gemini model ids exist for this key.

    Returns:
        Mapping of model id -> available. Empty when validation is not applicable.
    """
    if settings.llm_provider != "gemini":
        return {}
    router = create_llm(settings)
    available = set(await router.for_role("smart").list_models())  # type: ignore[attr-defined]
    configured = {settings.gemini_model_smart, settings.gemini_model_fast, *settings.gemini_fallback_models}
    return {m: m in available for m in configured}
