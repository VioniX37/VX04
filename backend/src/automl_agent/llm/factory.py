from __future__ import annotations

from automl_agent.config import Settings

from .base import LLMClient

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5-coder:7b",
    "openai_compatible": "",
    "fake": "fake-heuristic",
}

_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}


def create_llm(settings: Settings) -> LLMClient:
    """Build a fresh client for the configured provider (one per run, so usage is tracked per run)."""
    provider = settings.llm_provider
    model = settings.llm_model or DEFAULT_MODELS[provider]
    temp = settings.llm_temperature

    if provider == "fake":
        from .fake import FakeLLM

        return FakeLLM(model)
    if provider == "anthropic":
        from .anthropic import AnthropicClient

        return AnthropicClient(model, api_key=settings.anthropic_api_key, temperature=temp)
    if provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(model, api_key=settings.gemini_api_key, temperature=temp)

    from .openai_compat import OpenAICompatClient

    if provider == "openai_compatible" and not (settings.llm_base_url and model):
        raise ValueError("openai_compatible requires LLM_BASE_URL and LLM_MODEL")
    api_key = {"openai": settings.openai_api_key, "groq": settings.groq_api_key}.get(provider)
    return OpenAICompatClient(
        model,
        api_key=api_key,
        base_url=settings.llm_base_url or _BASE_URLS.get(provider),
        temperature=temp,
        provider=provider,
    )
