"""Liveness endpoint that also reports the active LLM configuration."""

from fastapi import APIRouter, Request

from automl_agent import __version__
from automl_agent.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    """Return service status, version and the configured Gemini models per role."""
    s = get_settings()
    fake = s.llm_provider == "fake"
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": s.llm_provider,
        "llm_model": "fake-heuristic" if fake else s.gemini_model_smart,
        "models": {} if fake else {"smart": s.gemini_model_smart, "fast": s.gemini_model_fast},
        "models_available": getattr(request.app.state, "models_available", {}),
        "codegen_mode": s.codegen_mode,
    }
