from fastapi import APIRouter

from automl_agent import __version__
from automl_agent.config import get_settings
from automl_agent.llm import DEFAULT_MODELS

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model or DEFAULT_MODELS[s.llm_provider],
        "codegen_mode": s.codegen_mode,
    }
