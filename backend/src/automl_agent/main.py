"""FastAPI entry point: ``python -m uvicorn automl_agent.main:app --reload --app-dir src``."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from automl_agent import __version__
from automl_agent.api import api_router
from automl_agent.config import get_settings
from automl_agent.llm import validate_models
from automl_agent.storage.db import init_db

log = logging.getLogger("automl_agent")


async def _check_models(app: FastAPI) -> None:
    """Warn early (without blocking startup) if a configured Gemini model id is unavailable."""
    try:
        result = await asyncio.wait_for(validate_models(get_settings()), timeout=15)
    except Exception as e:  # network errors, missing key, ...
        log.warning("Could not validate Gemini models: %s", e)
        return
    app.state.models_available = result
    for model, ok in result.items():
        if not ok:
            log.warning("Configured Gemini model '%s' is not available for this API key.", model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the workspace and database, then validate the model configuration."""
    get_settings().ensure_dirs()
    init_db()
    app.state.models_available = {}
    task = asyncio.create_task(_check_models(app))
    yield
    task.cancel()


def create_app() -> FastAPI:
    """Build the FastAPI application with CORS and all API routes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = FastAPI(title="AutoML-Agent", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()
