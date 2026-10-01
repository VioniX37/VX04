"""FastAPI entry point: ``python -m uvicorn automl_agent.main:app --reload --app-dir src``."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from automl_agent import __version__
from automl_agent.api import api_router
from automl_agent.config import Settings, get_settings
from automl_agent.llm import validate_models
from automl_agent.storage.db import init_db

log = logging.getLogger("automl_agent")


def use_workspace_tempdir(settings: Settings) -> None:
    """Buffer request bodies (multi-GB uploads) on the workspace drive, not the system temp folder.

    Starlette spools uploaded files to Python's temp directory, which on Windows is
    usually on C:. Large uploads then fail when C: is nearly full, even though the
    workspace drive has plenty of space.
    """
    tmp = settings.tmp_dir
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    free_gb = shutil.disk_usage(tmp).free / 1024**3
    log.info("Upload buffer: %s (%.1f GB free)", tmp, free_gb)
    if free_gb < 2 * settings.max_upload_mb / 1024:
        log.warning(
            "Only %.1f GB free for uploads; the largest upload needs about twice its size. "
            "Register large files by path instead of uploading them.",
            free_gb,
        )


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


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Turn FastAPI's generic body-parsing error into an actionable message when the disk is full."""
    cause = exc.__cause__
    if exc.status_code == 400 and isinstance(cause, OSError):
        return JSONResponse(
            status_code=507,
            content={
                "detail": f"Could not store the upload ({cause.strerror or cause}). The disk holding the "
                "upload buffer is probably full. Free some space, or register the file by path instead."
            },
        )
    return await http_exception_handler(request, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the workspace, upload buffer and database, then validate the model configuration."""
    settings = get_settings()
    settings.ensure_dirs()
    use_workspace_tempdir(settings)
    init_db()
    app.state.models_available = {}
    task = asyncio.create_task(_check_models(app))
    yield
    task.cancel()


def create_app() -> FastAPI:
    """Build the FastAPI application with CORS, error handling and all API routes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = FastAPI(title="AutoML-Agent", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.include_router(api_router)
    return app


app = create_app()
