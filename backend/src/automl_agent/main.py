"""FastAPI entry point: `uvicorn automl_agent.main:app --reload --app-dir src`."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from automl_agent import __version__
from automl_agent.api import api_router
from automl_agent.config import get_settings
from automl_agent.storage.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings().ensure_dirs()
    init_db()
    yield


def create_app() -> FastAPI:
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
