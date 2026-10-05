"""REST API routers mounted under /api."""

from fastapi import APIRouter

from . import datasets, health, inference, runs

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(runs.router)
api_router.include_router(inference.router)

__all__ = ["api_router"]
