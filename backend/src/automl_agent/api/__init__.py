from fastapi import APIRouter

from . import datasets, health, runs

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(runs.router)

__all__ = ["api_router"]
