from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class RunCreate(BaseModel):
    dataset_id: str
    prompt: str = Field(min_length=3)


class RunOut(BaseModel):
    id: str
    dataset_id: str
    prompt: str
    status: RunStatus
    created_at: datetime
    finished_at: datetime | None = None
    task_spec: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    code: str | None = None
    error: str | None = None
    llm_usage: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
