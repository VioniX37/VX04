"""SQLite persistence for datasets and runs (artifacts live on disk under workspace/)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cache
from typing import Any

from sqlalchemy import JSON, Column
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from automl_agent.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC)


class DatasetRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    path: str
    created_at: datetime = Field(default_factory=utcnow)
    profile: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


class RunRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    prompt: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    task_spec: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    plan: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    code: str | None = None
    error: str | None = None
    llm_usage: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


@cache
def get_engine() -> Engine:
    settings = get_settings()
    settings.ensure_dirs()
    return create_engine(settings.db_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
