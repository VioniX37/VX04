"""SQLite persistence for datasets, runs and experiment observations.

Large artifacts (data, scripts, models, event logs) live on disk under the
workspace; the database stores metadata and small JSON documents.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from automl_agent.config import get_settings


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


class DatasetRecord(SQLModel, table=True):
    """A registered dataset (always stored as Parquet)."""

    id: str = Field(primary_key=True)
    filename: str
    path: str
    source: str = "upload"
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    profile: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


class RunRecord(SQLModel, table=True):
    """One pipeline run and its final outcome."""

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
    config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


@cache
def get_engine(url: str | None = None) -> Engine:
    """Return a cached engine for `url` (defaults to the configured database)."""
    if url is None:
        settings = get_settings()
        settings.ensure_dirs()
        url = settings.db_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args={"check_same_thread": False})


def _add_missing_columns(engine: Engine) -> None:
    """Minimal forward migration: add columns that exist in the models but not yet in SQLite."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                default = (
                    column.default.arg if column.default is not None and column.default.is_scalar else None
                )
                clause = f" DEFAULT {default!r}" if isinstance(default, int | float | str) else ""
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}{clause}')
                )


def init_db(url: str | None = None) -> None:
    """Create tables and apply additive column migrations."""
    engine = get_engine(url)
    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session on the configured database."""
    with Session(get_engine()) as session:
        yield session
