"""Read and write experience records."""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from automl_agent.config import Settings
from automl_agent.storage.db import ExperienceRecord, get_engine, init_db


class ExperienceStore:
    """Experience records in the workspace database of `settings`."""

    def __init__(self, settings: Settings) -> None:
        self.url = settings.db_url
        init_db(self.url)

    def add(self, record: ExperienceRecord) -> None:
        """Insert or replace a record."""
        with Session(get_engine(self.url)) as session:
            session.merge(record)
            session.commit()

    def for_task(self, task_type: str, *, exclude_fingerprint: str | None = None) -> list[ExperienceRecord]:
        """All records of a task type, optionally excluding one dataset."""
        with Session(get_engine(self.url)) as session:
            query = select(ExperienceRecord).where(ExperienceRecord.task_type == task_type)
            if exclude_fingerprint:
                query = query.where(ExperienceRecord.dataset_fingerprint != exclude_fingerprint)
            return list(session.exec(query).all())

    def fixes(self, *, exclude_fingerprint: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent error->fix pairs across all tasks (errors are rarely task-specific)."""
        with Session(get_engine(self.url)) as session:
            query = select(ExperienceRecord).order_by(ExperienceRecord.created_at.desc()).limit(limit)
            rows = session.exec(query).all()
        out: list[dict[str, Any]] = []
        for r in rows:
            if exclude_fingerprint and r.dataset_fingerprint == exclude_fingerprint:
                continue
            out.extend(r.fixes or [])
        return out
