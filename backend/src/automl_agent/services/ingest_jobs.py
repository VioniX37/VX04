"""Background dataset registration with live progress.

Converting and profiling a multi-gigabyte file takes minutes, far longer than an
HTTP request should block. A job runs the registration in a worker thread and
records which phase it is in and how far along, so the UI can poll it.
Jobs live in memory: they only matter while someone is watching them.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlmodel import Session

from automl_agent.storage.db import DatasetRecord, get_engine

from .dataset_service import DatasetError, Progress

log = logging.getLogger(__name__)

KEEP_FINISHED_S = 3600


@dataclass
class IngestJob:
    """State of one registration, updated by the worker thread."""

    id: str
    filename: str
    steps: list[str]
    status: str = "running"  # running | succeeded | failed
    phase: str = ""
    fraction: float | None = None
    message: str = "Starting"
    error: str | None = None
    dataset_id: str | None = None
    started_at: float = field(default_factory=time.time)
    phase_started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def report(self, phase: str, fraction: float | None, message: str) -> None:
        """Record progress (the `Progress` callback handed to the registration)."""
        if phase != self.phase:
            self.phase, self.phase_started_at = phase, time.time()
        self.fraction, self.message = fraction, message


_jobs: dict[str, IngestJob] = {}
_lock = threading.Lock()


def steps_for(source: str, filename: str) -> list[str]:
    """The phases a registration will go through, in order, for the progress display."""
    steps = {"upload": ["upload"], "url": ["download"]}.get(source, [])
    if filename.lower().endswith(".gz"):
        steps.append("decompress")
    return [*steps, "convert", "profile"]


def start_job(
    filename: str, steps: list[str], work: Callable[[Session, Progress], DatasetRecord]
) -> IngestJob:
    """Run `work(session, progress)` in a worker thread and return its job."""
    job = IngestJob(id=uuid.uuid4().hex[:12], filename=filename, steps=steps)
    job.report(next(s for s in steps if s != "upload"), None, "Starting")  # the browser did the upload
    with _lock:
        cutoff = time.time() - KEEP_FINISHED_S
        for old in [j for j in _jobs.values() if j.finished_at and j.finished_at < cutoff]:
            del _jobs[old.id]
        _jobs[job.id] = job

    def run() -> None:
        try:
            with Session(get_engine()) as session:
                record = work(session, job.report)
            job.dataset_id = record.id
            job.report("ready", 1.0, f"Registered {record.filename}")
            job.status = "succeeded"
        except DatasetError as e:
            job.error, job.status = str(e), "failed"
        except MemoryError:
            job.error = "Ran out of memory while processing the file."
            job.status = "failed"
        except Exception as e:
            log.exception("Dataset registration %s failed", job.id)
            job.error, job.status = f"Unexpected error during {job.phase}: {e}", "failed"
        finally:
            job.finished_at = time.time()

    threading.Thread(target=run, name=f"ingest-{job.id}", daemon=True).start()
    return job


def get_job(job_id: str) -> IngestJob | None:
    """Look up a job by id."""
    with _lock:
        return _jobs.get(job_id)
