"""Dataset endpoints: upload, register by path/URL, list and inspect.

`POST /datasets` and `POST /datasets/register` block until the dataset is ready.
The `/datasets/jobs` variants return at once with a job to poll for progress,
which is what the UI uses for multi-gigabyte files.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session, select

from automl_agent.config import get_settings
from automl_agent.schemas.dataset import DatasetOut, DatasetProfile, DatasetRegister, IngestJobOut
from automl_agent.services.dataset_service import (
    DatasetError,
    check_path,
    check_url,
    new_dataset_dir,
    register_file,
    register_path,
    register_url,
)
from automl_agent.services.ingest_jobs import IngestJob, get_job, start_job, steps_for
from automl_agent.storage.db import DatasetRecord, get_session
from automl_agent.tools.ingest import is_supported, source_suffix

router = APIRouter(prefix="/datasets", tags=["datasets"])

COPY_CHUNK = 16 * 1024 * 1024


def to_out(rec: DatasetRecord) -> DatasetOut:
    """Convert a database record to the API schema."""
    return DatasetOut(
        id=rec.id,
        filename=rec.filename,
        source=rec.source or "upload",
        created_at=rec.created_at,
        profile=DatasetProfile.model_validate(rec.profile),
    )


def job_out(job: IngestJob, session: Session) -> IngestJobOut:
    """Snapshot a job for the API, attaching the dataset once it is registered."""
    now = job.finished_at or time.time()
    rec = session.get(DatasetRecord, job.dataset_id) if job.dataset_id else None
    return IngestJobOut(
        id=job.id,
        filename=job.filename,
        status=job.status,
        steps=job.steps,
        phase=job.phase,
        fraction=job.fraction,
        message=job.message,
        error=job.error,
        elapsed_s=round(now - job.started_at, 2),
        phase_elapsed_s=round(now - job.phase_started_at, 2),
        dataset=to_out(rec) if rec else None,
    )


def _receive_upload(file: UploadFile) -> tuple[str, str, Path, Path]:
    """Stream an upload to a new dataset directory; returns (id, name, dir, raw file)."""
    settings = get_settings()
    name = Path(file.filename or "data.csv").name
    if not is_supported(Path(name)):
        raise HTTPException(
            400, f"Unsupported file type '{source_suffix(Path(name))}'. Upload CSV, TSV, Parquet or JSONL."
        )
    dataset_id, dataset_dir = new_dataset_dir(settings)
    raw = dataset_dir / f"raw-{name}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with raw.open("wb") as out:
        while chunk := file.file.read(COPY_CHUNK):
            written += len(chunk)
            if written > limit:
                out.close()
                shutil.rmtree(dataset_dir, ignore_errors=True)
                raise HTTPException(
                    413,
                    f"File too large (max {settings.max_upload_mb} MB). "
                    "Register large files by path instead.",
                )
            out.write(chunk)
    return dataset_id, name, dataset_dir, raw


@router.post("", response_model=DatasetOut, status_code=201)
def upload_dataset(file: UploadFile, session: Session = Depends(get_session)) -> DatasetOut:
    """Upload a CSV/TSV/Parquet/JSONL file (streamed to disk, then converted to Parquet)."""
    dataset_id, name, dataset_dir, raw = _receive_upload(file)
    try:
        rec = register_file(
            session,
            get_settings(),
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            src=raw,
            filename=name,
            source="upload",
            delete_source=True,
        )
    except DatasetError as e:
        raise HTTPException(400, str(e)) from e
    return to_out(rec)


@router.post("/register", response_model=DatasetOut, status_code=201)
def register_dataset(body: DatasetRegister, session: Session = Depends(get_session)) -> DatasetOut:
    """Register a dataset by server-side path or http(s) URL (preferred for multi-GB files)."""
    if bool(body.path) == bool(body.url):
        raise HTTPException(422, "Provide exactly one of 'path' or 'url'.")
    try:
        if body.path:
            rec = register_path(session, get_settings(), body.path, body.name)
        else:
            rec = register_url(session, get_settings(), body.url or "", body.name)
    except DatasetError as e:
        raise HTTPException(400, str(e)) from e
    return to_out(rec)


@router.post("/jobs", response_model=IngestJobOut, status_code=202)
def upload_dataset_job(file: UploadFile, session: Session = Depends(get_session)) -> IngestJobOut:
    """Upload a file, then convert and profile it in the background; poll the returned job."""
    dataset_id, name, dataset_dir, raw = _receive_upload(file)
    settings = get_settings()
    job = start_job(
        name,
        steps_for("upload", name),
        lambda s, progress: register_file(
            s,
            settings,
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            src=raw,
            filename=name,
            source="upload",
            delete_source=True,
            progress=progress,
        ),
    )
    return job_out(job, session)


@router.post("/jobs/register", response_model=IngestJobOut, status_code=202)
def register_dataset_job(body: DatasetRegister, session: Session = Depends(get_session)) -> IngestJobOut:
    """Register by path or URL in the background; bad input is rejected at once, the rest is polled."""
    if bool(body.path) == bool(body.url):
        raise HTTPException(422, "Provide exactly one of 'path' or 'url'.")
    settings = get_settings()
    try:
        if body.path:
            src = check_path(settings, body.path)
            name = body.name or src.name
            job = start_job(
                name,
                steps_for("path", src.name),
                lambda s, progress: register_path(s, settings, str(src), body.name, progress),
            )
        else:
            url = body.url or ""
            name = check_url(url, body.name)
            job = start_job(
                name,
                steps_for("url", name),
                lambda s, progress: register_url(s, settings, url, body.name, progress),
            )
    except DatasetError as e:
        raise HTTPException(400, str(e)) from e
    return job_out(job, session)


@router.get("/jobs/{job_id}", response_model=IngestJobOut)
def get_dataset_job(job_id: str, session: Session = Depends(get_session)) -> IngestJobOut:
    """Current phase, progress and any error of a background registration."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Registration job not found (the server may have restarted).")
    return job_out(job, session)


@router.get("", response_model=list[DatasetOut])
def list_datasets(session: Session = Depends(get_session)) -> list[DatasetOut]:
    """List registered datasets, newest first."""
    rows = session.exec(select(DatasetRecord).order_by(DatasetRecord.created_at.desc())).all()
    return [to_out(r) for r in rows]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, session: Session = Depends(get_session)) -> DatasetOut:
    """Return one dataset and its profile."""
    rec = session.get(DatasetRecord, dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")
    return to_out(rec)
