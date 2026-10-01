"""Dataset endpoints: upload, register by path/URL, list and inspect."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session, select

from automl_agent.config import get_settings
from automl_agent.schemas.dataset import DatasetOut, DatasetProfile, DatasetRegister
from automl_agent.services.dataset_service import (
    DatasetError,
    new_dataset_dir,
    register_file,
    register_path,
    register_url,
)
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


@router.post("", response_model=DatasetOut, status_code=201)
def upload_dataset(file: UploadFile, session: Session = Depends(get_session)) -> DatasetOut:
    """Upload a CSV/TSV/Parquet/JSONL file (streamed to disk, then converted to Parquet)."""
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
    try:
        rec = register_file(
            session,
            settings,
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
