from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session, select

from automl_agent.config import get_settings
from automl_agent.schemas.dataset import DatasetOut, DatasetProfile
from automl_agent.storage.db import DatasetRecord, get_session
from automl_agent.tools.dataset_profiler import profile_file

router = APIRouter(prefix="/datasets", tags=["datasets"])

ALLOWED_SUFFIXES = {".csv", ".tsv"}
MAX_BYTES = 200 * 1024 * 1024


def to_out(rec: DatasetRecord) -> DatasetOut:
    return DatasetOut(
        id=rec.id,
        filename=rec.filename,
        created_at=rec.created_at,
        profile=DatasetProfile.model_validate(rec.profile),
    )


@router.post("", response_model=DatasetOut, status_code=201)
def upload_dataset(file: UploadFile, session: Session = Depends(get_session)) -> DatasetOut:
    name = Path(file.filename or "data.csv").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Upload a CSV or TSV file.")

    dataset_id = uuid.uuid4().hex[:12]
    dest_dir = get_settings().datasets_dir / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"data{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    if dest.stat().st_size > MAX_BYTES:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(413, "File too large (max 200 MB)")

    try:
        profile = profile_file(dest)
    except Exception as e:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not parse file: {e}") from e

    rec = DatasetRecord(id=dataset_id, filename=name, path=str(dest), profile=profile.model_dump(mode="json"))
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return to_out(rec)


@router.get("", response_model=list[DatasetOut])
def list_datasets(session: Session = Depends(get_session)) -> list[DatasetOut]:
    rows = session.exec(select(DatasetRecord).order_by(DatasetRecord.created_at.desc())).all()
    return [to_out(r) for r in rows]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, session: Session = Depends(get_session)) -> DatasetOut:
    rec = session.get(DatasetRecord, dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")
    return to_out(rec)
