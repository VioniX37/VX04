"""Register datasets from uploads, server-side paths or URLs.

Every source is converted to Parquet and profiled once, at registration, so
runs never touch the original file again.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlmodel import Session

from automl_agent.config import Settings
from automl_agent.storage.db import DatasetRecord
from automl_agent.tools.dataset_profiler import profile_file
from automl_agent.tools.ingest import IngestError, ingest_to_parquet, is_supported


class DatasetError(ValueError):
    """Registration failed for a reason the user can fix (bad file, bad path, too large...)."""


def new_dataset_dir(settings: Settings) -> tuple[str, Path]:
    """Allocate an id and an empty directory for a new dataset."""
    dataset_id = uuid.uuid4().hex[:12]
    path = settings.datasets_dir / dataset_id
    path.mkdir(parents=True, exist_ok=True)
    return dataset_id, path


def register_file(
    session: Session,
    settings: Settings,
    *,
    dataset_id: str,
    dataset_dir: Path,
    src: Path,
    filename: str,
    source: str,
    delete_source: bool,
) -> DatasetRecord:
    """Ingest `src` into `dataset_dir/data.parquet`, profile it and store the record."""
    size = src.stat().st_size
    try:
        parquet = ingest_to_parquet(src, dataset_dir / "data.parquet")
        profile = profile_file(parquet, sample_rows=settings.profile_sample_rows)
    except IngestError as e:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise DatasetError(str(e)) from e
    finally:
        if delete_source:
            src.unlink(missing_ok=True)
    record = DatasetRecord(
        id=dataset_id,
        filename=filename,
        path=str(parquet),
        source=source,
        size_bytes=size,
        profile=profile.model_dump(mode="json"),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def register_path(session: Session, settings: Settings, path: str, name: str | None = None) -> DatasetRecord:
    """Register a file that already exists on the server's disk (e.g. a Kaggle/Colab input)."""
    if not settings.allow_path_registration:
        raise DatasetError("Registering server-side paths is disabled (ALLOW_PATH_REGISTRATION=false).")
    src = Path(path).expanduser()
    if not src.is_file():
        raise DatasetError(f"File not found: {src}")
    if not is_supported(src):
        raise DatasetError(f"Unsupported file type: {src.name}")
    dataset_id, dataset_dir = new_dataset_dir(settings)
    return register_file(
        session, settings, dataset_id=dataset_id, dataset_dir=dataset_dir, src=src,
        filename=name or src.name, source="path", delete_source=False,
    )  # fmt: skip


def register_url(session: Session, settings: Settings, url: str, name: str | None = None) -> DatasetRecord:
    """Download a file over http(s) (streamed to disk) and register it."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise DatasetError("Only http(s) URLs are supported.")
    filename = name or Path(parsed.path).name or "download.csv"
    if not is_supported(Path(filename)):
        raise DatasetError(f"Cannot tell the file type from '{filename}'. Pass a name with an extension.")
    dataset_id, dataset_dir = new_dataset_dir(settings)
    raw = dataset_dir / f"raw-{filename}"
    limit = settings.max_upload_mb * 1024 * 1024 * 4  # URLs may be larger than browser uploads
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            written = 0
            with raw.open("wb") as out:
                for chunk in resp.iter_bytes(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise DatasetError("Download exceeds the size limit.")
                    out.write(chunk)
    except (httpx.HTTPError, DatasetError) as e:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise DatasetError(f"Download failed: {e}") from e
    return register_file(
        session, settings, dataset_id=dataset_id, dataset_dir=dataset_dir, src=raw,
        filename=filename, source="url", delete_source=True,
    )  # fmt: skip
