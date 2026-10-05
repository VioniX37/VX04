"""Inference endpoints for a finished run.

POST /api/runs/{id}/predict          – score JSON records
POST /api/runs/{id}/predict/batch    – score a CSV/Parquet upload; streams for large files
GET  /api/runs/{id}/artifacts/bundle – download model.joblib + predict.py + requirements.txt +
                                       schema.json + metrics.json as a zip
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response
from sqlmodel import Session

from automl_agent.config import get_settings
from automl_agent.execution import inference as inference_module
from automl_agent.execution.inference import (
    InferenceError,
    build_schema,
    load_bundle,
    predict_dataframe,
    score_file,
    validate_input,
)
from automl_agent.storage.db import RunRecord, get_session
from automl_agent.tools.ingest import IngestError, ingest_to_parquet

router = APIRouter(prefix="/runs", tags=["inference"])


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _get_run_or_404(run_id: str, session: Session) -> RunRecord:
    run = session.get(RunRecord, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


def _artifact_dir(run: RunRecord) -> Path:
    """Directory of the attempt that produced the run's selected model.

    The Manager records it in the run's metrics; it is not necessarily the latest
    attempt, because a later revision can score worse than an earlier one.
    """
    recorded = (run.metrics or {}).get("artifact_dir")
    if recorded:
        return Path(recorded)
    return get_settings().runs_dir / run.id


def _require_succeeded(run: RunRecord) -> None:
    if run.status != "succeeded":
        raise HTTPException(
            422,
            f"Run '{run.id}' has status '{run.status}'. Inference is only available for succeeded runs.",
        )


def _load_or_error(run: RunRecord) -> dict[str, Any]:
    try:
        return load_bundle(_artifact_dir(run) / "model.joblib")
    except InferenceError as exc:
        raise HTTPException(422, exc.detail()) from exc


def _ingest_upload(upload: UploadFile, workdir: Path) -> Path:
    """Stream *upload* to disk inside *workdir*, convert it to Parquet and return the Parquet path."""
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / Path(upload.filename or "upload.csv").name
    with raw_path.open("wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    parquet_path = workdir / "input.parquet"
    try:
        ingest_to_parquet(raw_path, parquet_path)
    except IngestError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        raw_path.unlink(missing_ok=True)
    return parquet_path


def _input_schema(bundle: dict[str, Any]) -> dict[str, Any]:
    """Expected input columns, their kinds and (for categoricals) the levels seen in training."""
    categories = bundle.get("categories") or {}
    kinds = build_schema(bundle)
    return {
        "task_type": bundle.get("task_type"),
        "target": bundle.get("target"),
        "features": [
            {"name": col, "kind": kinds[col], "categories": [str(v) for v in categories.get(col, [])]}
            for col in bundle["features"]
        ],
        "drop_columns": bundle.get("drop_columns", []),
    }


@router.get("/{run_id}/schema", summary="Input schema of the run's model")
def get_schema(run_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Return the columns a prediction request must contain, with their kinds and category levels.

    Raises:
        404: Run not found.
        422: Run did not succeed or no model found.
    """
    run = _get_run_or_404(run_id, session)
    _require_succeeded(run)
    return {"run_id": run_id, **_input_schema(_load_or_error(run))}


# --------------------------------------------------------------------------- #
# Predict (single / small batch via JSON)                                      #
# --------------------------------------------------------------------------- #


@router.post("/{run_id}/predict", summary="Score JSON records")
def predict_json(
    run_id: str,
    body: list[dict[str, Any]],
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Score one or more records supplied as a JSON array.

    Args:
        run_id: Id of a succeeded run.
        body: List of row dicts.  Column names must match the training schema.

    Returns:
        A JSON object with ``"predictions"`` and, for classifiers,
        ``"probabilities"`` and ``"classes"``.

    Raises:
        404: Run not found.
        422: Run did not succeed, no model found, or schema mismatch.
    """
    run = _get_run_or_404(run_id, session)
    _require_succeeded(run)
    if not body:
        raise HTTPException(422, "Request body must be a non-empty list of records.")
    bundle = _load_or_error(run)
    try:
        df = pd.DataFrame(body)
        validated = validate_input(df, bundle)
        result = predict_dataframe(validated, bundle)
    except InferenceError as exc:
        raise HTTPException(422, exc.detail()) from exc
    return {"run_id": run_id, "n": len(body), **result}


# --------------------------------------------------------------------------- #
# Batch predict (file upload)                                                  #
# --------------------------------------------------------------------------- #


@router.post("/{run_id}/predict/batch", summary="Score a CSV or Parquet file")
def predict_batch(
    run_id: str,
    file: UploadFile,
    session: Session = Depends(get_session),
) -> Response:
    """Score a CSV or Parquet upload.

    The result is returned as a Parquet file with a ``prediction`` column
    (and ``prob_<class>`` columns for classifiers).  For large files the
    server streams the response.

    Args:
        run_id: Id of a succeeded run.
        file: CSV or Parquet file to score.

    Returns:
        Scored file as ``application/octet-stream`` (Parquet).

    Raises:
        400: Unsupported file format or unreadable content.
        422: Run did not succeed, no model found, or schema mismatch.
    """
    run = _get_run_or_404(run_id, session)
    _require_succeeded(run)
    bundle = _load_or_error(run)
    workdir = get_settings().tmp_dir / "batch_uploads" / uuid.uuid4().hex
    try:
        parquet_path = _ingest_upload(file, workdir)
        scored_bytes = score_file(parquet_path, bundle, output_format="parquet")
    except InferenceError as exc:
        raise HTTPException(422, exc.detail()) from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    filename = f"{run_id}_scored.parquet"
    return Response(
        content=scored_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Bundle download                                                              #
# --------------------------------------------------------------------------- #

_PREDICT_SCRIPT = '''#!/usr/bin/env python3
"""Standalone inference script exported by AutoML-Agent.

Usage::

    python predict.py --input new_data.csv --output scored.parquet
    python predict.py --input new_data.csv --output scored.csv --format csv

Requirements::

    pip install -r requirements.txt
"""

import argparse
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from automl_inference import InferenceError, load_bundle, score_parquet_chunked  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Score new data with an exported AutoML-Agent model.")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    bundle = load_bundle(HERE / "model.joblib")
    src = Path(args.input)
    if src.suffix.lower() in (".parquet", ".pq"):
        parquet = src
    else:
        parquet = Path(args.output).with_suffix(".input.parquet")
        pl.scan_csv(src, infer_schema_length=10_000).sink_parquet(parquet)
    try:
        out = score_parquet_chunked(parquet, bundle)
    except InferenceError as exc:
        sys.exit(f"ERROR: {exc}")
    finally:
        if parquet != src:
            parquet.unlink(missing_ok=True)

    if args.format == "csv":
        out.write_csv(args.output)
    else:
        out.write_parquet(args.output)
    print(f"Scored {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
'''

_BASE_REQUIREMENTS = ("joblib", "scikit-learn", "polars", "pyarrow", "pandas", "numpy")
_FAMILY_REQUIREMENTS = {"lightgbm": "lightgbm", "xgboost": "xgboost"}


def _pinned_requirements(bundle: dict[str, Any]) -> str:
    """Build a requirements.txt pinned to the exact versions the model was trained with."""
    import importlib.metadata as im

    family = (bundle.get("config") or {}).get("model_family", "")
    packages = [*_BASE_REQUIREMENTS]
    if family in _FAMILY_REQUIREMENTS:
        packages.append(_FAMILY_REQUIREMENTS[family])
    lines = ["# Requirements for the exported AutoML-Agent model bundle (versions used in training)."]
    for pkg in packages:
        try:
            lines.append(f"{pkg}=={im.version(pkg)}")
        except im.PackageNotFoundError:
            lines.append(pkg)
    return "\n".join(lines) + "\n"


@router.get("/{run_id}/artifacts/bundle", summary="Download the deployment bundle")
def download_bundle(
    run_id: str,
    session: Session = Depends(get_session),
) -> Response:
    """Return a zip containing everything needed to run the model in a fresh environment.

    Contents:

    * ``model.joblib`` - trained model and the preprocessing state it was trained with.
    * ``automl_inference.py`` - the serving module the API itself uses.
    * ``predict.py`` - standalone CLI scoring script built on it.
    * ``requirements.txt`` - dependencies pinned to the training versions.
    * ``schema.json`` - expected input columns and their kinds.
    * ``metrics.json`` - test-split performance metrics (if available).
    * ``model_card.md`` / ``model_card.json`` - the run's model card (if generated).

    Args:
        run_id: Id of a succeeded run.

    Returns:
        A zip file as ``application/zip``.

    Raises:
        422: Run did not succeed or no model found.
    """
    run = _get_run_or_404(run_id, session)
    _require_succeeded(run)
    bundle = _load_or_error(run)
    artifact_dir = _artifact_dir(run)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact_dir / "model.joblib", "model.joblib")
        zf.writestr("automl_inference.py", Path(inference_module.__file__).read_text(encoding="utf-8"))
        zf.writestr("predict.py", _PREDICT_SCRIPT)
        zf.writestr("requirements.txt", _pinned_requirements(bundle))
        zf.writestr("schema.json", json.dumps(_input_schema(bundle), indent=2))
        for name in ("metrics.json", "model_card.md", "model_card.json"):
            path = artifact_dir / name
            if path.exists():
                zf.write(path, name)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="automl-bundle-{run_id}.zip"'},
    )
