"""Inference endpoints for a finished run.

POST /api/runs/{id}/predict          – score JSON records
POST /api/runs/{id}/predict/batch    – score a CSV/Parquet upload; streams for large files
GET  /api/runs/{id}/artifacts/bundle – download model.joblib + predict.py + requirements.txt +
                                       schema.json + metrics.json as a zip
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlmodel import Session

from automl_agent.config import get_settings
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

import pandas as pd

router = APIRouter(prefix="/runs", tags=["inference"])


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _get_run_or_404(run_id: str, session: Session) -> RunRecord:
    run = session.get(RunRecord, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


def _model_path(run_id: str) -> Path:
    """Return the path to model.joblib for a run's most recent attempt."""
    settings = get_settings()
    run_dir = settings.runs_dir / run_id
    # Prefer the latest attempt directory, fall back to the run root.
    attempt_dirs = sorted(run_dir.glob("attempt_*"), reverse=True)
    if attempt_dirs:
        candidate = attempt_dirs[0] / "model.joblib"
        if candidate.exists():
            return candidate
    return run_dir / "model.joblib"


def _metrics_path(run_id: str) -> Path:
    settings = get_settings()
    run_dir = settings.runs_dir / run_id
    attempt_dirs = sorted(run_dir.glob("attempt_*"), reverse=True)
    if attempt_dirs:
        candidate = attempt_dirs[0] / "metrics.json"
        if candidate.exists():
            return candidate
    return run_dir / "metrics.json"


def _require_succeeded(run: RunRecord) -> None:
    if run.status != "succeeded":
        raise HTTPException(
            422,
            f"Run '{run.id}' has status '{run.status}'. Inference is only available for succeeded runs.",
        )


def _load_or_error(run_id: str) -> dict[str, Any]:
    try:
        return load_bundle(_model_path(run_id))
    except InferenceError as exc:
        raise HTTPException(422, exc.detail()) from exc


def _ingest_upload(upload: UploadFile, run_id: str) -> Path:
    """Save *upload* to the tmp dir and convert it to Parquet, then return the Parquet path."""
    settings = get_settings()
    tmp = settings.tmp_dir / "batch_uploads" / run_id
    tmp.mkdir(parents=True, exist_ok=True)
    raw_path = tmp / (upload.filename or "upload")
    raw_path.write_bytes(upload.file.read())
    parquet_path = tmp / "input.parquet"
    try:
        ingest_to_parquet(raw_path, parquet_path)
    except IngestError as exc:
        raise HTTPException(400, str(exc)) from exc
    return parquet_path


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
    bundle = _load_or_error(run_id)
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
async def predict_batch(
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
    bundle = _load_or_error(run_id)
    parquet_path = _ingest_upload(file, run_id)
    try:
        scored_bytes = score_file(parquet_path, bundle, output_format="parquet")
    except InferenceError as exc:
        raise HTTPException(422, exc.detail()) from exc
    finally:
        parquet_path.unlink(missing_ok=True)

    filename = f"{run_id}_scored.parquet"
    return Response(
        content=scored_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Bundle download                                                              #
# --------------------------------------------------------------------------- #

_PREDICT_SCRIPT = '''\
#!/usr/bin/env python3
"""Standalone inference script exported by AutoML-Agent.

Usage::

    python predict.py --input new_data.csv --output scored.parquet
    python predict.py --input new_data.csv --output scored.csv --format csv

Requirements::

    pip install -r requirements.txt
"""

import argparse
import io
import sys

import joblib
import pandas as pd
import polars as pl
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Score new data with an exported AutoML-Agent model.")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    bundle = joblib.load("model.joblib")
    features = bundle["features"]
    classes = bundle.get("classes")
    task_type = bundle.get("task_type", "tabular_classification")
    is_clf = task_type != "tabular_regression"

    # Load input
    path = args.input
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Drop target / drop columns silently
    target = bundle.get("target")
    if target and target in df.columns:
        df = df.drop(columns=[target])
    for col in bundle.get("drop_columns", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Validate
    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns: {missing}")
    df = df[features].copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    # Predict
    model = bundle["model"]
    preds_raw = model.predict(df)
    if is_clf and classes is not None:
        preds = [str(classes[int(p)]) for p in preds_raw]
    elif is_clf:
        preds = [str(p) for p in preds_raw]
    else:
        preds = [float(p) for p in preds_raw]

    out = pl.DataFrame({"prediction": preds})
    if is_clf and hasattr(model, "predict_proba") and classes is not None:
        try:
            proba = model.predict_proba(df)
            for i, cls in enumerate(classes):
                out = out.with_columns(pl.Series(f"prob_{cls}", proba[:, i].tolist()))
        except Exception:
            pass

    if args.format == "csv":
        out.write_csv(args.output)
    else:
        out.write_parquet(args.output)
    print(f"Scored {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
'''

_REQUIREMENTS_TEMPLATE = """\
# Requirements for the exported AutoML-Agent model bundle.
# Install with: pip install -r requirements.txt
# Pinned versions match those used during training.
joblib>={joblib_version}
scikit-learn>={sklearn_version}
polars>={polars_version}
pyarrow>={pyarrow_version}
pandas>={pandas_version}
numpy>={numpy_version}
{extra_deps}
"""


def _pinned_requirements(bundle: dict[str, Any]) -> str:
    """Build a requirements.txt string with versions pinned to what is installed."""
    import importlib.metadata as im

    def ver(pkg: str) -> str:
        try:
            return im.version(pkg)
        except im.PackageNotFoundError:
            return "0"

    family = (bundle.get("config") or {}).get("model_family", "")
    extra_lines = []
    if family == "lightgbm":
        extra_lines.append(f"lightgbm>={ver('lightgbm')}")
    elif family == "xgboost":
        extra_lines.append(f"xgboost>={ver('xgboost')}")

    return _REQUIREMENTS_TEMPLATE.format(
        joblib_version=ver("joblib"),
        sklearn_version=ver("scikit-learn"),
        polars_version=ver("polars"),
        pyarrow_version=ver("pyarrow"),
        pandas_version=ver("pandas"),
        numpy_version=ver("numpy"),
        extra_deps="\n".join(extra_lines),
    ).rstrip() + "\n"


@router.get("/{run_id}/artifacts/bundle", summary="Download the deployment bundle")
def download_bundle(
    run_id: str,
    session: Session = Depends(get_session),
) -> Response:
    """Return a zip containing everything needed to run the model in a fresh environment.

    Contents:

    * ``model.joblib`` – trained model and preprocessing state.
    * ``predict.py`` – standalone CLI scoring script.
    * ``requirements.txt`` – pinned dependencies.
    * ``schema.json`` – expected input columns and dtypes.
    * ``metrics.json`` – test-split performance metrics (if available).

    Args:
        run_id: Id of a succeeded run.

    Returns:
        A zip file as ``application/zip``.

    Raises:
        422: Run did not succeed or no model found.
    """
    run = _get_run_or_404(run_id, session)
    _require_succeeded(run)
    bundle = _load_or_error(run_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # model.joblib (re-serialise to bytes so we don't have a dangling file handle)
        model_bytes = io.BytesIO()
        import joblib as _joblib
        _joblib.dump(bundle, model_bytes)
        zf.writestr("model.joblib", model_bytes.getvalue())

        # predict.py
        zf.writestr("predict.py", _PREDICT_SCRIPT)

        # requirements.txt
        zf.writestr("requirements.txt", _pinned_requirements(bundle))

        # schema.json
        schema = {
            "features": bundle["features"],
            "task_type": bundle.get("task_type"),
            "target": bundle.get("target"),
            "drop_columns": bundle.get("drop_columns", []),
            "dtypes": build_schema(bundle),
        }
        zf.writestr("schema.json", json.dumps(schema, indent=2))

        # metrics.json (best effort)
        metrics_file = _metrics_path(run_id)
        if metrics_file.exists():
            zf.writestr("metrics.json", metrics_file.read_text(encoding="utf-8"))

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="automl-bundle-{run_id}.zip"'},
    )
