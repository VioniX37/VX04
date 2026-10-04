"""Model-serving counterpart to the training templates.

Loads the bundle written by a template (``{"model", "classes", "features",
"task_type", "target", "drop_columns", "text_column", "config"}``),
applies the *same* preprocessing that the training script used, and returns
predictions (and probabilities for classifiers).

Design goals
------------
- Training and serving share one code path: the template writes a superset
  of what this module needs; no duplicate preprocessing logic.
- Validate inputs early and return structured errors – never 500.
- Support batch scoring of large files via Polars streaming so memory stays
  bounded by EXEC_MAX_MEM_MB.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

# --------------------------------------------------------------------------- #
# Bundle I/O                                                                   #
# --------------------------------------------------------------------------- #

BUNDLE_KEYS_REQUIRED = {"model", "features", "task_type"}
CHUNK_ROWS = 100_000


class InferenceError(ValueError):
    """A structured error that the API can forward directly (never a 500)."""

    def __init__(self, message: str, column: str | None = None) -> None:
        super().__init__(message)
        self.column = column

    def detail(self) -> dict[str, Any]:
        """Serialisable form for a FastAPI response body."""
        d: dict[str, Any] = {"error": str(self)}
        if self.column:
            d["column"] = self.column
        return d


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    """Load and minimally validate a ``model.joblib`` bundle.

    Args:
        bundle_path: Absolute path to the joblib file.

    Returns:
        The dict stored in the bundle.

    Raises:
        InferenceError: If the file is missing or the format is wrong.
    """
    if not bundle_path.exists():
        raise InferenceError("No trained model found for this run. The run may have failed or not yet saved a model.")
    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:
        raise InferenceError(f"Could not load model bundle: {exc}") from exc
    if not isinstance(bundle, dict):
        raise InferenceError("Model bundle has unexpected format (not a dict).")
    missing = BUNDLE_KEYS_REQUIRED - bundle.keys()
    if missing:
        raise InferenceError(f"Model bundle is missing keys: {', '.join(sorted(missing))}")
    return bundle


# --------------------------------------------------------------------------- #
# Schema / validation                                                          #
# --------------------------------------------------------------------------- #

def build_schema(bundle: dict[str, Any]) -> dict[str, str]:
    """Return a ``{column_name: dtype_str}`` schema from the bundle.

    The training templates don't persist column dtypes explicitly, so we
    reconstruct them from the fitted sklearn ColumnTransformer's named steps
    when available, and fall back to ``"unknown"`` otherwise.  The schema is
    used to produce ``schema.json`` and to validate incoming requests.
    """
    features: list[str] = bundle["features"]
    dtypes: dict[str, str] = {}
    model = bundle["model"]

    # Try to extract dtype knowledge from a sklearn Pipeline's preprocessor.
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer

        prep = model.named_steps.get("prep") if isinstance(model, Pipeline) else None
        if isinstance(prep, ColumnTransformer):
            for name, transformer, cols in prep.transformers_:
                if name == "num":
                    for c in cols:
                        dtypes[c] = "float"
                elif name == "cat":
                    for c in cols:
                        dtypes[c] = "category"
    except Exception:
        pass

    for col in features:
        dtypes.setdefault(col, "unknown")
    return dtypes


def validate_input(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Check that *df* has the required columns and return a frame ready for prediction.

    Applies the same column selection and categorical dtypes that the training
    script used.

    Args:
        df: Raw input dataframe.
        bundle: Loaded model bundle.

    Returns:
        A new dataframe containing only the feature columns in training order,
        with categorical dtypes restored.

    Raises:
        InferenceError: If a required column is missing or an extra column
            causes ambiguity (rejected to prevent silent misalignment).
    """
    features: list[str] = bundle["features"]
    drop_cols: list[str] = bundle.get("drop_columns", [])
    target: str | None = bundle.get("target")

    # Silently drop the target column if present (scoring convenience).
    if target and target in df.columns:
        df = df.drop(columns=[target])

    # Silently drop explicitly excluded columns.
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Check for missing required columns.
    for col in features:
        if col not in df.columns:
            raise InferenceError(
                f"Required column '{col}' is missing from the input. "
                f"Expected columns: {features}",
                column=col,
            )

    # Check for unexpected extra columns.
    extra = sorted(set(df.columns) - set(features))
    if extra:
        raise InferenceError(
            f"Input contains unexpected columns: {extra}. "
            f"Expected only: {features}",
            column=extra[0],
        )

    # Reorder to training order and restore categorical dtypes.
    df = df[features].copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    return df


# --------------------------------------------------------------------------- #
# Prediction                                                                   #
# --------------------------------------------------------------------------- #

def predict_dataframe(
    df: pd.DataFrame,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Run the model on a validated pandas DataFrame.

    Args:
        df: A frame already validated by :func:`validate_input` (feature
            columns only, in training order).
        bundle: Loaded model bundle.

    Returns:
        A dict with ``"predictions"`` (list) and, for classifiers,
        ``"probabilities"`` (list of lists) and ``"classes"`` (list).
    """
    model = bundle["model"]
    classes: np.ndarray | None = bundle.get("classes")
    task_type: str = bundle["task_type"]
    is_clf = task_type != "tabular_regression"

    preds_raw = model.predict(df)

    result: dict[str, Any] = {}
    if is_clf and classes is not None:
        # Map integer indices back to original class labels.
        pred_labels = [str(classes[int(p)]) for p in preds_raw]
        result["predictions"] = pred_labels
        result["classes"] = [str(c) for c in classes]
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                result["probabilities"] = proba.tolist()
            except Exception:
                pass
    elif is_clf:
        # Text classification bundles store label strings directly.
        result["predictions"] = [str(p) for p in preds_raw]
    else:
        result["predictions"] = [float(p) for p in preds_raw]

    return result


# --------------------------------------------------------------------------- #
# Batch (file) scoring                                                         #
# --------------------------------------------------------------------------- #

def score_parquet_chunked(
    src: Path,
    bundle: dict[str, Any],
    *,
    max_mem_mb: int = 0,
) -> pl.DataFrame:
    """Score a Parquet file in CHUNK_ROWS chunks and return the full result.

    Memory is bounded by the chunk size plus model overhead.  For huge files
    the caller should stream the result instead of collecting it in RAM.

    Args:
        src: Path to the (already-ingested) Parquet file.
        bundle: Loaded model bundle.
        max_mem_mb: Not enforced here; the caller's sandbox already enforces
            EXEC_MAX_MEM_MB for training.  Provided for future watchdog use.

    Returns:
        A Polars DataFrame with one ``prediction`` column (and optionally
        probability columns ``prob_<class>``).

    Raises:
        InferenceError: On schema mismatch.
    """
    features: list[str] = bundle["features"]
    classes: list[str] | None = None
    if bundle.get("classes") is not None:
        classes = [str(c) for c in bundle["classes"]]

    preds_chunks: list[pl.Series] = []
    proba_chunks: list[np.ndarray] = []
    total_rows = pl.scan_parquet(src).select(pl.len()).collect().item()

    for offset in range(0, total_rows, CHUNK_ROWS):
        chunk_lf = pl.scan_parquet(src).slice(offset, CHUNK_ROWS)
        chunk_pdf = chunk_lf.collect().to_pandas()
        chunk_validated = validate_input(chunk_pdf, bundle)
        result = predict_dataframe(chunk_validated, bundle)
        preds_chunks.append(pl.Series("prediction", result["predictions"]))
        if "probabilities" in result:
            proba_chunks.append(np.array(result["probabilities"]))

    predictions = pl.concat(preds_chunks)
    out = pl.DataFrame({"prediction": predictions})
    if proba_chunks and classes:
        proba_all = np.vstack(proba_chunks)
        for i, cls in enumerate(classes):
            out = out.with_columns(pl.Series(f"prob_{cls}", proba_all[:, i]))

    return out


def score_file(
    src: Path,
    bundle: dict[str, Any],
    *,
    output_format: str = "parquet",
) -> bytes:
    """Score *src* and return the result as bytes (Parquet or CSV).

    Args:
        src: Ingested Parquet file path.
        bundle: Loaded model bundle.
        output_format: ``"parquet"`` or ``"csv"``.

    Returns:
        Serialised scored file.
    """
    result_df = score_parquet_chunked(src, bundle)
    buf = io.BytesIO()
    if output_format == "csv":
        buf.write(result_df.write_csv().encode())
    else:
        result_df.write_parquet(buf)
    return buf.getvalue()
