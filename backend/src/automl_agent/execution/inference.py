"""Model-serving counterpart to the training templates.

Loads the bundle written by a template (``{"model", "classes", "features",
"task_type", "target", "drop_columns", "text_column", "dtypes", "categories",
"ordinal_columns", "config"}``), replays the *same* feature preparation the
training script used, and returns predictions (and probabilities for classifiers).

Design goals
------------
- Training and serving share one code path: the template records the dtypes and
  category levels it trained with, and :func:`prepare_features` replays them, so a
  category is encoded identically at training and serving time.
- Validate inputs early and return structured errors, never a 500.
- Score large files in chunks so memory stays bounded.
- Standalone: this module imports only joblib, numpy, pandas and polars, so it is
  shipped verbatim inside the exported bundle and used by its ``predict.py``.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

BUNDLE_KEYS_REQUIRED = {"model", "features", "task_type"}
CHUNK_ROWS = 100_000
_BOOL_STRINGS = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}


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
        raise InferenceError(
            "No trained model found for this run. The run may have failed or not yet saved a model."
        )
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


def is_text_bundle(bundle: dict[str, Any]) -> bool:
    """True for bundles written by the text-classification template."""
    return bundle.get("task_type") == "text_classification"


def build_schema(bundle: dict[str, Any]) -> dict[str, str]:
    """Return a ``{column_name: kind}`` schema (``numeric``, ``category``, ``text`` or ``unknown``).

    Uses the training dtypes recorded in the bundle; bundles written before those
    were recorded fall back to the fitted sklearn preprocessor, then to ``unknown``.
    """
    features: list[str] = bundle["features"]
    if is_text_bundle(bundle):
        return dict.fromkeys(features, "text")
    recorded: dict[str, str] = bundle.get("dtypes") or {}
    kinds: dict[str, str] = {}
    for col, dtype in recorded.items():
        kinds[col] = "category" if dtype == "category" else "numeric"
    if not recorded:
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline

            model = bundle["model"]
            prep = model.named_steps.get("prep") if isinstance(model, Pipeline) else None
            if isinstance(prep, ColumnTransformer):
                for name, _, cols in prep.transformers_:
                    if name in ("num", "cat"):
                        for c in cols:
                            kinds[c] = "numeric" if name == "num" else "category"
        except Exception:
            pass
    return {col: kinds.get(col, "unknown") for col in features}


def _to_numeric(s: pd.Series, col: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("float64")
    try:
        return pd.to_numeric(s, errors="raise")
    except (ValueError, TypeError):
        lowered = s.astype(str).str.strip().str.lower()
        if s.notna().all() and lowered.isin(_BOOL_STRINGS).all():
            return lowered.map(_BOOL_STRINGS).astype("float64")
        bad = s[pd.to_numeric(s, errors="coerce").isna() & s.notna()].iloc[0]
        raise InferenceError(f"Column '{col}' must be numeric; got {bad!r}.", column=col) from None


def _as_category(s: pd.Series, levels: list[Any]) -> pd.Categorical:
    """Encode *s* with the training levels; values unseen in training become missing."""
    if levels and all(isinstance(v, str) for v in levels):
        s = s.where(s.isna(), s.astype(str))
    s = s.astype(object)
    return pd.Categorical(s.where(s.isin(levels)), categories=levels)


def prepare_features(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Replay the training script's feature preparation on *df* (feature columns, training order)."""
    df = df.copy()
    dtypes: dict[str, str] = bundle.get("dtypes") or {}
    categories: dict[str, list[Any]] = bundle.get("categories") or {}
    ordinal = set(bundle.get("ordinal_columns") or [])
    if not dtypes:  # bundles written before dtypes were recorded
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category")
        return df
    for col in df.columns:
        dtype = dtypes.get(col)
        if col in categories:
            cat = _as_category(df[col], categories[col])
            if col in ordinal:  # HistGB fallback: integer codes of the training levels
                df[col] = pd.Series(cat.codes, index=df.index).replace(-1, np.nan).astype("float32")
            else:
                df[col] = pd.Series(cat, index=df.index)
        elif dtype:
            values = _to_numeric(df[col], col)
            if dtype.startswith("float") or values.notna().all():
                values = values.astype(dtype)
            df[col] = values
    return df


def validate_input(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Check that *df* has the required columns and return a frame ready for prediction.

    The target, the columns the run excluded and the training-data columns the model
    did not use are dropped silently; any other missing or extra column is rejected so
    features can never be silently misaligned.

    Args:
        df: Raw input dataframe.
        bundle: Loaded model bundle.

    Returns:
        A new dataframe containing only the feature columns in training order,
        prepared exactly as the training script prepared them.

    Raises:
        InferenceError: If a required column is missing, an unexpected column is
            present, or a value cannot be converted to the training dtype.
    """
    features: list[str] = bundle["features"]
    ignorable = {bundle.get("target"), *bundle.get("drop_columns", []), *bundle.get("ignored_columns", [])}
    df = df.drop(columns=[c for c in df.columns if c in ignorable])

    for col in features:
        if col not in df.columns:
            raise InferenceError(
                f"Required column '{col}' is missing from the input. Expected columns: {features}",
                column=col,
            )
    extra = sorted(set(df.columns) - set(features))
    if extra:
        raise InferenceError(
            f"Input contains unexpected columns: {extra}. Expected only: {features}",
            column=extra[0],
        )
    df = df[features]
    if is_text_bundle(bundle):
        return df
    return prepare_features(df, bundle)


def predict_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    """Run the model on a frame returned by :func:`validate_input`.

    Returns:
        A dict with ``"predictions"`` (list) and, for classifiers,
        ``"classes"`` and ``"probabilities"`` (list of lists, in ``classes`` order).
    """
    model = bundle["model"]
    is_clf = bundle["task_type"] != "tabular_regression"

    if is_text_bundle(bundle):
        text_col = bundle.get("text_column") or bundle["features"][0]
        X: Any = df[text_col].fillna("").astype(str).tolist()
        if bundle.get("vectorizer") is not None:  # sgd_hashing keeps the vectorizer separate
            X = bundle["vectorizer"].transform(X)
        preds = model.predict(X)
        result: dict[str, Any] = {"predictions": [str(p) for p in preds]}
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            result["classes"] = [str(c) for c in model.classes_]
            result["probabilities"] = np.asarray(model.predict_proba(X)).tolist()
        return result

    preds = model.predict(df)
    if not is_clf:
        return {"predictions": [float(p) for p in np.asarray(preds).ravel()]}
    classes = bundle.get("classes")
    if classes is None:
        return {"predictions": [str(p) for p in preds]}
    result = {
        "predictions": [str(classes[int(p)]) for p in np.asarray(preds).ravel()],
        "classes": [str(c) for c in classes],
    }
    if hasattr(model, "predict_proba"):
        with contextlib.suppress(Exception):  # e.g. SVC fitted without probability estimates
            result["probabilities"] = np.asarray(model.predict_proba(df)).tolist()
    return result


def score_parquet_chunked(src: Path, bundle: dict[str, Any]) -> pl.DataFrame:
    """Score a Parquet file in ``CHUNK_ROWS`` chunks and return the predictions in row order.

    Only one chunk of input is in memory at a time; the output holds one
    ``prediction`` column plus a ``prob_<class>`` column per class for classifiers.

    Raises:
        InferenceError: On schema mismatch or an unconvertible value.
    """
    frames: list[pl.DataFrame] = []
    lf = pl.scan_parquet(src)
    total_rows = lf.select(pl.len()).collect().item()
    for offset in range(0, total_rows, CHUNK_ROWS):
        chunk = validate_input(lf.slice(offset, CHUNK_ROWS).collect().to_pandas(), bundle)
        result = predict_dataframe(chunk, bundle)
        cols: dict[str, Any] = {"prediction": result["predictions"]}
        if "probabilities" in result:
            proba = np.asarray(result["probabilities"])
            for i, cls in enumerate(result["classes"]):
                cols[f"prob_{cls}"] = proba[:, i]
        frames.append(pl.DataFrame(cols))
    if not frames:
        return pl.DataFrame({"prediction": []})
    return pl.concat(frames)


def score_file(src: Path, bundle: dict[str, Any], *, output_format: str = "parquet") -> bytes:
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
