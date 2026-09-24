"""Cheap, deterministic dataset profiling that grounds the agents' prompts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from automl_agent.schemas.dataset import ColumnKind, ColumnProfile, DatasetProfile

_TARGET_NAMES = (
    "target",
    "label",
    "labels",
    "class",
    "y",
    "outcome",
    "churn",
    "sentiment",
    "category",
    "quality",
    "price",
    "species",
)
_ID_HINTS = ("id", "uuid", "index", "key")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    return pd.read_csv(path)


def _kind(s: pd.Series, name: str, n_rows: int) -> tuple[ColumnKind, float | None]:
    n_unique = s.nunique(dropna=True)
    lname = name.lower()
    if pd.api.types.is_bool_dtype(s):
        return "boolean", None
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", None
    # Name looks like an id; only treated as one when (almost) every value is unique.
    is_id_name = lname in _ID_HINTS or lname.endswith(("_id", "id"))
    if pd.api.types.is_numeric_dtype(s):
        if is_id_name and n_unique >= 0.95 * n_rows:
            return "identifier", None
        return "numeric", None
    as_str = s.dropna().astype(str)
    mean_len = float(as_str.str.len().mean()) if len(as_str) else 0.0
    if is_id_name and n_unique >= 0.95 * n_rows:
        return "identifier", mean_len
    if mean_len >= 30 and as_str.str.contains(" ").mean() > 0.5:
        return "text", mean_len
    if n_unique == n_rows and n_rows > 50:
        return "identifier", mean_len
    return "categorical", mean_len


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    n_rows = len(df)
    columns: list[ColumnProfile] = []
    for name in df.columns:
        s = df[name]
        kind, mean_len = _kind(s, str(name), n_rows)
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(s.dtype),
                kind=kind,
                n_unique=int(s.nunique(dropna=True)),
                n_missing=int(s.isna().sum()),
                sample_values=[str(v)[:80] for v in s.dropna().head(3).tolist()],
                mean_length=round(mean_len, 1) if mean_len is not None else None,
            )
        )

    text_cols = [c.name for c in columns if c.kind == "text"]
    guessed = next((c.name for c in columns if c.name.lower() in _TARGET_NAMES), None)
    if guessed is None:
        candidates = [c for c in columns if c.kind in ("categorical", "numeric", "boolean")]
        guessed = candidates[-1].name if candidates else None

    return DatasetProfile(
        n_rows=n_rows, n_cols=len(columns), columns=columns, guessed_target=guessed, text_columns=text_cols
    )


def profile_file(path: Path) -> DatasetProfile:
    return profile_dataframe(load_table(path))
