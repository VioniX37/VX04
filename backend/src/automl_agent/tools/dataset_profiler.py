"""Scalable dataset profiling with Polars' lazy engine.

Counts, null counts and distinct counts are computed in a single streaming
pass (distinct counts become HyperLogLog approximations above 2M rows).
Text/identifier detection and value samples use an evenly-strided sample, so
profiling a 10M-row file takes seconds and little memory.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from automl_agent.schemas.dataset import ColumnKind, ColumnProfile, DatasetProfile, scale_tier_for

from .ingest import source_suffix

_TARGET_NAMES = (
    "target", "label", "labels", "class", "y", "outcome", "churn", "sentiment", "category",
    "quality", "price", "species",
)  # fmt: skip
_ID_HINTS = ("id", "uuid", "index", "key")
EXACT_UNIQUE_MAX_ROWS = 2_000_000
TOP_VALUES_MAX_UNIQUE = 50


def scan_table(path: Path) -> pl.LazyFrame:
    """Lazily scan a Parquet/CSV/TSV/JSONL file."""
    suffix = source_suffix(path)
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.scan_ndjson(path)
    if suffix == ".json":
        return pl.read_json(path).lazy()
    separator = "\t" if suffix in {".tsv", ".tab"} else ","
    return pl.scan_csv(path, separator=separator, infer_schema_length=10_000, encoding="utf8-lossy")


def load_table(path: Path):
    """Load a whole table into pandas (small files only; kept for scripts and tests)."""
    return scan_table(path).collect().to_pandas()


def _kind(name: str, dtype: pl.DataType, sample: pl.Series, n_unique: int, n_rows: int) -> ColumnKind:
    lname = name.lower()
    unique_ratio = n_unique / max(n_rows, 1)
    looks_like_id = lname in _ID_HINTS or lname.endswith(("_id", "id"))
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_temporal():
        return "datetime"
    if dtype.is_numeric():
        return "identifier" if looks_like_id and unique_ratio >= 0.95 else "numeric"
    if looks_like_id and unique_ratio >= 0.95:
        return "identifier"
    strings = sample.drop_nulls().cast(pl.String)
    if len(strings):
        mean_len = strings.str.len_chars().mean() or 0
        has_space = strings.str.contains(" ").mean() or 0
        if mean_len >= 30 and has_space > 0.5:
            return "text"
    if unique_ratio > 0.98 and n_rows > 50:
        return "identifier"
    return "categorical"


def profile_lazy(lf: pl.LazyFrame, *, size_bytes: int = 0, sample_rows: int = 100_000) -> DatasetProfile:
    """Profile a lazy frame (see module docstring for the strategy)."""
    schema = lf.collect_schema()
    names = list(schema.names())
    n_rows = int(lf.select(pl.len()).collect().item())
    exact = n_rows <= EXACT_UNIQUE_MAX_ROWS

    exprs = []
    for i, name in enumerate(names):
        col = pl.col(name)
        exprs.append(col.null_count().alias(f"n{i}"))
        exprs.append((col.n_unique() if exact else col.approx_n_unique()).alias(f"u{i}"))
    stats = lf.select(exprs).collect().row(0, named=True) if names else {}

    step = max(1, n_rows // max(sample_rows, 1))
    sample = lf.gather_every(step).head(sample_rows).collect()

    columns: list[ColumnProfile] = []
    for i, name in enumerate(names):
        dtype = schema[name]
        s = sample.get_column(name)
        n_unique = int(stats.get(f"u{i}", 0))
        kind = _kind(name, dtype, s, n_unique, n_rows)
        non_null = s.drop_nulls()
        mean_length = None
        if dtype == pl.String and len(non_null):
            mean_length = round(float(non_null.str.len_chars().mean() or 0), 1)
        top_values = None
        low_cardinality = kind in ("categorical", "boolean") or (kind == "numeric" and n_unique <= 20)
        if low_cardinality and n_unique <= TOP_VALUES_MAX_UNIQUE and len(non_null):
            counts = non_null.cast(pl.String).value_counts(sort=True).head(10)
            total = len(non_null)
            top_values = {str(v): round(c / total, 4) for v, c in counts.iter_rows()}
        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(dtype),
                kind=kind,
                n_unique=n_unique,
                n_missing=int(stats.get(f"n{i}", 0)),
                sample_values=[str(v)[:80] for v in non_null.head(3).to_list()],
                mean_length=mean_length,
                top_values=top_values,
            )
        )

    text_cols = [c.name for c in columns if c.kind == "text"]
    guessed = next((c.name for c in columns if c.name.lower() in _TARGET_NAMES), None)
    if guessed is None:
        candidates = [c for c in columns if c.kind in ("categorical", "numeric", "boolean")]
        guessed = candidates[-1].name if candidates else None

    memory_mb = sample.estimated_size("mb") * (n_rows / max(len(sample), 1)) if len(sample) else 0.0
    return DatasetProfile(
        n_rows=n_rows,
        n_cols=len(columns),
        columns=columns,
        guessed_target=guessed,
        text_columns=text_cols,
        size_bytes=size_bytes,
        memory_estimate_mb=round(memory_mb, 1),
        scale_tier=scale_tier_for(n_rows),
        approximate_counts=not exact,
    )


def profile_file(path: Path, *, sample_rows: int = 100_000) -> DatasetProfile:
    """Profile a file on disk (Parquet preferred; CSV/TSV/JSONL also work)."""
    return profile_lazy(scan_table(path), size_bytes=path.stat().st_size, sample_rows=sample_rows)


def profile_dataframe(df) -> DatasetProfile:
    """Profile an in-memory pandas or Polars DataFrame."""
    frame = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    return profile_lazy(frame.lazy())
