"""Dataset profile: the compact, statistics-only view of a dataset that agents reason over."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ColumnKind = Literal["numeric", "categorical", "text", "datetime", "identifier", "boolean"]
ScaleTier = Literal["small", "medium", "large"]

SMALL_MAX_ROWS = 100_000
MEDIUM_MAX_ROWS = 2_000_000


def scale_tier_for(n_rows: int) -> ScaleTier:
    """Bucket a row count into the tier that drives model choice and grounding."""
    if n_rows < SMALL_MAX_ROWS:
        return "small"
    return "medium" if n_rows < MEDIUM_MAX_ROWS else "large"


class ColumnProfile(BaseModel):
    """Statistics for one column."""

    name: str
    dtype: str
    kind: ColumnKind
    n_unique: int
    n_missing: int
    sample_values: list[str] = Field(default_factory=list)
    mean_length: float | None = None
    top_values: dict[str, float] | None = Field(
        default=None, description="Share of the most frequent values (low-cardinality columns only)"
    )


class DatasetProfile(BaseModel):
    """Summary of a dataset; never contains raw rows beyond a few sample values."""

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    guessed_target: str | None = None
    text_columns: list[str] = Field(default_factory=list)
    size_bytes: int = 0
    memory_estimate_mb: float = 0.0
    scale_tier: ScaleTier = "small"
    approximate_counts: bool = False

    def column(self, name: str) -> ColumnProfile | None:
        """Return the profile of column `name`, if it exists."""
        return next((c for c in self.columns if c.name == name), None)

    def compact(self, max_cols: int = 50) -> dict:
        """Smaller representation to embed in LLM prompts."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "scale_tier": self.scale_tier,
            "memory_estimate_mb": round(self.memory_estimate_mb, 1),
            "guessed_target": self.guessed_target,
            "text_columns": self.text_columns,
            "columns": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "dtype": c.dtype,
                    "n_unique": c.n_unique,
                    "n_missing": c.n_missing,
                    "sample": c.sample_values[:3],
                    **({"top_values": dict(list(c.top_values.items())[:5])} if c.top_values else {}),
                }
                for c in self.columns[:max_cols]
            ],
        }


class DatasetOut(BaseModel):
    """API representation of a registered dataset."""

    id: str
    filename: str
    source: str = "upload"
    created_at: datetime
    profile: DatasetProfile


class DatasetRegister(BaseModel):
    """Request body for registering a dataset that already lives on disk or at a URL."""

    path: str | None = Field(default=None, description="Server-side file path (CSV/TSV/Parquet/JSONL)")
    url: str | None = Field(default=None, description="http(s) URL to download")
    name: str | None = Field(default=None, description="Display name; defaults to the file name")
