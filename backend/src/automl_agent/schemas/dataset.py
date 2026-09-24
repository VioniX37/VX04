from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ColumnKind = Literal["numeric", "categorical", "text", "datetime", "identifier", "boolean"]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: ColumnKind
    n_unique: int
    n_missing: int
    sample_values: list[str] = Field(default_factory=list)
    mean_length: float | None = None  # for string columns


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    guessed_target: str | None = None
    text_columns: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def compact(self, max_cols: int = 50) -> dict:
        """Smaller representation to embed in LLM prompts."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
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
                }
                for c in self.columns[:max_cols]
            ],
        }


class DatasetOut(BaseModel):
    id: str
    filename: str
    created_at: datetime
    profile: DatasetProfile
