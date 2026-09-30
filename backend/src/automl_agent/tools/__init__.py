"""Data tools: ingest, profiling, splits and heuristics."""

from .dataset_profiler import load_table, profile_dataframe, profile_file, profile_lazy, scan_table
from .heuristics import guess_task_spec
from .ingest import IngestError, ingest_to_parquet, is_supported
from .splits import SplitInfo, ensure_split

__all__ = [
    "IngestError",
    "SplitInfo",
    "ensure_split",
    "guess_task_spec",
    "ingest_to_parquet",
    "is_supported",
    "load_table",
    "profile_dataframe",
    "profile_file",
    "profile_lazy",
    "scan_table",
]
