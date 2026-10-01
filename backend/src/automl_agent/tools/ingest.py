"""Convert any supported tabular file into a single zstd-compressed Parquet file.

Parquet is the only format the rest of the pipeline reads. Conversion uses
Polars' streaming engine (`scan_* -> sink_parquet`), so multi-gigabyte CSVs
are converted with bounded memory.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import polars as pl

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".tab", ".txt", ".parquet", ".pq", ".jsonl", ".ndjson", ".json"}
INFER_ROWS = 100_000
ROW_GROUP_SIZE = 100_000


class IngestError(ValueError):
    """The file could not be read as a table."""


def source_suffix(path: Path) -> str:
    """Return the data suffix, looking through a trailing ``.gz``."""
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def is_supported(path: Path) -> bool:
    """Whether the file extension is one we can ingest."""
    return source_suffix(path) in SUPPORTED_SUFFIXES


def _decompress(path: Path, workdir: Path) -> Path:
    out = workdir / path.name.removesuffix(".gz")
    with gzip.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out


def _scan(path: Path, suffix: str, *, infer_rows: int | None) -> pl.LazyFrame:
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.scan_ndjson(path, infer_schema_length=infer_rows)
    if suffix == ".json":
        return pl.read_json(path).lazy()
    separator = "\t" if suffix in {".tsv", ".tab"} else ","
    return pl.scan_csv(path, separator=separator, infer_schema_length=infer_rows, encoding="utf8-lossy")


def ingest_to_parquet(src: Path, dest: Path) -> Path:
    """Convert `src` to Parquet at `dest` and return `dest`.

    Schema inference first looks at the first 100k rows; if a later row
    contradicts it (a common CSV problem), the conversion is retried with a
    full-file inference pass.

    Raises:
        IngestError: Unsupported extension or unreadable content.
    """
    suffix = source_suffix(src)
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"Unsupported file type '{suffix or src.name}'")
    dest.parent.mkdir(parents=True, exist_ok=True)
    decompressed: Path | None = None
    if src.suffix.lower() == ".gz":
        decompressed = _decompress(src, dest.parent)
        src = decompressed
    try:
        for infer_rows in (INFER_ROWS, None):
            try:
                _scan(src, suffix, infer_rows=infer_rows).sink_parquet(
                    dest, compression="zstd", row_group_size=ROW_GROUP_SIZE
                )
                break
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                if infer_rows is None:
                    raise IngestError(f"Could not parse file: {e}") from e
        if pl.scan_parquet(dest).select(pl.len()).collect().item() == 0:
            raise IngestError("File contains no rows")
        return dest
    except (OSError, pl.exceptions.PolarsError) as e:
        raise IngestError(f"Could not read file: {e}") from e
    finally:
        if decompressed is not None:
            decompressed.unlink(missing_ok=True)
