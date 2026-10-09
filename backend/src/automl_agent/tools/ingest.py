"""Convert any supported tabular file into a single zstd-compressed Parquet file.

Parquet is the only format the rest of the pipeline reads. Conversion streams
the source in batches (Polars' streaming engine for text formats, PyArrow for
Parquet) and appends each batch to the output, so multi-gigabyte files are
converted with bounded memory and the caller can be told how far along it is.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".tab", ".txt", ".parquet", ".pq", ".jsonl", ".ndjson", ".json"}
INFER_ROWS = 100_000
ROW_GROUP_SIZE = 100_000
BATCH_ROWS = 500_000
COPY_CHUNK = 16 * 1024 * 1024

ProgressFn = Callable[[str, float | None, str], None]
"""`progress(phase, fraction or None, message)`; phases are "decompress" and "convert"."""


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


def _noop(phase: str, fraction: float | None, message: str) -> None:
    pass


def _decompress(path: Path, workdir: Path, progress: ProgressFn) -> Path:
    out = workdir / path.name.removesuffix(".gz")
    total = max(path.stat().st_size, 1)
    with path.open("rb") as raw, gzip.GzipFile(fileobj=raw) as src, out.open("wb") as dst:
        while chunk := src.read(COPY_CHUNK):
            dst.write(chunk)
            progress("decompress", min(raw.tell() / total, 1.0), f"Decompressed {_gb(dst.tell())}")
    return out


def _scan(path: Path, suffix: str, *, infer_rows: int | None) -> pl.LazyFrame:
    if suffix in {".jsonl", ".ndjson"}:
        return pl.scan_ndjson(path, infer_schema_length=infer_rows)
    separator = "\t" if suffix in {".tsv", ".tab"} else ","
    return pl.scan_csv(path, separator=separator, infer_schema_length=infer_rows, encoding="utf8-lossy")


def _estimate_rows(path: Path) -> int:
    """Rough row count of a line-based file, from the line density of its first 4 MB."""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(4 * 1024 * 1024)
    lines = head.count(b"\n")
    if len(head) >= size or lines == 0:
        return max(lines, 1)
    return max(int(size / (len(head) / lines)), 1)


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e8 else f"{n_bytes / 1e6:.1f} MB"


class _Writer:
    """Appends Arrow batches to one Parquet file, opening it on the first batch."""

    def __init__(self, dest: Path) -> None:
        self.dest = dest
        self.rows = 0
        self._writer: pq.ParquetWriter | None = None

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.dest, table.schema, compression="zstd")
        self._writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        self.rows += table.num_rows

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        elif not self.dest.exists():
            pq.write_table(pa.table({}), self.dest)


def _convert_parquet(src: Path, dest: Path, progress: ProgressFn) -> None:
    source = pq.ParquetFile(src)
    total = max(source.metadata.num_rows, 1)
    writer = _Writer(dest)
    try:
        for batch in source.iter_batches(batch_size=BATCH_ROWS):
            writer.write(pa.Table.from_batches([batch]))
            progress("convert", writer.rows / total, f"{writer.rows:,} of {total:,} rows")
    finally:
        writer.close()


def _convert_text(src: Path, suffix: str, dest: Path, progress: ProgressFn) -> None:
    """Stream a CSV/TSV/JSONL file into Parquet, retrying with full schema inference on a late type change."""
    estimate = _estimate_rows(src)
    for infer_rows in (INFER_ROWS, None):
        if infer_rows is None:
            progress("convert", 0.0, "Type changed late in the file; re-reading with full type inference")
        writer = _Writer(dest)
        try:
            for frame in _scan(src, suffix, infer_rows=infer_rows).collect_batches(chunk_size=BATCH_ROWS):
                writer.write(frame.to_arrow(compat_level=pl.CompatLevel.oldest()))
                fraction = min(writer.rows / estimate, 0.99)
                progress("convert", fraction, f"{writer.rows:,} rows (~{estimate:,} expected)")
            writer.close()
            return
        except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
            writer.close()
            if infer_rows is None:
                raise IngestError(f"Could not parse file: {e}") from e


def ingest_to_parquet(src: Path, dest: Path, progress: ProgressFn | None = None) -> Path:
    """Convert `src` to Parquet at `dest` and return `dest`.

    Schema inference first looks at the first 100k rows; if a later row
    contradicts it (a common CSV problem), the conversion is retried with a
    full-file inference pass. `progress` is called after every batch.

    Raises:
        IngestError: Unsupported extension or unreadable content.
    """
    report = progress or _noop
    suffix = source_suffix(src)
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"Unsupported file type '{suffix or src.name}'")
    dest.parent.mkdir(parents=True, exist_ok=True)
    decompressed: Path | None = None
    try:
        if src.suffix.lower() == ".gz":
            decompressed = _decompress(src, dest.parent, report)
            src = decompressed
        if suffix in {".parquet", ".pq"}:
            _convert_parquet(src, dest, report)
        elif suffix == ".json":
            pl.read_json(src).write_parquet(dest, compression="zstd", row_group_size=ROW_GROUP_SIZE)
        else:
            _convert_text(src, suffix, dest, report)
        rows = pq.ParquetFile(dest).metadata.num_rows
        if rows == 0:
            raise IngestError("File contains no rows")
        report("convert", 1.0, f"{rows:,} rows")
        return dest
    except (OSError, pl.exceptions.PolarsError, pa.ArrowException) as e:
        raise IngestError(f"Could not read file: {e}") from e
    finally:
        if decompressed is not None:
            decompressed.unlink(missing_ok=True)
