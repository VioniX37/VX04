"""Download benchmark datasets into ``backend/evaluation/data/<name>/``.

    python -m evaluation.datasets.fetch --list
    python -m evaluation.datasets.fetch --group paper       # Kaggle datasets used by the paper
    python -m evaluation.datasets.fetch higgs nyc_taxi      # large-scale study

Kaggle sources need the Kaggle CLI (``pip install kaggle``) with an API token in
``~/.kaggle/kaggle.json``; competition data additionally requires accepting the
competition rules on kaggle.com once. Check each dataset's license before use;
the ``license`` field below is informational, not legal advice.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Source:
    """Where a dataset comes from and how to normalise it to ``data.csv`` / ``data.parquet``."""

    name: str
    group: str
    kind: str  # "kaggle-dataset" | "kaggle-competition" | "url"
    ref: str  # slug, competition name or URL
    file: str  # file to keep from the download
    output: str = "data.csv"
    columns: list[str] | None = None  # set when the raw file has no header
    rename: dict[str, str] = field(default_factory=dict)
    license: str = "check the source page"


SOURCES = [
    Source("banana_quality", "paper", "kaggle-dataset", "l3llff/banana", "banana_quality.csv",
           rename={"Quality": "Quality"}, license="see Kaggle page"),
    Source("software_defects", "paper", "kaggle-competition", "playground-series-s3e23", "train.csv",
           license="Kaggle competition rules"),
    Source("crab_age", "paper", "kaggle-competition", "playground-series-s3e16", "train.csv",
           license="Kaggle competition rules"),
    Source("ecommerce_text", "paper", "kaggle-dataset", "saurabhshahane/ecommerce-text-classification",
           "ecommerceDataset.csv", columns=["label", "text"], license="CC BY 4.0 (per Kaggle page)"),
    Source("higgs", "large", "url", "https://archive.ics.uci.edu/static/public/280/higgs.zip", "HIGGS.csv.gz",
           output="data.parquet", columns=["label", *[f"f{i:02d}" for i in range(28)]], license="CC BY 4.0"),
    Source("nyc_taxi", "large", "url",
           "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
           "yellow_tripdata_2023-01.parquet", output="data.parquet", license="NYC TLC open data"),
    Source("amazon_polarity", "large", "url",
           "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00000-of-00004.parquet",
           "train-00000-of-00004.parquet", output="data.parquet", license="Apache 2.0 (per dataset card)"),
]  # fmt: skip


def _download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as out:
            for chunk in resp.iter_bytes(8 * 1024 * 1024):
                out.write(chunk)


def _kaggle(kind: str, ref: str, dest: Path) -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit("The Kaggle CLI is required: pip install kaggle (and add ~/.kaggle/kaggle.json)")
    cmd = ["kaggle", "datasets", "download", "-d", ref] if kind == "kaggle-dataset" else [
        "kaggle", "competitions", "download", "-c", ref]  # fmt: skip
    subprocess.run([*cmd, "-p", str(dest)], check=True)


def _extract(raw_dir: Path, wanted: str) -> Path:
    for archive in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
    matches = list(raw_dir.rglob(wanted))
    if not matches:
        raise SystemExit(f"{wanted} not found in {raw_dir}")
    return matches[0]


def fetch(src: Source) -> Path:
    """Download one dataset and write the normalised file; returns its path."""
    target_dir = DATA_DIR / src.name
    out = target_dir / src.output
    if out.exists():
        print(f"{src.name}: already present at {out}")
        return out
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"{src.name}: downloading ({src.license})")
    if src.kind == "url":
        _download(src.ref, raw_dir / Path(src.ref.split("?")[0]).name)
    else:
        _kaggle(src.kind, src.ref, raw_dir)
    raw = _extract(raw_dir, src.file)

    if raw.suffix == ".parquet":
        frame = pl.scan_parquet(raw)
    else:
        frame = pl.scan_csv(raw, has_header=src.columns is None, new_columns=src.columns,
                            infer_schema_length=100_000)  # fmt: skip
    if src.rename:
        frame = frame.rename(src.rename)
    if src.output.endswith(".parquet"):
        frame.sink_parquet(out, compression="zstd")
    else:
        frame.sink_csv(out)
    shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"{src.name}: wrote {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", choices=sorted({s.group for s in SOURCES}))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for s in SOURCES:
            print(f"{s.name:18} {s.group:6} {s.kind:18} {s.ref}")
        return 0
    chosen = [s for s in SOURCES if s.name in args.names or (args.group and s.group == args.group)]
    if not chosen:
        ap.error("name at least one dataset or a --group")
    for src in chosen:
        fetch(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
