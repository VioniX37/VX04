"""Ingest, profiling and split tests for the large-data pipeline."""

import gzip
import shutil

import numpy as np
import polars as pl
import pytest

from automl_agent.schemas.dataset import scale_tier_for
from automl_agent.tools import IngestError, ensure_split, ingest_to_parquet, profile_file
from automl_agent.tools.splits import RAND_COL, SPLIT_COL, TEST, TRAIN, VALID


@pytest.fixture
def frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 3000
    return pl.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "label": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        }
    )


@pytest.mark.parametrize("fmt", ["csv", "tsv", "parquet", "jsonl", "csv.gz"])
def test_ingest_formats_roundtrip(tmp_path, frame, fmt):
    src = tmp_path / f"data.{fmt}"
    if fmt == "csv":
        frame.write_csv(src)
    elif fmt == "tsv":
        frame.write_csv(src, separator="\t")
    elif fmt == "parquet":
        frame.write_parquet(src)
    elif fmt == "jsonl":
        frame.write_ndjson(src)
    else:
        plain = tmp_path / "plain.csv"
        frame.write_csv(plain)
        with plain.open("rb") as f_in, gzip.open(src, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    out = ingest_to_parquet(src, tmp_path / "out" / "data.parquet")
    back = pl.read_parquet(out)
    assert back.shape == frame.shape and back.columns == frame.columns


def test_ingest_recovers_from_late_schema_change(tmp_path):
    # An integer column that turns into text after the inference window.
    rows = ["a,b"] + [f"{i},{i}" for i in range(120_000)] + ["oops,1"]
    src = tmp_path / "late.csv"
    src.write_text("\n".join(rows))
    out = ingest_to_parquet(src, tmp_path / "o" / "data.parquet")
    assert pl.read_parquet(out)["a"].dtype == pl.String


def test_ingest_rejects_unsupported(tmp_path):
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(IngestError):
        ingest_to_parquet(bad, tmp_path / "o.parquet")


def test_profile_parquet(tmp_path, frame):
    path = tmp_path / "d.parquet"
    frame.write_parquet(path)
    p = profile_file(path)
    assert p.n_rows == 3000 and p.scale_tier == "small" and p.size_bytes > 0
    assert p.column("id").kind == "identifier"
    assert p.guessed_target == "label"
    assert abs(p.column("label").top_values["no"] - 0.8) < 0.05


def test_scale_tiers():
    assert scale_tier_for(10) == "small"
    assert scale_tier_for(500_000) == "medium"
    assert scale_tier_for(5_000_000) == "large"


def test_split_is_stratified_nested_and_reused(tmp_path, frame):
    path = tmp_path / "d.parquet"
    frame.with_columns(
        pl.when(pl.col("id") < 30).then(None).otherwise(pl.col("label")).alias("label")
    ).write_parquet(path)
    info = ensure_split(path, "label", stratify=True)
    assert info.n_train + info.n_valid + info.n_test == 3000 - 30  # missing targets excluded
    split = pl.read_parquet(info.path)
    data = pl.concat([pl.read_parquet(path), split], how="horizontal")
    for part in (TRAIN, VALID, TEST):
        share = data.filter(pl.col(SPLIT_COL) == part)["label"].eq("yes").mean()
        assert abs(share - 0.2) < 0.04  # stratified
    small = set(data.filter((pl.col(SPLIT_COL) == TRAIN) & (pl.col(RAND_COL) < 0.1))["id"])
    large = set(data.filter((pl.col(SPLIT_COL) == TRAIN) & (pl.col(RAND_COL) < 0.5))["id"])
    assert small < large  # nested subsamples
    assert ensure_split(path, "label", stratify=True).path == info.path  # reused
