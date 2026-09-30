"""Generate a large synthetic tabular dataset as Parquet, in chunks (bounded memory).

    python data/samples/generate_large.py                      # 5M rows (~0.5 GB Parquet)
    python data/samples/generate_large.py --rows 10000000 --out big.parquet

A binary-classification problem with numeric and categorical features, missing
values, a weak interaction effect and class imbalance, so model choice matters
and fidelity-based search has real signal to find. Used by the scale test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REGIONS = np.array(["north", "south", "east", "west", "central"])
PLANS = np.array(["basic", "standard", "premium", "enterprise"])
CHANNELS = np.array([f"ch_{i:03d}" for i in range(300)])  # high-cardinality categorical


def make_chunk(rng: np.random.Generator, n: int, start_id: int) -> pa.Table:
    """Create `n` rows of the synthetic problem."""
    x = rng.normal(size=(n, 12)).astype(np.float32)
    region = rng.integers(0, len(REGIONS), n)
    plan = rng.choice(len(PLANS), n, p=[0.5, 0.3, 0.15, 0.05])
    channel = rng.integers(0, len(CHANNELS), n)
    tenure = rng.gamma(2.0, 12.0, n).astype(np.float32)
    logit = (
        -1.6 + 0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3] + 0.3 * np.sin(3 * x[:, 4])
        + 0.4 * (region == 2) - 0.5 * (plan >= 2) + 0.002 * (channel % 17) - 0.02 * tenure
    )
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    cols = {"row_id": np.arange(start_id, start_id + n, dtype=np.int64)}
    for i in range(12):
        col = x[:, i].copy()
        if i in (5, 9):
            col[rng.random(n) < 0.08] = np.nan
        cols[f"f{i:02d}"] = col
    cols.update(
        tenure_months=tenure,
        region=REGIONS[region],
        plan=PLANS[plan],
        channel=CHANNELS[channel],
        converted=np.where(y, "yes", "no"),
    )
    return pa.table(cols)


def main() -> None:
    """Parse arguments and write the dataset."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "large_conversion.parquet")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    writer = None
    written = 0
    while written < args.rows:
        n = min(args.chunk, args.rows - written)
        table = make_chunk(rng, n, written)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=100_000)
        written += n
        print(f"\r{written:,}/{args.rows:,} rows", end="", flush=True)
    if writer is not None:
        writer.close()
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
