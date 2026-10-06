"""Fixed train/validation/test assignment shared by every plan and fidelity.

For a (dataset, target) pair we write ``splits/<target>-<seed>.parquet`` with
two columns aligned row-by-row with the data file:

* ``__split`` - 0 = train, 1 = valid, 2 = test, 255 = excluded (missing target)
* ``__r``     - an independent uniform [0, 1) draw per row

Classification splits are stratified. ``__r`` makes subsampling trivial and
*nested*: the rows with ``__r < f`` at a small fidelity are a subset of those
at any larger fidelity, which keeps successive-halving comparisons fair.
The test split is never shown to the agents; it is used only for the final,
reported score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

TRAIN, VALID, TEST, EXCLUDED = 0, 1, 2, 255
SPLIT_COL, RAND_COL = "__split", "__r"


@dataclass(frozen=True)
class SplitInfo:
    """Location and sizes of a split file."""

    path: Path
    n_train: int
    n_valid: int
    n_test: int


def _assign(n: int, fractions: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Return split ids for `n` rows with (valid, test) fractions, randomly permuted."""
    n_valid = int(round(n * fractions[0]))
    n_test = int(round(n * fractions[1]))
    if n >= 3:
        n_valid, n_test = max(n_valid, 1), max(n_test, 1)
    ids = np.full(n, TRAIN, dtype=np.uint8)
    ids[:n_valid] = VALID
    ids[n_valid : n_valid + n_test] = TEST
    rng.shuffle(ids)
    return ids


def ensure_split(
    data_path: Path,
    target: str,
    *,
    stratify: bool,
    valid_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    temporal: bool = False,
    time_column: str | None = None,
    series_id_columns: list[str] | None = None,
) -> SplitInfo:
    """Create (or reuse) the split file for `target` next to `data_path`.

    If `temporal` is True, splits into contiguous non-shuffled time blocks
    (train -> valid -> test) and assigns `__r` as suffix-window rankings so that
    grounded verification low-fidelity rungs evaluate on the most recent data.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", target)[:60]
    if temporal and time_column:
        clean_time = re.sub(r"[^A-Za-z0-9_.-]", "_", time_column)[:30]
        mode_tag = f"-temporal-{clean_time}"
    elif temporal:
        mode_tag = "-temporal"
    else:
        mode_tag = ""
    out = data_path.parent / "splits" / f"{safe}{mode_tag}-{seed}.parquet"
    if not out.exists():
        if temporal:
            schema = pl.scan_parquet(data_path).collect_schema()
            t_col = time_column if time_column and time_column in schema.names() else None
            if not t_col:
                for name, dtype in schema.items():
                    if dtype.is_temporal() or name.lower() in ("date", "datetime", "timestamp", "time", "ds"):
                        t_col = name
                        break
            cols_to_scan = [target] + ([t_col] if t_col and t_col != target else [])
            df = pl.scan_parquet(data_path).select(cols_to_scan).collect()
            n = len(df)
            split = np.full(n, EXCLUDED, dtype=np.uint8)
            r = np.full(n, 1.0, dtype=np.float32)

            y = df[target]
            valid_mask = ~y.is_null().to_numpy()
            if t_col:
                valid_mask = valid_mask & (~df[t_col].is_null().to_numpy())
            idx = np.flatnonzero(valid_mask)
            if len(idx) > 0:
                if t_col:
                    t_vals = df[t_col].to_numpy()[idx]
                    try:
                        order = np.argsort(t_vals, kind="stable")
                        idx_sorted = idx[order]
                    except Exception:
                        idx_sorted = idx
                else:
                    idx_sorted = idx

                n_tot = len(idx_sorted)
                n_valid = int(round(n_tot * valid_fraction))
                n_test = int(round(n_tot * test_fraction))
                if n_tot >= 3:
                    n_valid, n_test = max(n_valid, 1), max(n_test, 1)
                n_train = max(n_tot - n_valid - n_test, 1)

                train_idx = idx_sorted[:n_train]
                valid_idx = idx_sorted[n_train : n_train + n_valid]
                test_idx = idx_sorted[n_train + n_valid :]

                split[train_idx] = TRAIN
                split[valid_idx] = VALID
                split[test_idx] = TEST

                # For suffix windows, most recent row in block has smallest __r
                for part_idx in (train_idx, valid_idx, test_idx):
                    n_part = len(part_idx)
                    if n_part > 0:
                        ranks_from_end = np.arange(n_part - 1, -1, -1, dtype=np.float32)
                        r[part_idx] = (ranks_from_end + 0.5) / n_part

            out.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({SPLIT_COL: split, RAND_COL: r}).write_parquet(
                out, compression="zstd", row_group_size=100_000
            )
        else:
            y = pl.scan_parquet(data_path).select(pl.col(target)).collect().to_series()
            n = len(y)
            rng = np.random.default_rng(seed)
            split = np.full(n, EXCLUDED, dtype=np.uint8)
            valid_mask = ~y.is_null().to_numpy()
            idx = np.flatnonzero(valid_mask)
            fractions = (valid_fraction, test_fraction)
            if stratify:
                _, inverse = np.unique(
                    y.filter(pl.Series(valid_mask)).cast(pl.String).to_numpy(), return_inverse=True
                )
                for cls in range(inverse.max() + 1 if len(inverse) else 0):
                    members = idx[inverse == cls]
                    split[members] = _assign(len(members), fractions, rng)
            else:
                split[idx] = _assign(len(idx), fractions, rng)
            out.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({SPLIT_COL: split, RAND_COL: rng.random(n, dtype=np.float32)}).write_parquet(
                out, compression="zstd", row_group_size=100_000
            )

    counts = pl.scan_parquet(out).group_by(SPLIT_COL).len().collect()
    by_id = dict(zip(counts[SPLIT_COL].to_list(), counts["len"].to_list(), strict=True))
    return SplitInfo(out, by_id.get(TRAIN, 0), by_id.get(VALID, 0), by_id.get(TEST, 0))
