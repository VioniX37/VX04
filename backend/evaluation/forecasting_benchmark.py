"""Score every forecasting family on a dataset's temporal test block.

    python -m evaluation.forecasting_benchmark ../data/samples/store_sales.csv \
        --target sales --time date --series store_id --freq D --horizon 14

Uses the platform's own temporal split (contiguous train/valid/test date blocks) and the
rolling-origin backtest from ``automl_agent.execution.forecasting``, i.e. exactly what a
pipeline run reports for each family. No LLM is involved, so the numbers are deterministic.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from automl_agent.execution import forecasting as fc
from automl_agent.tools.dataset_profiler import scan_table
from automl_agent.tools.splits import ensure_split


def benchmark(
    path: Path, target: str, time_col: str, series: list[str], freq: str, horizon: int
) -> dict[str, dict[str, float]]:
    """sMAPE/MAE/RMSE/MAPE of each family on the test block of ``path``."""
    tmp = Path(tempfile.mkdtemp()) / "data.parquet"
    scan_table(path).sink_parquet(tmp)
    info = ensure_split(tmp, target, stratify=False, temporal=True, time_column=time_col)

    frame = pl.concat([pl.read_parquet(tmp), pl.read_parquet(info.path)], how="horizontal_extend").to_pandas()
    blocks = [fc.to_series(frame[frame["__split"] == k], time_col, target, series) for k in (0, 1, 2)]
    train, valid, test = blocks
    history = {
        sid: (dates.append(valid[sid][0]), np.concatenate([values, valid[sid][1]]))
        for sid, (dates, values) in train.items()
        if sid in valid
    }
    y_true = np.concatenate([values for _, values in test.values()])
    out = {}
    for family in fc.FAMILIES:
        model = fc.fit(family, train, horizon=horizon, freq=freq, valid=valid)
        preds = fc.backtest(model, history, test)
        out[family] = fc.metrics(y_true, np.concatenate([preds[sid] for sid in test]))
    return out


def main() -> None:
    """Command-line entry point: print a Markdown table of the scores."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--time", required=True, help="time column")
    parser.add_argument("--series", default="", help="comma-separated series id columns (empty: one series)")
    parser.add_argument("--freq", default="D")
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()
    series = [c for c in args.series.split(",") if c]
    results = benchmark(args.data, args.target, args.time, series, args.freq, args.horizon)
    print(f"{args.data.name}  (horizon {args.horizon}, freq {args.freq})")
    print("| Model | sMAPE | MAE | RMSE |")
    print("|---|---|---|---|")
    for family, m in results.items():
        print(f"| {family} | {m['smape']:.2f} | {m['mae']:.2f} | {m['rmse']:.2f} |")


if __name__ == "__main__":
    main()
