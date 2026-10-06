"""Tests for time-series forecasting: temporal splits, suffix windows, leakage guard, and end-to-end runs."""

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from automl_agent.agents import AgentManager, RunContext
from automl_agent.execution import forecasting as fc
from automl_agent.llm import create_llm
from automl_agent.schemas.events import Stage
from automl_agent.schemas.task_spec import TaskType
from automl_agent.services.event_bus import EventBus
from automl_agent.tools import ensure_split, profile_file
from automl_agent.tools.heuristics import guess_task_spec
from automl_agent.tools.splits import RAND_COL, SPLIT_COL, TEST, TRAIN, VALID
from automl_agent.verification.request import verify_request


def test_leakage_guard_strictly_past_features():
    """Features at origin t depend only on values at or before t.

    Changing every value from index 30 on must leave the features of all origins before 30
    unchanged, while their targets (which may lie after 30) do change.
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.sin(np.linspace(0, 10, 60)) * 50.0 + 100.0
    corrupted = values.copy()
    corrupted[30:] += 10_000.0
    horizon = 7

    def examples(v):
        return fc.training_examples({"s": (dates, v)}, {"s": 0}, horizon, season=7)

    X_clean, y_clean = examples(values)
    X_bad, y_bad = examples(corrupted)
    origins = np.concatenate([np.arange(fc.MIN_HISTORY - 1, 60 - h) for h in range(1, horizon + 1)])
    past = origins < 30
    feature_cols = [c for c in fc.FEATURES if c not in ("dayofweek", "month", "day", "is_weekend")]
    pd.testing.assert_frame_equal(
        X_clean.loc[past, feature_cols].reset_index(drop=True),
        X_bad.loc[past, feature_cols].reset_index(drop=True),
    )
    # The guard is meaningful: some of those origins forecast targets after index 30.
    assert not np.allclose(y_clean[past], y_bad[past])


def test_temporal_split_contiguous_and_unshuffled(tmp_path: Path):
    """Verify that temporal splits partition data into contiguous time blocks without shuffling."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": "store_A",
            "sales": np.arange(100, dtype=float),
        }
    )
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path)

    split_info = ensure_split(
        data_path,
        target="sales",
        stratify=False,
        valid_fraction=0.15,
        test_fraction=0.15,
        temporal=True,
        time_column="date",
    )

    split_df = pl.read_parquet(split_info.path)
    splits = split_df[SPLIT_COL].to_numpy()

    # Train, valid, test must be contiguous non-empty blocks
    train_indices = np.where(splits == TRAIN)[0]
    valid_indices = np.where(splits == VALID)[0]
    test_indices = np.where(splits == TEST)[0]

    assert len(train_indices) == 70
    assert len(valid_indices) == 15
    assert len(test_indices) == 15

    # Check contiguous and chronologically ordered
    assert train_indices.max() < valid_indices.min()
    assert valid_indices.max() < test_indices.min()
    assert np.all(np.diff(train_indices) == 1)
    assert np.all(np.diff(valid_indices) == 1)
    assert np.all(np.diff(test_indices) == 1)


def test_suffix_window_grounding_rungs(tmp_path: Path):
    """Verify that subsampling on __r selects suffix windows (the most recent observations)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "sales": np.arange(100, dtype=float),
        }
    )
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path)

    split_info = ensure_split(
        data_path,
        target="sales",
        stratify=False,
        valid_fraction=0.15,
        test_fraction=0.15,
        temporal=True,
        time_column="date",
    )
    combined = pl.read_parquet(data_path).hstack(pl.read_parquet(split_info.path))
    train_part = combined.filter(pl.col(SPLIT_COL) == TRAIN)
    n_train = len(train_part)
    assert n_train == 70

    # Low-fidelity rung: subsample to 20 rows
    r20 = train_part.filter(pl.col(RAND_COL) < 20 / n_train)
    assert len(r20) == 20

    # Higher-fidelity rung: subsample to 40 rows
    r40 = train_part.filter(pl.col(RAND_COL) < 40 / n_train)
    assert len(r40) == 40

    # The 20 selected rows must be the most recent 20 rows of the 70 training rows (indices 50..69)
    assert r20["sales"].to_list() == list(range(50, 70))
    # The 40 selected rows must be indices 30..69
    assert r40["sales"].to_list() == list(range(30, 70))

    # Strict nesting: r20 rows must be a subset of r40 rows
    r20_set = set(r20["sales"].to_list())
    r40_set = set(r40["sales"].to_list())
    assert r20_set.issubset(r40_set)


def test_temporal_split_shares_date_boundaries_across_series(tmp_path: Path):
    """In panel data every series is cut at the same dates; no date falls into two blocks."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "date": np.tile(dates.strftime("%Y-%m-%d"), 3),
            "store": np.repeat(["a", "b", "c"], 50),
            "sales": np.arange(150, dtype=float),
        }
    )
    data_path = tmp_path / "panel.parquet"
    df.to_parquet(data_path)
    info = ensure_split(data_path, "sales", stratify=False, temporal=True, time_column="date")
    joined = pl.read_parquet(data_path).hstack(pl.read_parquet(info.path))
    per_date = joined.group_by("date").agg(pl.col(SPLIT_COL).n_unique().alias("blocks"))
    assert per_date["blocks"].max() == 1
    bounds = joined.group_by(SPLIT_COL).agg(
        pl.col("date").min().alias("lo"), pl.col("date").max().alias("hi")
    )
    bounds = bounds.sort(SPLIT_COL)
    assert bounds["hi"][0] < bounds["lo"][1] and bounds["hi"][1] < bounds["lo"][2]
    assert joined.group_by("store").agg(pl.col(SPLIT_COL).n_unique().alias("n"))["n"].min() == 3


def test_profiler_time_series_detection(sample_csvs):
    """Verify that the profiler detects datetime, infers frequency, gaps, and series info."""
    prof = profile_file(sample_csvs["stores"])

    # Datetime detection
    dt_col = prof.column("date")
    assert dt_col is not None
    assert dt_col.kind == "datetime"

    # Time series profile attached
    assert prof.time_series is not None
    ts = prof.time_series
    assert ts.time_column == "date"
    assert ts.frequency in ("D", "day", None)
    assert ts.n_series == 3
    assert "store_1" in ts.series_lengths or len(ts.series_lengths) > 0
    assert ts.series_id_columns == ["store_id"]


def test_prompt_agent_heuristic_extraction(sample_csvs):
    """Verify extraction of horizon, frequency, series IDs from natural language."""
    prof = profile_file(sample_csvs["stores"])
    prompt = "Forecast next 14 days of sales per store. Optimise sMAPE."
    spec = guess_task_spec(prompt, prof)

    assert spec.task_type == TaskType.time_series_forecasting
    assert spec.target_column == "sales"
    assert spec.horizon == 14
    assert spec.frequency == "D"
    assert spec.metric == "smape"
    assert "store_id" in spec.series_id_columns
    assert spec.time_column == "date"

    # Request verification check
    verdict = verify_request(spec, prof)
    assert verdict.ok, verdict.issues


def test_time_series_pipeline_end_to_end(settings, sample_csvs):
    """Verify full end-to-end AutoML pipeline run on store_sales dataset."""
    bus = EventBus()
    csv = sample_csvs["stores"]
    prompt = "Forecast next 14 days of store sales per store. Optimise sMAPE."

    ctx = RunContext(
        run_id="test-ts-run",
        prompt=prompt,
        dataset_path=csv,
        profile=profile_file(csv),
        workdir=settings.runs_dir / "test-ts-run",
        settings=settings,
        llm=create_llm(settings),
        bus=bus,
    )

    result = asyncio.run(AgentManager(ctx).run())

    assert result.success, result.error
    assert result.task_spec.task_type == TaskType.time_series_forecasting
    assert result.task_spec.metric == "smape"
    assert isinstance(result.metrics["score"], float)
    assert "baseline_scores" in result.metrics
    assert "seasonal_naive" in result.metrics["baseline_scores"]
    assert "beats_seasonal_naive" in result.metrics

    metrics_file = ctx.workdir / "attempt_1" / "metrics.json"
    assert metrics_file.exists()

    events = bus.history("test-ts-run")
    stages = {e.stage for e in events}
    expected_stages = (
        Stage.parse,
        Stage.verify_request,
        Stage.prepare,
        Stage.ground,
        Stage.select,
        Stage.implement,
        Stage.verify_impl,
    )
    for expected in expected_stages:
        assert expected in stages

    model_file = ctx.workdir / "attempt_1" / "model.joblib"
    assert model_file.exists()

    from automl_agent.execution.inference import (
        InferenceError,
        build_schema,
        load_bundle,
        predict_dataframe,
        validate_input,
    )

    bundle = load_bundle(model_file)
    assert bundle["task_type"] == "time_series_forecasting"
    schema = build_schema(bundle)
    assert len(schema) > 0

    # Forecasts continue each series from its own recent history (level and weekly pattern),
    # not from empty features.
    stores = bundle["categories"]["store_id"]
    data = pd.read_csv(csv)
    request = pd.DataFrame(
        {"store_id": np.repeat(stores, 14), "horizon_step": np.tile(np.arange(1, 15), len(stores))}
    )
    pred_res = predict_dataframe(validate_input(request, bundle), bundle)
    preds = np.asarray(pred_res["predictions"]).reshape(len(stores), 14)
    for i, store in enumerate(stores):
        recent = data[data["store_id"] == store]["sales"].tail(28).mean()
        assert abs(preds[i].mean() - recent) < 0.3 * recent, (store, preds[i].mean(), recent)
        assert preds[i].std() > 0  # not a flat line
    assert pred_res["forecast_dates"][0] > str(data["date"].max())

    with pytest.raises(InferenceError) as unknown:
        predict_dataframe(
            validate_input(pd.DataFrame({"store_id": ["nope"], "horizon_step": [1]}), bundle), bundle
        )
    assert unknown.value.column == "store_id"
    with pytest.raises(InferenceError):
        predict_dataframe(
            validate_input(pd.DataFrame({"store_id": [stores[0]], "horizon_step": [99]}), bundle), bundle
        )


def test_forecasting_bundle_loads_without_the_package(settings, sample_csvs, tmp_path: Path):
    """The exported serving files forecast with automl_agent unavailable (as in a deployment bundle)."""
    import shutil
    import subprocess
    import sys

    from automl_agent.execution import forecasting, inference

    csv = sample_csvs["stores"]
    ctx = RunContext(
        run_id="ts-bundle",
        prompt="Forecast next 14 days of sales per store",
        dataset_path=csv,
        profile=profile_file(csv),
        workdir=settings.runs_dir / "ts-bundle",
        settings=settings,
        llm=create_llm(settings),
        bus=EventBus(),
    )
    result = asyncio.run(AgentManager(ctx).run())
    assert result.success, result.error
    shutil.copy(Path(result.metrics["artifact_dir"]) / "model.joblib", tmp_path / "model.joblib")
    shutil.copy(inference.__file__, tmp_path / "automl_inference.py")
    shutil.copy(forecasting.__file__, tmp_path / "automl_forecasting.py")
    script = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'automl_agent': raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import pandas as pd, automl_inference as inf\n"
        "b = inf.load_bundle(__import__('pathlib').Path('model.joblib'))\n"
        "sid = b['categories']['store_id'][0]\n"
        "req = pd.DataFrame({'store_id': [sid], 'horizon_step': [1]})\n"
        "out = inf.predict_dataframe(inf.validate_input(req, b), b)\n"
        "print(out['predictions'][0])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert float(proc.stdout.strip()) > 0
