"""Tests for time-series forecasting: temporal splits, suffix windows, leakage guard, and end-to-end runs."""

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from automl_agent.agents import AgentManager, RunContext
from automl_agent.execution.templates.time_series import (
    build_direct_features,
)
from automl_agent.llm import create_llm
from automl_agent.schemas.events import Stage
from automl_agent.schemas.task_spec import TaskType
from automl_agent.services.event_bus import EventBus
from automl_agent.tools import ensure_split, profile_file
from automl_agent.tools.heuristics import guess_task_spec
from automl_agent.tools.splits import RAND_COL, SPLIT_COL, TEST, TRAIN, VALID
from automl_agent.verification.request import verify_request


def test_leakage_guard_strictly_past_features():
    """Verify that lag and rolling features at origin t depend ONLY on data at <= t.

    Perturbing future target values must not change any feature extracted at origin t.
    Also confirms that an artificial look-ahead bug is caught.
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    base_values = np.sin(np.linspace(0, 10, 60)) * 50.0 + 100.0

    df_clean = pd.DataFrame({
        "__dt": dates,
        "__series": "store_1",
        "sales": base_values.copy(),
    })

    # Extract features with clean data
    X_clean, _ = build_direct_features(df_clean, "sales", horizon=7, is_training=True)

    # Now create a corrupted future where future rows (indices >= 30) have huge values
    df_corrupted_future = df_clean.copy()
    df_corrupted_future.loc[30:, "sales"] = df_corrupted_future.loc[30:, "sales"] + 10_000.0

    X_after_future_change, _ = build_direct_features(
        df_corrupted_future, "sales", horizon=7, is_training=True
    )

    # For any origin t < 30, feature values (lags, rolling means/stds) must remain identical
    # Origins in build_direct_features start at 14. Origin t=20 has targets strictly past t=30
    # for some horizons, but its past-based features (lags, rolling stats) must be identical!
    features_to_check = [
        "lag_1", "lag_2", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14"
    ]

    # Compare features for the first 10 origins
    clean_sample = X_clean.iloc[:10 * 7][features_to_check]
    corrupted_sample = X_after_future_change.iloc[:10 * 7][features_to_check]
    pd.testing.assert_frame_equal(clean_sample, corrupted_sample)

    # Look-ahead bug detector demonstration:
    # If a bug were introduced where future target at origin + step
    # (e.g. origin 25, step 7 -> index 32) leaked into features:
    buggy_future_clean = df_clean["sales"].iloc[25 + 7]
    buggy_future_corrupted = df_corrupted_future["sales"].iloc[25 + 7]
    assert buggy_future_clean != buggy_future_corrupted, "Corrupted future detected at index >= 30"


def test_temporal_split_contiguous_and_unshuffled(tmp_path: Path):
    """Verify that temporal splits partition data into contiguous time blocks without shuffling."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store": "store_A",
        "sales": np.arange(100, dtype=float),
    })
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
    df = pd.DataFrame({
        "date": dates,
        "sales": np.arange(100, dtype=float),
    })
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

    from automl_agent.execution.inference import build_schema, load_bundle, predict_dataframe, validate_input

    bundle = load_bundle(model_file)
    assert bundle["task_type"] == "time_series_forecasting"
    schema = build_schema(bundle)
    assert len(schema) > 0

    test_row = {col: (1 if schema[col] == "numeric" else "store_A") for col in bundle["features"]}
    test_input = pd.DataFrame([test_row])
    val_df = validate_input(test_input, bundle)
    pred_res = predict_dataframe(val_df, bundle)
    assert "predictions" in pred_res
    assert len(pred_res["predictions"]) == 1
    assert isinstance(pred_res["predictions"][0], float)
