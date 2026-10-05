"""Tests for pre-training data audit and leakage detection."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from automl_agent.schemas.dataset import ColumnProfile, DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.tools.data_audit import (
    apply_audit_to_task_spec,
    check_train_test_duplicates,
    run_data_audit,
)
from automl_agent.tools.splits import RAND_COL, SPLIT_COL, TEST, ensure_split


@pytest.fixture
def leaky_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary dataset with planted target leakage, IDs, and train/test duplicates."""
    rng = np.random.default_rng(42)
    n = 1000

    tenure = rng.integers(1, 60, n)
    charges = rng.normal(50, 15, n).round(2)
    churn = rng.choice(["yes", "no"], n, p=[0.3, 0.7])

    # Leaked column 1: near-perfect predictor (post-outcome cancellation code)
    cancellation = np.where(churn == "yes", 1, 0)
    # Leaked column 2: ID-like column
    guid = [f"usr_{i}_{rng.integers(1000, 9999)}" for i in range(n)]

    df = pl.DataFrame(
        {
            "account_guid": guid,
            "tenure": tenure,
            "monthly_charges": charges,
            "cancellation_code": cancellation,
            "churn": churn,
        }
    )

    # Duplicate 40 rows
    dups = df.head(40)
    df = pl.concat([df, dups])

    data_path = tmp_path / "leaky_data.parquet"
    df.write_parquet(data_path)

    split = ensure_split(data_path, "churn", stratify=True, seed=42)
    return data_path, split.path


def test_audit_flags_planted_leakage(leaky_dataset):
    data_path, split_path = leaky_dataset

    spec = TaskSpec(
        task_type=TaskType.tabular_classification,
        target_column="churn",
        metric="accuracy",
    )
    profile = DatasetProfile(
        n_rows=1040,
        n_cols=5,
        columns=[
            ColumnProfile(
                name="account_guid",
                dtype="String",
                kind="identifier",
                n_unique=1000,
                n_missing=0,
                sample_values=[],
            ),
            ColumnProfile(
                name="tenure", dtype="Int64", kind="numeric", n_unique=60, n_missing=0, sample_values=[]
            ),
            ColumnProfile(
                name="monthly_charges",
                dtype="Float64",
                kind="numeric",
                n_unique=800,
                n_missing=0,
                sample_values=[],
            ),
            ColumnProfile(
                name="cancellation_code",
                dtype="Int64",
                kind="numeric",
                n_unique=2,
                n_missing=0,
                sample_values=[],
            ),
            ColumnProfile(
                name="churn", dtype="String", kind="categorical", n_unique=2, n_missing=0, sample_values=[]
            ),
        ],
        guessed_target="churn",
        text_columns=[],
        size_bytes=10000,
        memory_estimate_mb=1.0,
        scale_tier="small",
        approximate_counts=False,
    )

    report = run_data_audit(data_path, spec, profile, split_path)

    assert report.has_leakage
    assert "cancellation_code" in report.dropped_columns
    assert "account_guid" in report.dropped_columns

    # Verify findings structure
    pred_leak = next(f for f in report.findings if f.column == "cancellation_code")
    assert pred_leak.check == "predictive_leakage"
    assert pred_leak.severity == "critical"
    assert pred_leak.suggested_action == "drop"

    id_leak = next(f for f in report.findings if f.column == "account_guid")
    assert id_leak.check == "id_leakage"
    assert id_leak.suggested_action == "drop"

    # Test TaskSpec update
    updated_spec = apply_audit_to_task_spec(report, spec)
    assert "cancellation_code" in updated_spec.drop_columns
    assert "account_guid" in updated_spec.drop_columns
    assert any("cancellation_code" in a for a in updated_spec.assumptions)


def test_duplicate_detection_5m_rows_within_seconds(tmp_path: Path):
    """Verify that train/test duplicate detection scales to 5M rows in Polars within seconds."""
    n_rows = 5_000_000
    rng = np.random.default_rng(123)

    data_path = tmp_path / "large_synth.parquet"
    split_path = tmp_path / "splits" / "target-42.parquet"
    split_path.parent.mkdir(parents=True, exist_ok=True)

    # 5 million rows with 2 feature columns
    col1 = rng.integers(0, 100_000, n_rows, dtype=np.int32)
    col2 = rng.integers(0, 50_000, n_rows, dtype=np.int32)

    pl.DataFrame({"feat1": col1, "feat2": col2}).write_parquet(
        data_path, compression="zstd", row_group_size=500_000
    )

    # Assign 70% train, 15% valid, 15% test
    splits = np.zeros(n_rows, dtype=np.uint8)  # 0 = train
    n_test = 750_000
    splits[n_rows - n_test :] = TEST  # 2 = test

    pl.DataFrame(
        {
            SPLIT_COL: splits,
            RAND_COL: rng.random(n_rows, dtype=np.float32),
        }
    ).write_parquet(split_path, compression="zstd", row_group_size=500_000)

    # Benchmark duplicate detection
    t0 = time.perf_counter()
    n_leaked, pct = check_train_test_duplicates(data_path, split_path, ["feat1", "feat2"])
    duration = time.perf_counter() - t0

    assert isinstance(n_leaked, int)
    assert isinstance(pct, float)
    # Must complete within 10 seconds (typically ~1-2 seconds on modern CPU)
    assert duration < 10.0, f"Duplicate detection on 5M rows took {duration:.2f}s, expected < 10s"


def test_audit_quality_checks():
    spec = TaskSpec(
        task_type=TaskType.tabular_classification,
        target_column="label",
        metric="accuracy",
    )
    profile = DatasetProfile(
        n_rows=1000,
        n_cols=4,
        columns=[
            ColumnProfile(
                name="constant_feat",
                dtype="Int64",
                kind="numeric",
                n_unique=1,
                n_missing=0,
                sample_values=["1"],
            ),
            ColumnProfile(
                name="mostly_null",
                dtype="Float64",
                kind="numeric",
                n_unique=10,
                n_missing=850,
                sample_values=[],
            ),
            ColumnProfile(
                name="near_constant",
                dtype="String",
                kind="categorical",
                n_unique=2,
                n_missing=0,
                sample_values=[],
                top_values={"A": 0.998, "B": 0.002},
            ),
            ColumnProfile(
                name="label",
                dtype="String",
                kind="categorical",
                n_unique=2,
                n_missing=0,
                sample_values=[],
                top_values={"neg": 0.995, "pos": 0.005},
            ),
        ],
        guessed_target="label",
        text_columns=[],
        size_bytes=5000,
        memory_estimate_mb=0.5,
        scale_tier="small",
        approximate_counts=False,
    )

    report = run_data_audit(Path("dummy.parquet"), spec, profile)

    checks = {f.check: f for f in report.findings}
    assert "constant_column" in checks
    assert "missing_values" in checks
    assert "near_constant_column" in checks
    assert "class_imbalance" in checks
    assert "constant_feat" in report.dropped_columns
    assert "mostly_null" in report.dropped_columns


def test_pipeline_with_planted_leaky_column(settings):
    import asyncio

    from automl_agent.agents import AgentManager, RunContext
    from automl_agent.llm import create_llm
    from automl_agent.services.event_bus import EventBus
    from automl_agent.tools import profile_file

    leaky_csv = Path(__file__).resolve().parents[2] / "data" / "samples" / "customer_churn_leaky.csv"
    assert leaky_csv.exists()

    bus = EventBus()
    ctx = RunContext(
        run_id="audit_test",
        prompt="Predict customer churn. Metric: accuracy",
        dataset_path=leaky_csv,
        profile=profile_file(leaky_csv),
        workdir=settings.runs_dir / "audit_test",
        settings=settings,
        llm=create_llm(settings),
        bus=bus,
    )
    result = asyncio.run(AgentManager(ctx).run())

    assert result.success, result.error
    assert "cancellation_confirmation" in result.task_spec.drop_columns
    assert "account_guid" in result.task_spec.drop_columns

    assert "model_card" in result.metrics
    model_card = result.metrics["model_card"]
    feature_names = [fi["feature"] for fi in model_card["feature_importances"]]
    assert "cancellation_confirmation" not in feature_names
    assert "account_guid" not in feature_names
    assert (ctx.workdir / "attempt_1" / "model_card.json").exists()
    assert (ctx.workdir / "attempt_1" / "model_card.md").exists()


def _audit_frame(tmp_path: Path, df: pl.DataFrame, spec: TaskSpec):
    from automl_agent.tools import profile_file

    path = tmp_path / "audit.parquet"
    df.write_parquet(path)
    return run_data_audit(path, spec, profile_file(path))


def test_continuous_features_are_not_mistaken_for_identifiers(tmp_path: Path):
    """A float measurement is unique per row but carries signal; it must not be dropped as an ID."""
    rng = np.random.default_rng(0)
    n = 2000
    income = rng.normal(50_000, 15_000, n)
    df = pl.DataFrame(
        {"income": income, "label": np.where(income + rng.normal(0, 20_000, n) > 50_000, "a", "b")}
    )
    spec = TaskSpec(task_type=TaskType.tabular_classification, target_column="label", metric="accuracy")
    report = _audit_frame(tmp_path, df, spec)
    assert "income" not in report.dropped_columns


def test_text_column_is_never_dropped(tmp_path: Path):
    """The input of a text task is near-unique by nature and must survive the audit."""
    n = 500
    df = pl.DataFrame(
        {
            "review_id_text": [
                f"review number {i} says the product was {'good' if i % 2 else 'bad'}" for i in range(n)
            ],
            "sentiment": ["pos" if i % 2 else "neg" for i in range(n)],
        }
    )
    spec = TaskSpec(
        task_type=TaskType.text_classification,
        target_column="sentiment",
        text_column="review_id_text",
        metric="f1_macro",
    )
    report = _audit_frame(tmp_path, df, spec)
    assert "review_id_text" not in report.dropped_columns


def test_many_category_leak_is_detected_even_when_sorted_by_target(tmp_path: Path):
    """A 40-level category that maps to the target is a leak, also when the file is sorted by target."""
    rng = np.random.default_rng(1)
    n = 4000
    status = rng.integers(0, 40, n)
    label = np.where(status < 20, "churned", "stayed")
    df = pl.DataFrame(
        {"status_code": [f"S{s:02d}" for s in status], "noise": rng.normal(size=n), "label": label}
    )
    df = df.sort("label")
    spec = TaskSpec(task_type=TaskType.tabular_classification, target_column="label", metric="accuracy")
    report = _audit_frame(tmp_path, df, spec)
    assert "status_code" in report.dropped_columns
    assert "noise" not in report.dropped_columns
