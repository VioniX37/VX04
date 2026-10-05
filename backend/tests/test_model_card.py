"""Tests for post-training Model Card generation and explainability."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from automl_agent.extensions.model_card import (
    ModelCardHook,
    extract_feature_importances,
    format_model_card_markdown,
)
from automl_agent.schemas.model_card import (
    CalibrationPoint,
    CalibrationReport,
    ConfusionMatrixData,
    FeatureImportance,
    ModelCard,
)
from automl_agent.schemas.task_spec import TaskSpec, TaskType


def test_extract_feature_importances_sklearn_trees():
    X = np.random.randn(100, 3)
    y = (X[:, 0] * 2 + X[:, 1] > 0).astype(int)
    feature_names = ["alpha", "beta", "gamma"]

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    fi = extract_feature_importances(rf, feature_names, X, y)
    assert len(fi) == 3
    assert all(isinstance(f, FeatureImportance) for f in fi)
    assert sum(f.importance for f in fi) == pytest.approx(1.0, rel=1e-2)
    assert fi[0].feature in ("alpha", "beta")


def test_format_model_card_markdown():
    card = ModelCard(
        task_type=TaskType.tabular_classification,
        model_family="lightgbm",
        primary_metric="accuracy",
        primary_score=0.9450,
        feature_importances=[
            FeatureImportance(feature="monthly_charges", importance=0.65, method="native_gain"),
            FeatureImportance(feature="tenure", importance=0.35, method="native_gain"),
        ],
        confusion_matrix=ConfusionMatrixData(
            labels=["no", "yes"],
            matrix=[[80, 5], [6, 40]],
            per_class={
                "no": {"precision": 0.93, "recall": 0.94, "f1-score": 0.94, "support": 85},
                "yes": {"precision": 0.89, "recall": 0.87, "f1-score": 0.88, "support": 46},
            },
        ),
        calibration=CalibrationReport(
            brier_score=0.065,
            points=[
                CalibrationPoint(prob_pred=0.1, prob_true=0.08),
                CalibrationPoint(prob_pred=0.9, prob_true=0.92),
            ],
        ),
        audit_summary=["No target leakage detected", "Minor class imbalance handled via stratification"],
        why_this_model=(
            "Selected LightGBM with accuracy=0.9450 because it outperformed "
            "candidates on validation fidelity."
        ),
    )

    md = format_model_card_markdown(card)
    assert "# Model Card: lightgbm" in md
    assert "accuracy" in md
    assert "0.9450" in md
    assert "monthly_charges" in md
    assert "Brier Score" in md
    assert "Confusion Matrix" in md
    assert "Why This Model" in md


def test_model_card_hook_generates_artifacts(tmp_path: Path):
    artifact_dir = tmp_path / "attempt_1"
    artifact_dir.mkdir(parents=True)

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    X = np.random.randn(50, 2)
    y = X[:, 0] * 3.0 + 1.0
    rf.fit(X, y)

    joblib.dump(
        {"model": rf, "classes": None, "features": ["f1", "f2"], "task_type": "tabular_regression"},
        artifact_dir / "model.joblib",
    )

    spec = TaskSpec(
        task_type=TaskType.tabular_regression,
        target_column="price",
        metric="rmse",
    )

    class DummyResult:
        success = True
        task_spec = spec
        plan = type("Plan", (), {"model_family": "random_forest"})()
        metrics = {"score": 0.1234, "artifact_dir": str(artifact_dir)}

    class DummyContext:
        dataset_path = Path("nonexistent.parquet")
        split = None
        workdir = tmp_path
        state = {"audit_report": None}

        async def emit(self, *args, **kwargs):
            pass

    hook = ModelCardHook()
    res = DummyResult()
    ctx = DummyContext()
    asyncio.run(hook.on_run_finished(ctx, res))

    assert (artifact_dir / "model_card.json").exists()
    assert (artifact_dir / "model_card.md").exists()
    assert (tmp_path / "model_card.json").exists()

    card_data = json.loads((artifact_dir / "model_card.json").read_text(encoding="utf-8"))
    assert card_data["model_family"] == "random_forest"
    assert card_data["primary_metric"] == "rmse"
    assert len(card_data["feature_importances"]) == 2
    assert "model_card" in res.metrics


@pytest.mark.parametrize(
    ("sample", "prompt", "section"),
    [
        ("customer_churn.csv", "Predict churn", "confusion_matrix"),
        ("house_prices.csv", "Predict the house price", "residuals"),
        ("product_reviews.csv", "Classify review sentiment", "confusion_matrix"),
    ],
)
def test_model_card_has_test_diagnostics_for_every_task_type(settings, sample, prompt, section):
    """Real runs get test-split diagnostics and non-zero importances, not just an empty card."""
    from automl_agent.agents import AgentManager, RunContext
    from automl_agent.llm import create_llm
    from automl_agent.services.event_bus import EventBus
    from automl_agent.tools import profile_file

    path = Path(__file__).resolve().parents[2] / "data" / "samples" / sample
    ctx = RunContext(
        run_id="card",
        prompt=prompt,
        dataset_path=path,
        profile=profile_file(path),
        workdir=settings.runs_dir / "card",
        settings=settings,
        llm=create_llm(settings),
        bus=EventBus(),
    )
    result = asyncio.run(AgentManager(ctx).run())
    assert result.success, result.error
    card = result.metrics["model_card"]
    assert card[section], f"model card has no {section}"
    assert sum(f["importance"] for f in card["feature_importances"]) > 0
    if sample == "customer_churn.csv":
        assert card["calibration"] is not None
