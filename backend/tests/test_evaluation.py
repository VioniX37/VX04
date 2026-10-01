"""Evaluation metrics and a smoke run of the benchmark harness (offline)."""

import csv
import json

import pytest

from evaluation import analysis
from evaluation.metrics import calibration, comprehensive_score, normalized_performance, success_rate
from evaluation.run_benchmark import build_parser, main_async


def test_success_rate_grading():
    assert success_rate(False, False) == 0
    assert success_rate(True, False) == 0.5
    assert success_rate(True, True) == 1


def test_nps():
    assert normalized_performance("accuracy", 0.8) == 0.8
    assert normalized_performance("rmse", 1.0) == 0.5
    assert normalized_performance("rmse", 30_000.0, {"rmsle": 0.25}) == pytest.approx(0.8)
    assert normalized_performance("r2", -3.0) == 0.0
    assert normalized_performance("accuracy", None) == 0.0
    assert comprehensive_score(1.0, 0.8) == pytest.approx(0.9)


def test_calibration_on_first_rung():
    obs = [
        {"run_id": "r", "revision": 1, "plan_id": p, "predicted_score": pred, "observed_score": obs,
         "fidelity_rows": 100, "metric": "accuracy", "higher_is_better": True, "final": False}
        for p, pred, obs in [("a", 0.9, 0.6), ("b", 0.8, 0.7), ("c", 0.7, 0.8)]
    ] + [  # a later rung must be ignored
        {"run_id": "r", "revision": 1, "plan_id": "c", "predicted_score": 0.7, "observed_score": 0.85,
         "fidelity_rows": 400, "metric": "accuracy", "higher_is_better": True, "final": False},
    ]  # fmt: skip
    c = calibration(obs)
    assert c["groups"] == 1
    assert c["spearman"] == pytest.approx(-1.0)
    assert c["top1_hit"] == 0.0
    assert c["mae"] == pytest.approx((0.3 + 0.1 + 0.1) / 3)


def test_benchmark_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    from automl_agent.config import get_settings

    get_settings.cache_clear()
    try:
        args = build_parser().parse_args(
            ["--out", str(tmp_path / "out"), "--variants", "paper,grounded", "--tasks", "churn,reviews",
             "--set", "MAX_REVISIONS=0", "--isolated-cache"]
        )  # fmt: skip
        import asyncio

        out = asyncio.run(main_async(args))
    finally:
        get_settings.cache_clear()

    with (out / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    variants = {r["variant"] for r in rows}
    assert {"paper", "grounded", "baseline:zero_shot", "baseline:optuna_lgbm"} <= variants
    assert all(r["success"] == "True" for r in rows if r["kind"] == "pipeline")
    assert not any(r["variant"] == "baseline:optuna_lgbm" and r["task"] == "reviews" for r in rows)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["grounded"]["calibration"]["groups"] >= 2
    assert summary["paper"]["calibration"]["groups"] == 0  # nothing is observed in pseudo mode

    assert analysis.main([str(out)]) == 0
    assert (out / "figures" / "fig_cs_by_variant.png").exists()
    assert (out / "figures" / "fig_calibration.png").exists()
