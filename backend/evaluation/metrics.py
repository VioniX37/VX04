"""Evaluation metrics: the paper's SR / NPS / CS, plus calibration and cost metrics.

Success rate (SR) follows the paper's graded scheme, adapted to our pipeline
(which stops at a trained, verified model rather than a deployed endpoint):

    SR = 0    the pipeline produced no working model
    SR = 0.5  a working model was produced but a user constraint was not met on the test split
    SR = 1    a working model was produced and every stated constraint was met
              (for constraint-free requests, any working model scores 1)

Normalized performance score (NPS), as in the paper: bounded "higher is better"
metrics are used directly; loss metrics s map to 1 / (1 + s). For regression the
paper uses RMSLE, so we prefer it whenever the run reports it.

Comprehensive score: CS = 0.5 * SR + 0.5 * NPS.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from scipy.stats import spearmanr

LOSS_METRICS = {"rmse", "mae", "mape", "rmsle"}


def success_rate(success: bool, target_met: bool) -> float:
    """Graded success rate of one run (see module docstring)."""
    if not success:
        return 0.0
    return 1.0 if target_met else 0.5


def normalized_performance(
    metric: str | None, score: float | None, metrics: dict[str, Any] | None = None
) -> float:
    """NPS in [0, 1]; 0 when the run produced no score."""
    metrics = metrics or {}
    if metric in {"rmse", "mae", "mape"} and isinstance(metrics.get("rmsle"), int | float):
        metric, score = "rmsle", metrics["rmsle"]
    if score is None or metric is None or (isinstance(score, float) and math.isnan(score)):
        return 0.0
    if metric in LOSS_METRICS:
        return 1.0 / (1.0 + max(float(score), 0.0))
    return min(max(float(score), 0.0), 1.0)


def comprehensive_score(sr: float, nps: float) -> float:
    """CS = 0.5 * SR + 0.5 * NPS."""
    return 0.5 * sr + 0.5 * nps


def calibration(observations: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    """How well LLM-predicted scores match observed ones (RQ1).

    Uses grounding rows at each group's lowest fidelity, where every candidate plan
    of a planning round was run on identical data. Groups are (run, revision).

    Returns:
        ``spearman`` (mean rank correlation over groups with >= 3 plans), ``top1_hit``
        (share of groups where the predicted best plan is the observed best), ``mae``
        (mean |predicted - observed| over bounded metrics) and the group count.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in observations:
        if row.get("final") or row.get("observed_score") is None or row.get("predicted_score") is None:
            continue
        groups.setdefault((row.get("run_id"), row.get("revision")), []).append(row)

    rhos, hits, errors = [], [], []
    for rows in groups.values():
        lowest = min(r.get("fidelity_rows") or 0 for r in rows)
        rung = [r for r in rows if (r.get("fidelity_rows") or 0) == lowest]
        higher = rung[0].get("higher_is_better", True)
        sign = 1 if higher in (True, 1, "True", "true") else -1
        pred = [sign * float(r["predicted_score"]) for r in rung]
        obs = [sign * float(r["observed_score"]) for r in rung]
        if len(rung) >= 3 and len(set(pred)) > 1 and len(set(obs)) > 1:
            rhos.append(float(spearmanr(pred, obs).statistic))
        if len(rung) >= 2:
            hits.append(
                float(
                    max(range(len(rung)), key=pred.__getitem__) == max(range(len(rung)), key=obs.__getitem__)
                )
            )
        if rung[0].get("metric") not in LOSS_METRICS:
            errors.extend(abs(p - o) for p, o in zip(pred, obs, strict=True))

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {"spearman": mean(rhos), "top1_hit": mean(hits), "mae": mean(errors), "groups": len(groups)}
