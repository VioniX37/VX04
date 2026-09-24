"""Execution verification (paper §3.4): rank the pseudo-executed plans and pick one to implement."""

from __future__ import annotations

from automl_agent.schemas.plan import PlanEvaluation
from automl_agent.schemas.task_spec import TaskSpec


def rank_plans(evaluations: list[PlanEvaluation], spec: TaskSpec) -> list[PlanEvaluation]:
    """Best first: predicted metric (direction-aware), then time budget, then faster training."""

    def key(ev: PlanEvaluation) -> tuple:
        score = ev.model.predicted_score if spec.higher_is_better else -ev.model.predicted_score
        over_budget = (
            spec.max_train_time_s is not None and ev.model.predicted_train_time_s > spec.max_train_time_s
        )
        return (not over_budget, score, -ev.model.predicted_train_time_s)

    ranked = sorted(evaluations, key=key, reverse=True)
    for i, ev in enumerate(ranked, start=1):
        ev.rank = i
    return ranked
