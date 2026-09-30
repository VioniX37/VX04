"""Grounded verification: schedule, successive halving, budgets, and end-to-end modes."""

import asyncio

import pytest

from automl_agent.agents.budget import BudgetTracker, RunBudget
from automl_agent.llm.base import LLMUsage
from automl_agent.schemas.plan import DataAgentResult, ModelAgentResult, Observation, Plan, PlanEvaluation
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.verification import fidelity_schedule, rank_by_observations, successive_halving


def _ev(pid: str, predicted: float = 0.5) -> PlanEvaluation:
    return PlanEvaluation(
        plan=Plan(id=pid, title=pid, rationale="", preprocessing=[], model_family="lightgbm"),
        data=DataAgentResult(summary="", steps=[]),
        model=ModelAgentResult(summary="", model_family="lightgbm", predicted_score=predicted),
    )


CLF = TaskSpec(task_type=TaskType.tabular_classification, target_column="y", metric="accuracy")
REG = TaskSpec(task_type=TaskType.tabular_regression, target_column="y", metric="rmse")


@pytest.mark.parametrize(
    ("n_train", "n_plans", "expected"),
    [
        (1_000, 3, [None]),  # small data: one full-data rung
        (3_500_000, 3, [20_000, 80_000]),  # 3 -> 2 -> 1
        (3_500_000, 5, [20_000, 80_000, 320_000]),  # 5 -> 3 -> 2 -> 1
        (50_000, 5, [20_000, None]),  # growth reaches the full split
        (3_500_000, 1, [20_000]),
    ],
)
def test_fidelity_schedule(n_train, n_plans, expected):
    assert fidelity_schedule(n_train, n_plans, min_rows=20_000, growth=4, eta=2).rungs == expected


def _runner(true_scores, durations=None, fail=()):
    calls = []

    async def run(ev, rows):
        calls.append((ev.plan.id, rows))
        if ev.plan.id in fail:
            return Observation(fidelity_rows=rows or 0, score=None, ok=False, duration_s=1, error="boom")
        return Observation(
            fidelity_rows=rows or 0,
            score=true_scores[ev.plan.id],
            ok=True,
            duration_s=(durations or {}).get(ev.plan.id, 1.0),
        )

    return run, calls


def test_halving_follows_observed_not_predicted():
    evs = [_ev("a", predicted=0.95), _ev("b", predicted=0.60), _ev("c", predicted=0.70)]
    run, calls = _runner({"a": 0.70, "b": 0.90, "c": 0.80})
    schedule = fidelity_schedule(1_000_000, 3, min_rows=20_000, growth=4, eta=2)
    ranked = asyncio.run(successive_halving(evs, CLF, schedule, run, n_train=1_000_000))
    assert [ev.plan.id for ev in ranked] == ["b", "c", "a"]
    assert calls == [("a", 20_000), ("b", 20_000), ("c", 20_000), ("b", 80_000), ("c", 80_000)]
    assert len(ranked[0].observations) == 2 and len(ranked[2].observations) == 1


def test_halving_lower_is_better_and_failures_rank_last():
    evs = [_ev("a"), _ev("b"), _ev("c")]
    run, _ = _runner({"a": 10.0, "b": 5.0, "c": 1.0}, fail={"c"})
    schedule = fidelity_schedule(1_000, 3, min_rows=20_000, growth=4, eta=2)
    ranked = asyncio.run(successive_halving(evs, REG, schedule, run, n_train=1_000))
    assert [ev.plan.id for ev in ranked] == ["b", "a", "c"]


def test_halving_stops_when_next_rung_exceeds_budget():
    evs = [_ev("a"), _ev("b"), _ev("c")]
    run, calls = _runner({"a": 0.7, "b": 0.8, "c": 0.6}, durations={"a": 10, "b": 10, "c": 10})
    schedule = fidelity_schedule(1_000_000, 3, min_rows=20_000, growth=4, eta=2)
    reports = []

    async def on_rung(r):
        reports.append(r)

    ranked = asyncio.run(
        successive_halving(
            evs, CLF, schedule, run, n_train=1_000_000, wall_remaining=lambda: 30.0, on_rung=on_rung
        )
    )
    assert len(calls) == 3  # next rung needs ~2 plans x 40s > 30s
    assert reports[-1].stopped_for_budget
    assert ranked[0].plan.id == "b"


def test_rank_by_observations_prefers_further_rungs():
    a, b = _ev("a"), _ev("b")
    a.observations = [Observation(fidelity_rows=10, score=0.99, ok=True, duration_s=1)]
    b.observations = [
        Observation(fidelity_rows=10, score=0.8, ok=True, duration_s=1),
        Observation(fidelity_rows=40, score=0.85, ok=True, duration_s=1),
    ]
    assert [ev.plan.id for ev in rank_by_observations([a, b], CLF)] == ["b", "a"]


def test_budget_tracker():
    usage = LLMUsage(calls=5, input_tokens=900, output_tokens=200)
    assert BudgetTracker(RunBudget(llm_calls=5), usage).exhausted() == "llm_calls"
    assert BudgetTracker(RunBudget(tokens=1000), usage).exhausted() == "tokens"
    assert BudgetTracker(RunBudget(wall_s=0), usage).exhausted() == "time"
    free = BudgetTracker(RunBudget(), usage)
    assert free.exhausted() is None and free.wall_remaining() is None
