"""Grounded execution verification via multi-fidelity successive halving.

The paper's execution verification ranks candidate plans on the Data/Model
agents' *pseudo-execution*: LLM-predicted scores that are never checked. Here
each surviving plan is actually trained on a growing, nested subsample of the
training split and scored on (a capped slice of) the validation split:

    rung 0: all P plans   on r0 rows
    rung 1: best ceil(P/eta) plans on r0*growth rows
    ...     until one plan remains or the full training set is reached

Because subsamples are nested (see ``tools.splits``), a plan's score at a
higher rung is always measured on a superset of its earlier data. The rungs
are sized to fit the remaining wall-clock budget: when the next rung is
predicted not to fit, grounding stops and ranks on what it has observed.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from automl_agent.schemas.plan import Observation, PlanEvaluation
from automl_agent.schemas.task_spec import TaskSpec

RunFn = Callable[[PlanEvaluation, int | None], Awaitable[Observation]]


@dataclass(frozen=True)
class GroundingSchedule:
    """Training rows per rung; None means the full training split."""

    rungs: list[int | None]
    eta: int

    def survivors_after(self, n: int) -> int:
        """How many plans advance from a rung that evaluated `n` plans."""
        return max(1, math.ceil(n / self.eta))


def fidelity_schedule(
    n_train: int, n_plans: int, *, min_rows: int, growth: int, eta: int
) -> GroundingSchedule:
    """Plan the rungs for `n_plans` plans on a training split of `n_train` rows.

    Small datasets (``n_train <= min_rows``) get a single full-data rung, which
    simply evaluates every plan for real.
    """
    if n_plans < 1:
        raise ValueError("n_plans must be >= 1")
    growth, eta = max(growth, 2), max(eta, 2)
    rungs: list[int | None] = []
    rows = min_rows
    survivors = n_plans
    while True:
        if rows >= n_train:
            rungs.append(None)
            break
        rungs.append(rows)
        survivors = max(1, math.ceil(survivors / eta))
        if survivors == 1:
            break
        rows *= growth
    return GroundingSchedule(rungs, eta)


def _sort_key(ev: PlanEvaluation, higher_is_better: bool) -> tuple:
    obs = ev.last_observation
    if obs is None or not obs.ok or obs.score is None:
        return (len(ev.observations), 0, -math.inf)
    score = obs.score if higher_is_better else -obs.score
    return (len(ev.observations), 1, score)


def rank_by_observations(evaluations: list[PlanEvaluation], spec: TaskSpec) -> list[PlanEvaluation]:
    """Best first: furthest rung reached, then a successful run, then the observed score."""
    ranked = sorted(evaluations, key=lambda ev: _sort_key(ev, spec.higher_is_better), reverse=True)
    for i, ev in enumerate(ranked, start=1):
        ev.rank = i
    return ranked


@dataclass
class RungReport:
    """What happened at one rung (for events and the UI)."""

    rows: int | None
    results: list[dict]
    stopped_for_budget: bool = False


async def successive_halving(
    evaluations: list[PlanEvaluation],
    spec: TaskSpec,
    schedule: GroundingSchedule,
    run: RunFn,
    *,
    n_train: int,
    wall_remaining: Callable[[], float | None] = lambda: None,
    on_rung: Callable[[RungReport], Awaitable[None]] | None = None,
) -> list[PlanEvaluation]:
    """Run the schedule and return all evaluations ranked best-first.

    Args:
        run: Trains one plan at a fidelity (rows, None = full) and returns the observation.
        n_train: Size of the full training split (to estimate the cost of a full-data rung).
        wall_remaining: Seconds left in the run budget (None = unlimited).
        on_rung: Called after each rung with its results.
    """
    survivors = list(evaluations)
    seconds_per_plan: float | None = None
    for i, rows in enumerate(schedule.rungs):
        remaining = wall_remaining()
        if i > 0 and seconds_per_plan is not None and remaining is not None:
            prev_rows = schedule.rungs[i - 1] or n_train
            growth = (rows or n_train) / max(prev_rows, 1)
            estimate = seconds_per_plan * growth * len(survivors)  # training time ~ linear in rows
            if estimate > remaining:
                if on_rung:
                    await on_rung(RungReport(rows, [], stopped_for_budget=True))
                break
        rung_seconds = 0.0
        for ev in survivors:
            obs = await run(ev, rows)
            ev.observations.append(obs)
            rung_seconds += obs.duration_s
        seconds_per_plan = rung_seconds / max(len(survivors), 1)
        if on_rung:
            await on_rung(
                RungReport(
                    rows,
                    [
                        {
                            "plan_id": ev.plan.id,
                            "model_family": ev.model.model_family,
                            **ev.last_observation.model_dump(mode="json"),
                        }  # type: ignore[union-attr]
                        for ev in survivors
                    ],
                )
            )
        ranked = rank_by_observations(survivors, spec)
        if i < len(schedule.rungs) - 1:
            survivors = ranked[: schedule.survivors_after(len(ranked))]
            if all(not (ev.last_observation and ev.last_observation.ok) for ev in survivors):
                break  # nothing works at this fidelity; more data won't help
    return rank_by_observations(evaluations, spec)
