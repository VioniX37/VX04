"""Per-run budgets for wall-clock time, LLM calls and tokens.

Budgets make comparisons between pipeline variants fair ("equal cost") and
keep a free-tier key from being drained by a single run. They are soft
limits: the Manager checks them between stages, stops revising when a budget
is spent, and sizes grounding rungs to fit the remaining time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from automl_agent.config import Settings
from automl_agent.llm.base import LLMUsage


@dataclass(frozen=True)
class RunBudget:
    """Limits for one run; None means unlimited."""

    wall_s: float | None = None
    llm_calls: int | None = None
    tokens: int | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> RunBudget:
        """Read the ``BUDGET_*`` settings (0 = unlimited)."""
        return cls(
            wall_s=settings.budget_wall_s or None,
            llm_calls=settings.budget_llm_calls or None,
            tokens=settings.budget_tokens or None,
        )


class BudgetTracker:
    """Tracks consumption of a :class:`RunBudget` against the run's LLM usage and the clock."""

    def __init__(self, budget: RunBudget, usage: LLMUsage) -> None:
        self.budget = budget
        self.usage = usage
        self._start = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        """Seconds since the run started."""
        return time.monotonic() - self._start

    def wall_remaining(self) -> float | None:
        """Seconds left, or None when unlimited."""
        if self.budget.wall_s is None:
            return None
        return max(0.0, self.budget.wall_s - self.elapsed_s)

    def exhausted(self) -> str | None:
        """Name of the first exhausted budget, or None."""
        b = self.budget
        if b.wall_s is not None and self.elapsed_s >= b.wall_s:
            return "time"
        if b.llm_calls is not None and self.usage.calls >= b.llm_calls:
            return "llm_calls"
        if b.tokens is not None and self.usage.total_tokens >= b.tokens:
            return "tokens"
        return None

    def snapshot(self) -> dict[str, Any]:
        """Budget and consumption, for prompts, events and results."""
        remaining = self.wall_remaining()
        return {
            "elapsed_s": round(self.elapsed_s, 1),
            "wall_budget_s": self.budget.wall_s,
            "wall_remaining_s": None if remaining is None else round(remaining, 1),
            "llm_calls": self.usage.calls,
            "llm_call_budget": self.budget.llm_calls,
            "tokens": self.usage.total_tokens,
            "token_budget": self.budget.tokens,
        }
