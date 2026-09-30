"""Structured task specification produced by the Prompt Agent (paper §3.1).

The paper parses the user's request into a JSON object describing the user,
problem, dataset, model and deployment constraints. We keep the parts that
drive our pipeline and validate them with Pydantic.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TaskType(StrEnum):
    """Supported machine-learning task types."""

    tabular_classification = "tabular_classification"
    tabular_regression = "tabular_regression"
    text_classification = "text_classification"


CLASSIFICATION_METRICS = {"accuracy", "f1_macro", "f1_weighted", "roc_auc", "balanced_accuracy"}
REGRESSION_METRICS = {"rmse", "mae", "r2", "mape"}
LOWER_IS_BETTER = {"rmse", "mae", "mape"}

DEFAULT_METRIC = {
    TaskType.tabular_classification: "accuracy",
    TaskType.tabular_regression: "rmse",
    TaskType.text_classification: "f1_macro",
}


def allowed_metrics(task_type: TaskType) -> set[str]:
    """Metrics valid for a task type."""
    return REGRESSION_METRICS if task_type == TaskType.tabular_regression else CLASSIFICATION_METRICS


class TaskSpec(BaseModel):
    """Machine-readable task produced by the Prompt Agent and checked by request verification."""

    task_type: TaskType
    target_column: str = Field(description="Column to predict")
    text_column: str | None = Field(default=None, description="Free-text input column (text tasks only)")
    feature_columns: list[str] | None = Field(
        default=None, description="Columns to use as features; null means all except target"
    )
    drop_columns: list[str] = Field(default_factory=list, description="Columns to exclude (ids, leakage)")
    metric: str = Field(description="Primary evaluation metric")
    metric_target: float | None = Field(default=None, description="Desired value of the metric, if stated")
    max_train_time_s: int | None = Field(default=None, description="Training time budget, if stated")
    domain: str | None = Field(default=None, description="Application domain, if mentioned")
    notes: str = Field(default="", description="Any other user constraints or preferences")
    user_expertise: Literal["beginner", "intermediate", "expert"] = Field(
        default="intermediate", description="User's apparent ML expertise, inferred from the request"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made where the request was ambiguous (shown to the user)",
    )

    @property
    def higher_is_better(self) -> bool:
        """Whether larger values of the metric are better."""
        return self.metric not in LOWER_IS_BETTER

    @model_validator(mode="after")
    def _normalize(self) -> TaskSpec:
        self.metric = self.metric.strip().lower()
        if self.metric not in allowed_metrics(self.task_type):
            self.metric = DEFAULT_METRIC[self.task_type]
        return self

    def meets_target(self, value: float) -> bool:
        """Whether `value` satisfies the user's metric target (always true without one)."""
        if self.metric_target is None:
            return True
        return value >= self.metric_target if self.higher_is_better else value <= self.metric_target
