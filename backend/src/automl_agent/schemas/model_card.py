"""Schemas for post-training model card generation and diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from automl_agent.schemas.task_spec import TaskType


class FeatureImportance(BaseModel):
    """Importance score for a single feature."""

    feature: str
    importance: float
    method: Literal["native_gain", "permutation", "coefficient"] = "native_gain"


class CalibrationPoint(BaseModel):
    """One bin point on a reliability curve."""

    prob_pred: float
    prob_true: float


class CalibrationReport(BaseModel):
    """Probability calibration diagnostics for classification."""

    brier_score: float
    points: list[CalibrationPoint] = Field(default_factory=list)


class ConfusionMatrixData(BaseModel):
    """Confusion matrix and per-class classification metrics."""

    labels: list[str]
    matrix: list[list[int]]
    per_class: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Per-class precision, recall, f1, support"
    )


class ResidualsData(BaseModel):
    """Residual distribution and metrics for regression."""

    mae: float
    rmse: float
    r2: float
    max_error: float
    quantiles: dict[str, float] = Field(default_factory=dict)
    sample_residuals: list[float] = Field(default_factory=list)


class ModelCard(BaseModel):
    """Machine-readable model card artifact summarizing performance and explainability."""

    task_type: TaskType
    model_family: str
    primary_metric: str
    primary_score: float | None = None
    feature_importances: list[FeatureImportance] = Field(default_factory=list)
    confusion_matrix: ConfusionMatrixData | None = None
    residuals: ResidualsData | None = None
    calibration: CalibrationReport | None = None
    audit_summary: list[str] = Field(default_factory=list)
    why_this_model: str = Field(
        default="", description="Explanatory narrative citing only measured numbers and audit findings"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
