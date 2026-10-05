"""Schemas for pre-training data audit and leakage detection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AuditSeverity = Literal["critical", "high", "medium", "low"]
AuditAction = Literal["drop", "flag", "impute", "stratify", "none"]


class AuditFinding(BaseModel):
    """One finding produced by the pre-training data audit."""

    check: str = Field(description="Name of the check, e.g. predictive_leakage, id_leakage, split_leakage")
    severity: AuditSeverity = Field(description="Severity of the finding")
    column: str | None = Field(default=None, description="Column name associated with the finding, if any")
    message: str = Field(description="Human-readable description of the issue")
    suggested_action: AuditAction = Field(default="flag", description="Action recommended by the audit")
    metric_name: str | None = Field(default=None, description="Diagnostic metric, e.g. auc, r2, null_share")
    metric_value: float | None = Field(default=None, description="Value of the diagnostic metric")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional context or diagnostics")


class AuditReport(BaseModel):
    """Full report of the pre-training data audit."""

    findings: list[AuditFinding] = Field(default_factory=list, description="List of individual findings")
    dropped_columns: list[str] = Field(
        default_factory=list, description="Columns automatically dropped due to high-severity leakage"
    )
    train_test_duplicates: int = Field(
        default=0, description="Exact or near duplicate rows shared between train and test"
    )
    duplicate_pct: float = Field(default=0.0, description="Percentage of test rows that duplicate train rows")
    has_leakage: bool = Field(
        default=False, description="True if any critical or high leakage finding was detected"
    )
    duration_s: float = Field(default=0.0, description="Time taken to audit the dataset in seconds")
