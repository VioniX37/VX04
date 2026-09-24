"""Request verification (paper §3.1): is the parsed TaskSpec valid and actionable for this dataset?"""

from __future__ import annotations

from pydantic import BaseModel

from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec, TaskType


class VerificationResult(BaseModel):
    ok: bool
    issues: list[str] = []

    @property
    def feedback(self) -> str:
        return "; ".join(self.issues)


def verify_request(spec: TaskSpec, profile: DatasetProfile) -> VerificationResult:
    issues: list[str] = []
    names = {c.name for c in profile.columns}

    target = profile.column(spec.target_column)
    if target is None:
        issues.append(f"target_column '{spec.target_column}' is not a column of the dataset")
    else:
        if target.n_unique < 2:
            issues.append(f"target '{target.name}' has fewer than 2 distinct values")
        if spec.task_type == TaskType.tabular_regression and target.kind != "numeric":
            issues.append(f"regression target '{target.name}' is not numeric")
        if spec.task_type != TaskType.tabular_regression and target.n_unique > max(50, profile.n_rows // 2):
            issues.append(
                f"classification target '{target.name}' has {target.n_unique} classes - "
                "is this a regression task?"
            )

    if spec.task_type == TaskType.text_classification:
        if not spec.text_column:
            issues.append("text_classification requires text_column")
        elif spec.text_column not in names:
            issues.append(f"text_column '{spec.text_column}' is not a column of the dataset")

    for col in [*(spec.feature_columns or []), *spec.drop_columns]:
        if col not in names:
            issues.append(f"column '{col}' does not exist")
    if spec.target_column in (spec.feature_columns or []):
        issues.append("target_column must not be a feature")

    if profile.n_rows < 20:
        issues.append(f"dataset has only {profile.n_rows} rows; at least 20 are needed")

    return VerificationResult(ok=not issues, issues=issues)
