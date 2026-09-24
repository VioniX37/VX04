"""Implementation verification (paper §3.5): did the executed code meet the user's requirements?"""

from __future__ import annotations

from automl_agent.schemas.plan import ExecutionResult
from automl_agent.schemas.task_spec import TaskSpec

from .request import VerificationResult


def verify_implementation(spec: TaskSpec, result: ExecutionResult) -> VerificationResult:
    if not result.ok or result.metrics is None:
        return VerificationResult(ok=False, issues=["code did not run successfully"])
    score = result.metrics.get("score")
    if not isinstance(score, int | float):
        return VerificationResult(ok=False, issues=[f"metrics.json has no numeric score for '{spec.metric}'"])

    issues = []
    if not spec.meets_target(score):
        direction = ">=" if spec.higher_is_better else "<="
        issues.append(f"{spec.metric}={score:.4f} does not meet the target {direction} {spec.metric_target}")
    train_time = result.metrics.get("train_time_s")
    if (
        spec.max_train_time_s is not None
        and isinstance(train_time, int | float)
        and train_time > spec.max_train_time_s
    ):
        issues.append(f"training took {train_time:.1f}s, over the {spec.max_train_time_s}s budget")
    return VerificationResult(ok=not issues, issues=issues)
