"""Rule-based fallbacks used by the offline Fake LLM and for input sanity checks."""

from __future__ import annotations

import re

from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import (
    CLASSIFICATION_METRICS,
    DEFAULT_METRIC,
    REGRESSION_METRICS,
    TaskSpec,
    TaskType,
)

_TEXT_HINTS = ("text", "review", "sentiment", "message", "comment", "tweet", "email", "document", "nlp")
_REGRESSION_HINTS = ("regress", "predict the price", "forecast", "estimate", "how much", "continuous")
_METRIC_ALIASES = {
    "accuracy": "accuracy",
    "f1": "f1_macro",
    "f1-score": "f1_macro",
    "f1 score": "f1_macro",
    "auc": "roc_auc",
    "roc": "roc_auc",
    "rmse": "rmse",
    "mae": "mae",
    "r2": "r2",
    "r^2": "r2",
    "mape": "mape",
}


def _mentioned_columns(prompt: str, profile: DatasetProfile) -> list[str]:
    p = prompt.lower()
    return [c.name for c in profile.columns if re.search(rf"\b{re.escape(c.name.lower())}\b", p)]


def guess_task_spec(prompt: str, profile: DatasetProfile) -> TaskSpec:
    """Rule-based TaskSpec (used by the offline FakeLLM and as a sanity reference)."""
    p = prompt.lower()
    mentioned = _mentioned_columns(prompt, profile)

    # Target: a column named right after "predict"/"target"/"classify", else a mentioned
    # non-text column, else the profiler's guess.
    target = None
    m = re.search(
        r"(?:predict|target(?: column)?(?: is)?|classify|label(?:ed)? by)\s+(?:the\s+)?[`'\"]?(\w+)", p
    )
    if m:
        target = next((c.name for c in profile.columns if c.name.lower() == m.group(1)), None)
    if target is None:
        target = next((c for c in mentioned if c not in profile.text_columns), None)
    if target is None:
        target = profile.guessed_target or profile.columns[-1].name

    text_col = next((c for c in profile.text_columns if c != target), None)
    target_col = profile.column(target)

    if text_col and (any(h in p for h in _TEXT_HINTS) or len(profile.text_columns) > 0):
        task_type = TaskType.text_classification
    elif any(h in p for h in _REGRESSION_HINTS) or (
        target_col is not None and target_col.kind == "numeric" and target_col.n_unique > 20
    ):
        task_type = TaskType.tabular_regression
    else:
        task_type = TaskType.tabular_classification

    metric = DEFAULT_METRIC[task_type]
    allowed = REGRESSION_METRICS if task_type == TaskType.tabular_regression else CLASSIFICATION_METRICS
    for alias, name in _METRIC_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", p) and name in allowed:
            metric = name
            break

    metric_target = None
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", p)
    if pct:
        metric_target = float(pct.group(1)) / 100
    else:
        num = re.search(rf"{metric.split('_')[0]}\D{{0,20}}(0\.\d+)", p)
        if num:
            metric_target = float(num.group(1))

    drop = [c.name for c in profile.columns if c.kind == "identifier" and c.name != target]
    return TaskSpec(
        task_type=task_type,
        target_column=target,
        text_column=text_col if task_type == TaskType.text_classification else None,
        drop_columns=drop,
        metric=metric,
        metric_target=metric_target,
        notes="parsed heuristically",
    )
