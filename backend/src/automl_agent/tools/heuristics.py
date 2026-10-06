"""Rule-based fallbacks used by the offline Fake LLM and for input sanity checks."""

from __future__ import annotations

import re

from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import (
    DEFAULT_METRIC,
    TaskSpec,
    TaskType,
    allowed_metrics,
)

_TEXT_HINTS = ("text", "review", "sentiment", "message", "comment", "tweet", "email", "document", "nlp")
_FORECAST_HINTS = ("forecast", "forecasting", "predict next", "time series", "time-series")
_REGRESSION_HINTS = ("regress", "predict the price", "estimate", "how much", "continuous")
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
    "smape": "smape",
}


def _mentioned_columns(prompt: str, profile: DatasetProfile) -> list[str]:
    p = prompt.lower()
    return [c.name for c in profile.columns if re.search(rf"\b{re.escape(c.name.lower())}\b", p)]


def guess_task_spec(prompt: str, profile: DatasetProfile) -> TaskSpec:
    """Rule-based TaskSpec (used by the offline FakeLLM and as a sanity reference)."""
    p = prompt.lower()
    mentioned = _mentioned_columns(prompt, profile)

    is_forecast = any(h in p for h in _FORECAST_HINTS) or (
        hasattr(profile, "time_series")
        and profile.time_series is not None
        and ("forecast" in p or "predict" in p)
    )

    # Target:
    target = None
    # For forecasting: "forecast next 14 days of <target>" or "forecast <target>"
    m_fc = re.search(
        r"(?:forecast|predict)(?:\s+the)?(?:\s+next\s+\d+\s+\w+)?\s+(?:of\s+)?(?:the\s+)?([`'\"]?\w+)", p
    )
    if is_forecast and m_fc:
        cand = m_fc.group(1).strip("`'\"")
        target = next((c.name for c in profile.columns if c.name.lower() == cand), None)

    if target is None:
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

    # Time series specific fields
    time_column = None
    horizon = None
    frequency = None
    series_id_columns: list[str] = []

    if is_forecast:
        task_type = TaskType.time_series_forecasting
        # Extract horizon & frequency from "next 14 days" or "14 days"
        h_m = re.search(r"(?:next\s+)?(\d+)\s*(days?|hours?|weeks?|months?|minutes?|steps?|periods?)", p)
        if h_m:
            horizon = int(h_m.group(1))
            unit = h_m.group(2).lower()
            if "day" in unit:
                frequency = "D"
            elif "hour" in unit:
                frequency = "H"
            elif "week" in unit:
                frequency = "W"
            elif "month" in unit:
                frequency = "M"
            elif "min" in unit:
                frequency = "1min"

        if hasattr(profile, "time_series") and profile.time_series is not None:
            time_column = profile.time_series.time_column
            if not frequency and profile.time_series.frequency:
                frequency = profile.time_series.frequency
            if not horizon:
                horizon = 14 if frequency == "D" else (24 if frequency == "H" else 7)
            if profile.time_series.series_id_columns:
                series_id_columns = list(profile.time_series.series_id_columns)

        if not time_column:
            dt_cols = [c.name for c in profile.columns if c.kind == "datetime"]
            if dt_cols:
                time_column = dt_cols[0]

        # Check for series indicator in prompt: "per store", "by store", "for each store"
        s_m = re.search(r"(?:per|by|for each)\s+([`'\"]?\w+)", p)
        if s_m:
            s_cand = s_m.group(1).strip("`'\"")
            matched_col = next(
                (
                    c.name
                    for c in profile.columns
                    if c.name.lower() in (s_cand, f"{s_cand}_id", f"{s_cand}id")
                ),
                None,
            )
            if matched_col and matched_col not in series_id_columns:
                series_id_columns = [matched_col]

        if not horizon:
            horizon = 14
        if not frequency:
            frequency = "D"
    elif text_col and (any(h in p for h in _TEXT_HINTS) or len(profile.text_columns) > 0):
        task_type = TaskType.text_classification
    elif any(h in p for h in _REGRESSION_HINTS) or (
        target_col is not None and target_col.kind == "numeric" and target_col.n_unique > 20
    ):
        task_type = TaskType.tabular_regression
    else:
        task_type = TaskType.tabular_classification

    metric = DEFAULT_METRIC[task_type]
    allowed = allowed_metrics(task_type)
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

    drop = [
        c.name
        for c in profile.columns
        if c.kind == "identifier" and c.name != target and c.name not in series_id_columns
    ]
    return TaskSpec(
        task_type=task_type,
        target_column=target,
        time_column=time_column,
        horizon=horizon,
        frequency=frequency,
        series_id_columns=series_id_columns,
        text_column=text_col if task_type == TaskType.text_classification else None,
        drop_columns=drop,
        metric=metric,
        metric_target=metric_target,
        notes="parsed heuristically",
    )
