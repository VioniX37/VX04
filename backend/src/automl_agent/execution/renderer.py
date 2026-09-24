"""Render a code template into a runnable script for a given TaskSpec + plan."""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Any

from automl_agent.schemas.plan import Plan
from automl_agent.schemas.task_spec import TaskSpec

from .model_registry import TEMPLATE_FOR_TASK, normalize_model

TEMPLATES_DIR = Path(__file__).parent / "templates"
PLACEHOLDER = "__CONFIG__"


def build_config(
    spec: TaskSpec, plan: Plan, data_path: Path, hyperparameters: dict[str, Any] | None = None
) -> dict:
    return {
        "task_type": spec.task_type.value,
        "data_path": data_path.resolve().as_posix(),
        "target": spec.target_column,
        "text_column": spec.text_column,
        "feature_columns": spec.feature_columns,
        "drop_columns": spec.drop_columns,
        "metric": spec.metric,
        "model_family": normalize_model(spec.task_type, plan.model_family),
        "hyperparameters": hyperparameters if hyperparameters is not None else plan.hyperparameters,
        "test_size": 0.2,
        "random_state": 42,
    }


def render_template(spec: TaskSpec, plan: Plan, data_path: Path, hyperparameters: dict | None = None) -> str:
    source = (TEMPLATES_DIR / TEMPLATE_FOR_TASK[spec.task_type]).read_text(encoding="utf-8")
    config = build_config(spec, plan, data_path, hyperparameters)
    return source.replace(PLACEHOLDER, pprint.pformat(config, indent=4, sort_dicts=False), 1)
