"""Render a code template into a runnable script for a given TaskSpec + plan."""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Any, Literal

from automl_agent.schemas.plan import Plan
from automl_agent.schemas.task_spec import TaskSpec

from .model_registry import TEMPLATE_FOR_TASK, normalize_model

TEMPLATES_DIR = Path(__file__).parent / "templates"
PLACEHOLDER = "__CONFIG__"


def build_config(
    spec: TaskSpec,
    plan: Plan,
    data_path: Path,
    split_path: Path,
    *,
    hyperparameters: dict[str, Any] | None = None,
    fidelity_rows: int | None = None,
    eval_split: Literal["valid", "test"] = "test",
    eval_max_rows: int | None = None,
    save_model: bool = True,
    n_jobs: int = -1,
    n_rows: int | None = None,
) -> dict[str, Any]:
    """Build the CONFIG dict injected into a template.

    Args:
        fidelity_rows: Subsample the train split to about this many rows (None = all).
        eval_split: Split to report the score on; the test split is reserved for final runs.
        eval_max_rows: Cap on validation rows (keeps low-fidelity runs cheap).
        n_rows: Full training-set size, used to keep the model family within its scale limits.
    """
    return {
        "task_type": spec.task_type.value,
        "data_path": data_path.resolve().as_posix(),
        "split_path": split_path.resolve().as_posix(),
        "target": spec.target_column,
        "text_column": spec.text_column,
        "feature_columns": spec.feature_columns,
        "drop_columns": spec.drop_columns,
        "metric": spec.metric,
        "model_family": normalize_model(spec.task_type, plan.model_family, n_rows),
        "hyperparameters": hyperparameters if hyperparameters is not None else plan.hyperparameters,
        "fidelity_rows": fidelity_rows,
        "eval_split": eval_split,
        "eval_max_rows": eval_max_rows,
        "save_model": save_model,
        "n_jobs": n_jobs,
        "random_state": 42,
    }


def render_template(spec: TaskSpec, plan: Plan, data_path: Path, split_path: Path, **kwargs: Any) -> str:
    """Return the template for `spec.task_type` with its CONFIG filled in (see `build_config`)."""
    source = (TEMPLATES_DIR / TEMPLATE_FOR_TASK[spec.task_type]).read_text(encoding="utf-8")
    config = build_config(spec, plan, data_path, split_path, **kwargs)
    return source.replace(PLACEHOLDER, pprint.pformat(config, indent=4, sort_dicts=False), 1)
