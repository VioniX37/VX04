"""Baselines compared against the agent pipeline.

* ``zero_shot``   - one Gemini call writes the whole training script from the request, the
                    dataset profile and the data contract; no planning, verification or repair
                    (the paper's "GPT-4 zero-shot" baseline, with our backbone).
* ``optuna_lgbm`` - TPE search over LightGBM hyperparameters under the same wall-clock budget
                    (tabular tasks only); a strong non-LLM AutoML baseline.

Both use the same Parquet data, the same train/valid/test split and the same
``metrics.json`` contract as the pipeline, so scores are directly comparable.
"""

from __future__ import annotations

import pprint
import time
from pathlib import Path
from typing import Any

from automl_agent.agents.base import render_context
from automl_agent.config import Settings
from automl_agent.execution.model_registry import supported_models
from automl_agent.execution.renderer import render_template
from automl_agent.execution.sandbox import run_script
from automl_agent.llm import LLMError, create_llm
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.plan import CodeDraft, Plan
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.tools.splits import SplitInfo

HERE = Path(__file__).resolve().parent

ZERO_SHOT_SYSTEM = """You are an expert machine-learning engineer.
Write ONE complete, runnable Python script that solves the task below. Contract:
- Read the data with polars: `pl.scan_parquet(CONFIG["data_path"])`, horizontally concatenated with
  `pl.scan_parquet(CONFIG["split_path"])`, whose column `__split` is 0=train, 1=valid, 2=test.
- Train on train (you may use valid for early stopping/tuning). Never train on test.
- Evaluate on the test split and write `metrics.json` with keys `metric` (= CONFIG["metric"]),
  `score` (value of that metric on test) and `metrics` (dict of all metrics you computed).
- Allowed libraries: python stdlib, numpy, pandas, polars, scikit-learn, lightgbm, xgboost, joblib.
- Start the script with the CONFIG dict given in the context, copied verbatim."""


def _data_config(spec: TaskSpec, data_path: Path, split: SplitInfo) -> dict[str, Any]:
    return {
        "task_type": spec.task_type.value,
        "data_path": data_path.resolve().as_posix(),
        "split_path": split.path.resolve().as_posix(),
        "target": spec.target_column,
        "text_column": spec.text_column,
        "drop_columns": spec.drop_columns,
        "metric": spec.metric,
    }


async def run_zero_shot(
    settings: Settings,
    *,
    spec: TaskSpec,
    prompt: str,
    profile: DatasetProfile,
    data_path: Path,
    split: SplitInfo,
    workdir: Path,
) -> dict[str, Any]:
    """Single-call baseline. `spec` is the ground-truth task (the baseline gets no parsing help)."""
    llm = create_llm(settings)
    config = _data_config(spec, data_path, split)
    fallback_family = next(iter(supported_models(spec.task_type, split.n_train)))
    # base_code is only used by the offline FakeLLM so that smoke tests exercise the path.
    base_code = render_template(
        spec, Plan(id="zs", title="zero-shot", rationale="", preprocessing=[], model_family=fallback_family),
        data_path, split.path, n_rows=split.n_train,
    )  # fmt: skip
    context = {"expected": "CodeDraft", "user_request": prompt, "dataset_profile": profile.compact(),
               "CONFIG": config, "base_code": base_code}  # fmt: skip
    start = time.perf_counter()
    try:
        draft = await llm.for_role("smart").complete_json(
            [{"role": "system", "content": ZERO_SHOT_SYSTEM},
             {"role": "user", "content": f"Write the script.\n\n{render_context(context)}"}],
            CodeDraft,
        )  # fmt: skip
        result = await run_script(draft.code, workdir, timeout_s=settings.exec_timeout_s)
    except LLMError as e:
        return {"success": False, "error": str(e), "llm_calls": llm.usage.calls, "wall_s": 0.0}
    metrics = result.metrics or {}
    score = metrics.get("score") if result.ok else None
    return {
        "success": isinstance(score, int | float),
        "target_met": isinstance(score, int | float) and spec.meets_target(score),
        "score": score,
        "metrics": metrics.get("metrics"),
        "model_family": "zero-shot",
        "wall_s": round(time.perf_counter() - start, 2),
        "llm_calls": llm.usage.calls,
        "cache_hits": llm.usage.cache_hits,
        "tokens": llm.usage.total_tokens,
        "error": None if result.ok else result.stderr[-500:],
    }


async def run_optuna_lgbm(
    settings: Settings,
    *,
    spec: TaskSpec,
    data_path: Path,
    split: SplitInfo,
    workdir: Path,
    time_budget_s: float,
) -> dict[str, Any] | None:
    """Optuna + LightGBM under a wall-clock budget; None for non-tabular tasks."""
    if spec.task_type == TaskType.text_classification:
        return None
    config = {
        **_data_config(spec, data_path, split),
        "time_budget_s": time_budget_s,
        "n_jobs": settings.exec_n_jobs,
    }
    source = (HERE / "optuna_lgbm_script.py").read_text(encoding="utf-8")
    code = source.replace("__CONFIG__", pprint.pformat(config, indent=4, sort_dicts=False), 1)
    result = await run_script(code, workdir, timeout_s=int(time_budget_s * 2 + 120))
    metrics = result.metrics or {}
    score = metrics.get("score") if result.ok else None
    return {
        "success": isinstance(score, int | float),
        "target_met": isinstance(score, int | float) and spec.meets_target(score),
        "score": score,
        "metrics": metrics.get("metrics"),
        "model_family": "lightgbm (optuna)",
        "wall_s": result.duration_s,
        "llm_calls": 0,
        "cache_hits": 0,
        "tokens": 0,
        "error": None if result.ok else result.stderr[-500:],
    }
