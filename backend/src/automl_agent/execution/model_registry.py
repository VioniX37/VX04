"""Model families the code templates implement, with the data sizes they scale to.

Plans must pick from these families so the Operation Agent always starts from
working code. ``max_rows`` removes families that would be too slow or too
memory-hungry on the full training set (e.g. RBF SVMs beyond ~20k rows).
To add a family: register it here and implement it in the matching template's
``build_model``.
"""

from __future__ import annotations

from dataclasses import dataclass

from automl_agent.schemas.task_spec import TaskType


@dataclass(frozen=True)
class ModelFamily:
    """A model family available to plans."""

    description: str
    max_rows: int | None = None  # None = scales to any size we support


_TABULAR_SHARED = {
    "lightgbm": ModelFamily(
        "LightGBM gradient boosting: fastest and strongest default for large tabular data"
    ),
    "xgboost": ModelFamily("XGBoost (hist) gradient boosting with native categoricals"),
    "hist_gradient_boosting": ModelFamily("scikit-learn histogram gradient boosting"),
    "random_forest": ModelFamily("Random forest: robust, little tuning", max_rows=1_000_000),
    "extra_trees": ModelFamily("Extremely randomized trees", max_rows=1_000_000),
    "gradient_boosting": ModelFamily("Classic (non-histogram) gradient boosting", max_rows=200_000),
    "sgd": ModelFamily("Linear model trained with SGD: scales to any size, fast baseline"),
    "svm": ModelFamily("Kernel SVM (RBF): small data only", max_rows=20_000),
    "knn": ModelFamily("k-nearest neighbours: small data only", max_rows=100_000),
}

SUPPORTED_MODELS: dict[TaskType, dict[str, ModelFamily]] = {
    TaskType.tabular_classification: {
        **_TABULAR_SHARED,
        "logistic_regression": ModelFamily("Regularized logistic regression", max_rows=2_000_000),
    },
    TaskType.tabular_regression: {
        **_TABULAR_SHARED,
        "ridge": ModelFamily("Ridge regression (linear baseline)"),
        "lasso": ModelFamily("Lasso regression (sparse linear)", max_rows=2_000_000),
    },
    TaskType.text_classification: {
        "logistic_regression": ModelFamily("TF-IDF + logistic regression", max_rows=1_000_000),
        "linear_svm": ModelFamily("TF-IDF + linear SVM", max_rows=1_000_000),
        "naive_bayes": ModelFamily("TF-IDF + complement naive Bayes", max_rows=1_000_000),
        "sgd": ModelFamily("TF-IDF + SGD classifier", max_rows=1_000_000),
        "sgd_hashing": ModelFamily("Streaming hashing vectorizer + SGD (out-of-core, any size)"),
    },
    TaskType.time_series_forecasting: {
        "lightgbm": ModelFamily("LightGBM on lag, rolling-window and calendar features (global model)"),
        "seasonal_naive": ModelFamily("Seasonal-naive baseline (repeats observed seasonal cycle)"),
        "ets": ModelFamily("Exponential smoothing baseline (ETS with level, trend and seasonality)"),
    },
}

TEMPLATE_FOR_TASK: dict[TaskType, str] = {
    TaskType.tabular_classification: "tabular.py",
    TaskType.tabular_regression: "tabular.py",
    TaskType.text_classification: "text_classification.py",
    TaskType.time_series_forecasting: "time_series.py",
}


def supported_models(task_type: TaskType, n_rows: int | None = None) -> dict[str, str]:
    """Return ``{family: description}`` usable for a training set of `n_rows` rows."""
    return {
        key: fam.description
        for key, fam in SUPPORTED_MODELS[task_type].items()
        if n_rows is None or fam.max_rows is None or n_rows <= fam.max_rows
    }


def normalize_model(task_type: TaskType, name: str, n_rows: int | None = None) -> str:
    """Map a free-form model name to a supported family (falls back to the first allowed one)."""
    options = supported_models(task_type, n_rows)
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in options:
        return key
    for opt in options:
        if opt in key or key in opt:
            return opt
    return next(iter(options))
