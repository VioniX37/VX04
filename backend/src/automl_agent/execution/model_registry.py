"""Model families supported by the code templates, per task type.

Plans must choose from these so the Operation Agent always has a working
starting point. Extending the system to a new model = add it here and in the
matching template's `build_model` function.
"""

from __future__ import annotations

from automl_agent.schemas.task_spec import TaskType

SUPPORTED_MODELS: dict[TaskType, dict[str, str]] = {
    TaskType.tabular_classification: {
        "hist_gradient_boosting": "Histogram gradient boosting (fast, handles large tabular data well)",
        "random_forest": "Random forest (robust baseline, little tuning)",
        "extra_trees": "Extremely randomized trees",
        "gradient_boosting": "Classic gradient boosting",
        "logistic_regression": "Regularized logistic regression (linear baseline)",
        "svm": "Support vector machine with RBF kernel (small/medium data)",
        "knn": "k-nearest neighbours",
    },
    TaskType.tabular_regression: {
        "hist_gradient_boosting": "Histogram gradient boosting regressor",
        "random_forest": "Random forest regressor",
        "extra_trees": "Extremely randomized trees regressor",
        "gradient_boosting": "Classic gradient boosting regressor",
        "ridge": "Ridge regression (linear baseline)",
        "lasso": "Lasso regression (sparse linear)",
        "svm": "Support vector regression",
        "knn": "k-nearest neighbours regressor",
    },
    TaskType.text_classification: {
        "logistic_regression": "TF-IDF + logistic regression",
        "linear_svm": "TF-IDF + linear SVM",
        "sgd": "TF-IDF + SGD classifier (scales to large corpora)",
        "naive_bayes": "TF-IDF + complement naive Bayes",
    },
}

TEMPLATE_FOR_TASK: dict[TaskType, str] = {
    TaskType.tabular_classification: "tabular.py",
    TaskType.tabular_regression: "tabular.py",
    TaskType.text_classification: "text_classification.py",
}


def supported_models(task_type: TaskType) -> dict[str, str]:
    return SUPPORTED_MODELS[task_type]


def normalize_model(task_type: TaskType, name: str) -> str:
    """Map a free-form model name to a supported family (falls back to the first/default one)."""
    options = SUPPORTED_MODELS[task_type]
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in options:
        return key
    for opt in options:
        if opt in key or key in opt:
            return opt
    return next(iter(options))
