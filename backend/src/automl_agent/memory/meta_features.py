"""Dataset meta-features for similarity search, and a stable dataset fingerprint.

Meta-features are cheap statistics taken from the profile (never raw data):
size, shape, column-type mix, missingness and target balance. Two datasets are
"similar" when these vectors are close and the task type matches.
"""

from __future__ import annotations

import hashlib
import math

from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec, TaskType

FEATURES = (
    "log_rows", "log_cols", "frac_numeric", "frac_categorical", "frac_text", "frac_missing",
    "log_classes", "log_imbalance",
)  # fmt: skip
# Relative importance of each feature in the distance.
WEIGHTS = {
    "log_rows": 0.6, "log_cols": 0.6, "frac_numeric": 1.0, "frac_categorical": 1.0, "frac_text": 1.5,
    "frac_missing": 0.8, "log_classes": 0.8, "log_imbalance": 0.8,
}  # fmt: skip


def meta_features(profile: DatasetProfile, spec: TaskSpec) -> dict[str, float]:
    """Compute the meta-feature vector of a (dataset, task) pair."""
    features = [
        c for c in profile.columns if c.name != spec.target_column and c.name not in spec.drop_columns
    ]
    n_feat = max(len(features), 1)
    cells = max(profile.n_rows * max(profile.n_cols, 1), 1)
    target = profile.column(spec.target_column)
    n_classes, imbalance = 0.0, 1.0
    if spec.task_type != TaskType.tabular_regression and target is not None:
        n_classes = float(target.n_unique)
        shares = list((target.top_values or {}).values())
        if len(shares) >= 2 and min(shares) > 0:
            imbalance = max(shares) / min(shares)
    return {
        "log_rows": math.log10(max(profile.n_rows, 1)),
        "log_cols": math.log10(max(profile.n_cols, 1)),
        "frac_numeric": sum(c.kind == "numeric" for c in features) / n_feat,
        "frac_categorical": sum(c.kind in ("categorical", "boolean") for c in features) / n_feat,
        "frac_text": sum(c.kind == "text" for c in features) / n_feat,
        "frac_missing": sum(c.n_missing for c in profile.columns) / cells,
        "log_classes": math.log2(n_classes) if n_classes > 1 else 0.0,
        "log_imbalance": math.log10(imbalance),
    }


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Weighted Euclidean distance between two meta-feature vectors."""
    return math.sqrt(sum(WEIGHTS[k] * (a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in FEATURES))


def dataset_fingerprint(profile: DatasetProfile) -> str:
    """Identify a dataset by its schema and size (stable across re-registrations of the same file)."""
    signature = "|".join(f"{c.name}:{c.dtype}" for c in profile.columns) + f"#{profile.n_rows}"
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
