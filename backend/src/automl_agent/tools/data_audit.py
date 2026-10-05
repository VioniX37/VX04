"""Scalable pre-training data audit and target leakage detection.

Performs deterministic, fast checks using Polars' lazy engine:
1. Target leakage checks:
   - Single-feature predictive power against target using fast univariate models
   - Near-unique identifier columns
   - Column names suspiciously matching or derived from the target
   - Train/test exact duplicate row detection across splits
2. Data quality checks:
   - Severe class imbalance
   - Constant and near-constant columns
   - High-cardinality categorical features
   - Excessive missing value share
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl_agent.schemas.audit import AuditFinding, AuditReport
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.tools.dataset_profiler import scan_table
from automl_agent.tools.splits import SPLIT_COL, TEST, TRAIN

log = logging.getLogger(__name__)

_ID_PATTERNS = re.compile(r"(^|_)(id|uuid|guid|key|index|hash|token|code|pk)($|_)", re.IGNORECASE)
PREDICTIVE_SAMPLE_ROWS = 10_000
UNIVARIATE_LEAKAGE_THRESHOLD = 0.98


def _is_suspicious_target_name(col_name: str, target_name: str) -> bool:
    """Check if a feature column name suspiciously mirrors the target."""
    c = col_name.strip().lower()
    t = target_name.strip().lower()
    if c == t:
        return True
    # Strip common punctuation/underscores
    c_clean = re.sub(r"[^a-z0-9]", "", c)
    t_clean = re.sub(r"[^a-z0-9]", "", t)
    if not t_clean or len(t_clean) < 3:
        return False
    if t_clean in c_clean or c_clean in t_clean:
        # Check if it looks like target_pred, target_outcome, is_target, true_target, etc.
        suspicious_affixes = (
            "pred",
            "target",
            "true",
            "actual",
            "outcome",
            "label",
            "result",
            "post",
            "copy",
            "leak",
            "after",
        )
        for affix in suspicious_affixes:
            if affix in c:
                return True
        if len(c_clean) - len(t_clean) <= 4:
            return True
    return False


def _evaluate_univariate_power(
    df: pl.DataFrame,
    feature: str,
    target: str,
    task_type: TaskType,
) -> tuple[float, str]:
    """Train a fast depth-2 decision tree on a feature and evaluate performance against target."""
    sub = df.select([feature, target]).drop_nulls()
    if len(sub) < 50:
        return 0.0, "insufficient_data"

    # Shuffle so the 80/20 holdout is not an artefact of the file's row order (e.g. sorted by target).
    sub = sub.sample(fraction=1.0, shuffle=True, seed=42)
    target_s = sub[target]
    feat_s = sub[feature]

    if feat_s.dtype in (pl.String, pl.Categorical, pl.Enum):
        return _evaluate_categorical_power(feat_s.cast(pl.String), target_s, task_type)
    if feat_s.dtype == pl.Boolean:
        X = feat_s.to_numpy().astype(np.float32).reshape(-1, 1)
    elif feat_s.dtype.is_numeric():
        X = feat_s.to_numpy().astype(np.float32).reshape(-1, 1)
        # Handle inf / nan if any slipped through
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        return 0.0, "unsupported_dtype"

    if task_type == TaskType.tabular_regression:
        y = target_s.to_numpy().astype(np.float64)
        if np.std(y) < 1e-9:
            return 0.0, "zero_variance_target"
        # 80/20 train/valid split
        split_idx = int(len(X) * 0.8)
        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]
        if len(y_val) < 10 or np.std(y_val) < 1e-9:
            return 0.0, "insufficient_eval"

        tree = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree.fit(X_tr, y_tr)
        preds = tree.predict(X_val)
        score = float(r2_score(y_val, preds))
        return max(0.0, score), "r2"
    else:
        # Classification
        classes, y = np.unique(target_s.cast(pl.String).to_numpy(), return_inverse=True)
        if len(classes) < 2:
            return 0.0, "single_class"

        split_idx = int(len(X) * 0.8)
        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]
        if len(np.unique(y_val)) < 2:
            return 0.0, "single_class_eval"

        tree = DecisionTreeClassifier(max_depth=2, random_state=42)
        tree.fit(X_tr, y_tr)
        if len(classes) == 2:
            try:
                probs = tree.predict_proba(X_val)[:, 1]
                score = float(roc_auc_score(y_val, probs))
            except Exception:
                preds = tree.predict(X_val)
                score = float(accuracy_score(y_val, preds))
                return score, "accuracy"
            return score, "auc"
        else:
            preds = tree.predict(X_val)
            score = float(accuracy_score(y_val, preds))
            return score, "accuracy"


def _evaluate_categorical_power(
    feat_s: pl.Series, target_s: pl.Series, task_type: TaskType
) -> tuple[float, str]:
    """Score a categorical feature by predicting each holdout row from its category's training rows.

    A per-category lookup captures any mapping from category to target, which an
    ordinal encoding with a depth-2 tree cannot when there are more than a few categories.
    """
    split_idx = int(len(feat_s) * 0.8)
    x_tr, x_val = feat_s[:split_idx].to_list(), feat_s[split_idx:].to_list()
    if task_type == TaskType.tabular_regression:
        y = target_s.cast(pl.Float64).to_numpy()
        y_tr, y_val = y[:split_idx], y[split_idx:]
        if len(y_val) < 10 or np.std(y_val) < 1e-9:
            return 0.0, "insufficient_eval"
        means = pl.DataFrame({"x": x_tr, "y": y_tr}).group_by("x").agg(pl.col("y").mean())
        lookup = dict(zip(means["x"].to_list(), means["y"].to_list(), strict=True))
        prior = float(np.mean(y_tr))
        preds = np.array([lookup.get(v, prior) for v in x_val], dtype=np.float64)
        return max(0.0, float(r2_score(y_val, preds))), "r2"

    classes, y = np.unique(target_s.cast(pl.String).to_numpy(), return_inverse=True)
    if len(classes) < 2:
        return 0.0, "single_class"
    y_tr, y_val = y[:split_idx], y[split_idx:]
    if len(np.unique(y_val)) < 2:
        return 0.0, "single_class_eval"
    counts = np.zeros((0, len(classes)))
    index: dict[object, int] = {}
    rows = []
    for v, label in zip(x_tr, y_tr, strict=True):
        if v not in index:
            index[v] = len(rows)
            rows.append(np.zeros(len(classes)))
        rows[index[v]][label] += 1
    counts = np.vstack(rows) if rows else counts
    prior = np.bincount(y_tr, minlength=len(classes)) / max(len(y_tr), 1)
    proba = np.array(
        [counts[index[v]] / counts[index[v]].sum() if v in index else prior for v in x_val], dtype=np.float64
    )
    if len(classes) == 2:
        return float(roc_auc_score(y_val, proba[:, 1])), "auc"
    return float(accuracy_score(y_val, proba.argmax(axis=1))), "accuracy"


def check_train_test_duplicates(
    data_path: Path,
    split_path: Path,
    feature_cols: list[str],
) -> tuple[int, float]:
    """Check for exact duplicate rows between train and test splits using Polars 64-bit struct hashing."""
    if not data_path.exists() or not split_path.exists() or not feature_cols:
        return 0, 0.0

    data_lf = pl.scan_parquet(data_path)
    split_lf = pl.scan_parquet(split_path).select(SPLIT_COL)

    # Both files are guaranteed strictly row-aligned by ensure_split
    combined = pl.concat([data_lf.select(feature_cols), split_lf], how="horizontal_extend")
    hashed = combined.select(
        [
            pl.struct(feature_cols).hash().alias("__row_hash"),
            pl.col(SPLIT_COL),
        ]
    )

    train_hashes = (
        hashed.filter(pl.col(SPLIT_COL) == TRAIN)
        .select(pl.col("__row_hash").unique())
        .collect()["__row_hash"]
    )
    test_hashes = hashed.filter(pl.col(SPLIT_COL) == TEST).select("__row_hash").collect()["__row_hash"]

    n_test = len(test_hashes)
    if n_test == 0 or len(train_hashes) == 0:
        return 0, 0.0

    n_leaked = int(test_hashes.is_in(train_hashes.implode()).sum())
    pct = round((n_leaked / n_test) * 100.0, 2)
    return n_leaked, pct


def run_data_audit(
    data_path: Path,
    spec: TaskSpec,
    profile: DatasetProfile,
    split_path: Path | None = None,
    *,
    sample_rows: int = PREDICTIVE_SAMPLE_ROWS,
) -> AuditReport:
    """Run comprehensive pre-training leakage and quality checks."""
    t0 = time.perf_counter()
    findings: list[AuditFinding] = []
    dropped_cols: set[str] = set()

    n_rows = profile.n_rows
    target_col = spec.target_column

    # Candidate feature columns to examine (exclude target); the text input of a text task is never dropped
    all_col_names = [c.name for c in profile.columns if c.name != target_col]
    already_dropped = set(spec.drop_columns)
    protected = {target_col, spec.text_column} - {None}

    # 1. Check ID-like and near-unique columns
    for col_prof in profile.columns:
        if col_prof.name in protected or col_prof.name in already_dropped:
            continue
        cname = col_prof.name
        n_unique = col_prof.n_unique
        unique_ratio = n_unique / max(n_rows, 1)
        # Continuous measurements are naturally near-unique; only discrete values can be identifiers.
        is_continuous = col_prof.kind == "numeric" and not col_prof.dtype.lower().startswith(("int", "uint"))
        if is_continuous:
            continue

        is_id_name = bool(_ID_PATTERNS.search(cname))
        if is_id_name and unique_ratio >= 0.90:
            findings.append(
                AuditFinding(
                    check="id_leakage",
                    severity="high",
                    column=cname,
                    message=f"Column '{cname}' appears to be an identifier (unique ratio {unique_ratio:.2%})",
                    suggested_action="drop",
                    metric_name="unique_ratio",
                    metric_value=round(unique_ratio, 4),
                )
            )
            dropped_cols.add(cname)
        elif unique_ratio > 0.99 and n_rows > 100 and col_prof.kind in ("identifier", "categorical"):
            findings.append(
                AuditFinding(
                    check="id_leakage",
                    severity="high",
                    column=cname,
                    message=(
                        f"Column '{cname}' has near-100% unique values ({unique_ratio:.2%}) and may leak IDs"
                    ),
                    suggested_action="drop",
                    metric_name="unique_ratio",
                    metric_value=round(unique_ratio, 4),
                )
            )
            dropped_cols.add(cname)

    # 2. Check target name matching
    for col_prof in profile.columns:
        if col_prof.name in protected or col_prof.name in already_dropped or col_prof.name in dropped_cols:
            continue
        cname = col_prof.name
        if _is_suspicious_target_name(cname, target_col):
            findings.append(
                AuditFinding(
                    check="target_name_match",
                    severity="critical",
                    column=cname,
                    message=(
                        f"Column '{cname}' suspiciously matches or includes the target name '{target_col}'"
                    ),
                    suggested_action="drop",
                )
            )
            dropped_cols.add(cname)

    # 3. Check constant, near-constant, high-missing and high-cardinality columns
    for col_prof in profile.columns:
        if col_prof.name in protected:
            continue
        cname = col_prof.name
        n_unique = col_prof.n_unique
        n_missing = col_prof.n_missing
        missing_ratio = n_missing / max(n_rows, 1)

        if n_unique <= 1:
            findings.append(
                AuditFinding(
                    check="constant_column",
                    severity="high",
                    column=cname,
                    message=f"Column '{cname}' is constant (has only {n_unique} unique value)",
                    suggested_action="drop",
                    metric_name="n_unique",
                    metric_value=float(n_unique),
                )
            )
            dropped_cols.add(cname)
        elif col_prof.top_values:
            top_share = max(col_prof.top_values.values()) if col_prof.top_values else 0.0
            if top_share >= 0.995 and n_rows > 200:
                findings.append(
                    AuditFinding(
                        check="near_constant_column",
                        severity="medium",
                        column=cname,
                        message=(
                            f"Column '{cname}' is near-constant ({top_share:.1%} of rows take a single value)"
                        ),
                        suggested_action="flag",
                        metric_name="top_value_share",
                        metric_value=round(top_share, 4),
                    )
                )

        if missing_ratio >= 0.80:
            findings.append(
                AuditFinding(
                    check="missing_values",
                    severity="high",
                    column=cname,
                    message=f"Column '{cname}' has {missing_ratio:.1%} missing values",
                    suggested_action="drop",
                    metric_name="missing_ratio",
                    metric_value=round(missing_ratio, 4),
                )
            )
            dropped_cols.add(cname)
        elif missing_ratio >= 0.50:
            findings.append(
                AuditFinding(
                    check="missing_values",
                    severity="medium",
                    column=cname,
                    message=f"Column '{cname}' has {missing_ratio:.1%} missing values",
                    suggested_action="flag",
                    metric_name="missing_ratio",
                    metric_value=round(missing_ratio, 4),
                )
            )

        if col_prof.kind == "categorical" and n_unique > 200 and (n_unique / max(n_rows, 1)) > 0.3:
            findings.append(
                AuditFinding(
                    check="high_cardinality",
                    severity="medium",
                    column=cname,
                    message=(
                        f"Column '{cname}' is a high-cardinality categorical ({n_unique} distinct values)"
                    ),
                    suggested_action="flag",
                    metric_name="n_unique",
                    metric_value=float(n_unique),
                )
            )

    # 4. Check class imbalance for classification tasks
    if spec.task_type != TaskType.tabular_regression:
        target_prof = profile.column(target_col)
        if target_prof and target_prof.top_values:
            shares = list(target_prof.top_values.values())
            if shares:
                min_share = min(shares)
                if min_share < 0.01:
                    findings.append(
                        AuditFinding(
                            check="class_imbalance",
                            severity="high",
                            column=target_col,
                            message=(
                                f"Extreme class imbalance: minority class is only {min_share:.2%} of rows"
                            ),
                            suggested_action="stratify",
                            metric_name="minority_share",
                            metric_value=round(min_share, 4),
                        )
                    )
                elif min_share < 0.05:
                    findings.append(
                        AuditFinding(
                            check="class_imbalance",
                            severity="medium",
                            column=target_col,
                            message=f"Substantial class imbalance: minority class is {min_share:.2%}",
                            suggested_action="stratify",
                            metric_name="minority_share",
                            metric_value=round(min_share, 4),
                        )
                    )

    # 5. Check univariate predictive power (near-perfect predictor leakage)
    candidates_for_power = [
        c for c in all_col_names if c not in dropped_cols and c not in already_dropped and c not in protected
    ]
    if candidates_for_power and data_path.exists():
        try:
            lf = scan_table(data_path)
            # Take a sample for fast evaluation
            step = max(1, n_rows // max(sample_rows, 1))
            sample_df = (
                lf.select([*candidates_for_power, target_col]).gather_every(step).head(sample_rows).collect()
            )
            for cname in candidates_for_power:
                score, metric_name = _evaluate_univariate_power(sample_df, cname, target_col, spec.task_type)
                if score >= UNIVARIATE_LEAKAGE_THRESHOLD:
                    findings.append(
                        AuditFinding(
                            check="predictive_leakage",
                            severity="critical",
                            column=cname,
                            message=(
                                f"Feature '{cname}' achieves near-perfect univariate predictive power "
                                f"({metric_name}={score:.4f}) against target; likely target leakage."
                            ),
                            suggested_action="drop",
                            metric_name=metric_name,
                            metric_value=round(score, 4),
                        )
                    )
                    dropped_cols.add(cname)
        except Exception:  # the audit must never crash the pipeline
            log.warning("univariate leakage check failed", exc_info=True)

    # 6. Check train/test duplicate rows across splits
    n_dups, dup_pct = 0, 0.0
    if split_path and split_path.exists() and data_path.exists():
        clean_features = [c for c in all_col_names if c not in dropped_cols and c not in already_dropped]
        if clean_features:
            n_dups, dup_pct = check_train_test_duplicates(data_path, split_path, clean_features)
            if n_dups > 0:
                severity = "critical" if dup_pct >= 10.0 else ("high" if dup_pct >= 1.0 else "medium")
                findings.append(
                    AuditFinding(
                        check="split_leakage",
                        severity=severity,
                        column=None,
                        message=(
                            f"Detected {n_dups:,} test rows ({dup_pct:.2f}% of test set) that duplicate "
                            "training rows exactly across feature columns."
                        ),
                        suggested_action="flag",
                        metric_name="duplicate_pct",
                        metric_value=dup_pct,
                        details={"duplicate_count": n_dups, "test_rows_pct": dup_pct},
                    )
                )

    has_leakage = any(f.severity in ("critical", "high") and "leakage" in f.check for f in findings)
    duration = time.perf_counter() - t0

    return AuditReport(
        findings=findings,
        dropped_columns=sorted(dropped_cols),
        train_test_duplicates=n_dups,
        duplicate_pct=dup_pct,
        has_leakage=has_leakage,
        duration_s=round(duration, 3),
    )


def apply_audit_to_task_spec(report: AuditReport, spec: TaskSpec) -> TaskSpec:
    """Update TaskSpec with auto-dropped columns and transparent assumptions."""
    new_drops = [c for c in report.dropped_columns if c not in spec.drop_columns]
    if not new_drops and report.train_test_duplicates == 0:
        return spec

    updated_drops = list(spec.drop_columns) + new_drops
    updated_assumptions = list(spec.assumptions)

    for finding in report.findings:
        if finding.suggested_action == "drop" and finding.column in new_drops:
            updated_assumptions.append(
                f"Data audit automatically dropped '{finding.column}': {finding.message}"
            )

    if report.train_test_duplicates > 0:
        updated_assumptions.append(
            f"Data audit flagged {report.train_test_duplicates} train/test duplicate rows "
            f"({report.duplicate_pct:.2f}% of test split)."
        )

    return spec.model_copy(
        update={
            "drop_columns": updated_drops,
            "assumptions": updated_assumptions,
        }
    )
