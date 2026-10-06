"""Post-training Model Card extension hook.

Generates a comprehensive, diagnostic Model Card in JSON and Markdown after a successful run:
1. Feature importances (native gain for LightGBM/XGBoost, permutation fallback).
2. Per-class metrics and confusion matrix (classification) or residual diagnostics (regression).
3. Probability calibration curve and Brier score.
4. "Why this model" section combining audit, grounded observations,
   and importances (grounded in measured numbers).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from automl_agent.execution.inference import InferenceError, load_bundle, predict_dataframe, validate_input
from automl_agent.extensions import PipelineHooks
from automl_agent.schemas.events import Stage
from automl_agent.schemas.model_card import (
    CalibrationPoint,
    CalibrationReport,
    ConfusionMatrixData,
    FeatureImportance,
    ModelCard,
    ResidualsData,
)
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.tools.splits import SPLIT_COL, TEST

if TYPE_CHECKING:
    from automl_agent.agents.context import RunContext
    from automl_agent.agents.manager import PipelineResult


log = logging.getLogger(__name__)

MAX_DIAGNOSTIC_ROWS = 50_000


def extract_feature_importances(
    model: Any,
    feature_names: list[str],
    X_sample: pd.DataFrame | np.ndarray | None = None,
    y_sample: np.ndarray | None = None,
) -> list[FeatureImportance]:
    """Extract native gain or fallback permutation importances."""
    actual_model = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        actual_model = model.named_steps["model"]

    # 1. LightGBM native gain
    if hasattr(actual_model, "booster_"):
        try:
            gains = actual_model.booster_.feature_importance(importance_type="gain")
            names = actual_model.booster_.feature_name()
            if len(gains) == len(names):
                total = float(sum(gains)) or 1.0
                pairs = sorted(
                    zip(names, [float(g / total) for g in gains], strict=False),
                    key=lambda p: p[1],
                    reverse=True,
                )
                return [
                    FeatureImportance(feature=f, importance=round(imp, 4), method="native_gain")
                    for f, imp in pairs
                ]
        except Exception:
            pass

    # 2. XGBoost native gain
    if hasattr(actual_model, "get_booster"):
        try:
            booster = actual_model.get_booster()
            score_dict = booster.get_score(importance_type="total_gain")
            total = float(sum(score_dict.values())) or 1.0
            results = []
            for f in feature_names:
                imp = float(score_dict.get(f, 0.0)) / total
                results.append((f, imp))
            results.sort(key=lambda p: p[1], reverse=True)
            return [
                FeatureImportance(feature=f, importance=round(imp, 4), method="native_gain")
                for f, imp in results
            ]
        except Exception:
            pass

    # 3. Scikit-learn feature_importances_
    if hasattr(actual_model, "feature_importances_"):
        try:
            fi = actual_model.feature_importances_
            if len(fi) == len(feature_names):
                total = float(sum(fi)) or 1.0
                pairs = sorted(
                    zip(feature_names, [float(val / total) for val in fi], strict=False),
                    key=lambda p: p[1],
                    reverse=True,
                )
                return [
                    FeatureImportance(feature=f, importance=round(imp, 4), method="native_gain")
                    for f, imp in pairs
                ]
        except Exception:
            pass

    # 4. Fallback: Permutation importance
    if X_sample is not None and y_sample is not None and len(X_sample) >= 10:
        try:
            n_sub = min(len(X_sample), 300)
            X_sub = X_sample.iloc[:n_sub] if isinstance(X_sample, pd.DataFrame) else X_sample[:n_sub]
            perm = permutation_importance(model, X_sub, y_sample[:n_sub], n_repeats=3, random_state=42)
            raw = np.maximum(0, perm.importances_mean)
            total = float(np.sum(raw)) or 1.0
            pairs = sorted(
                zip(feature_names, [float(val / total) for val in raw], strict=False),
                key=lambda p: p[1],
                reverse=True,
            )
            return [
                FeatureImportance(feature=f, importance=round(imp, 4), method="permutation")
                for f, imp in pairs
            ]
        except Exception:
            pass

    return [FeatureImportance(feature=f, importance=0.0, method="native_gain") for f in feature_names]


def format_model_card_markdown(card: ModelCard) -> str:
    """Format model card as a GitHub-flavored Markdown document."""
    score_str = f"{card.primary_score:.4f}" if card.primary_score is not None else "N/A"
    lines = [
        f"# Model Card: {card.model_family}",
        "",
        f"- **Task Type**: `{card.task_type.value}`",
        f"- **Primary Metric**: `{card.primary_metric}` = **{score_str}**",
        "",
        "## Why This Model",
        card.why_this_model or "Selected via grounded verification based on observed performance.",
        "",
        "## Top Features by Importance",
        "| Feature | Relative Importance | Extraction Method |",
        "|---|---|---|",
    ]
    for fi in card.feature_importances[:10]:
        lines.append(f"| `{fi.feature}` | {fi.importance:.4f} | `{fi.method}` |")
    lines.append("")

    if card.confusion_matrix:
        cm = card.confusion_matrix
        lines.append("## Confusion Matrix & Per-Class Metrics")
        lines.append("| Metric | " + " | ".join(f"`{lbl}`" for lbl in cm.labels) + " |")
        lines.append("|---" * (len(cm.labels) + 1) + "|")
        for metric_name in ("precision", "recall", "f1-score", "support"):
            vals = [str(round(cm.per_class.get(lbl, {}).get(metric_name, 0.0), 3)) for lbl in cm.labels]
            lines.append(f"| **{metric_name}** | " + " | ".join(vals) + " |")
        lines.append("")

    if card.calibration:
        lines.append("## Probability Calibration")
        lines.append(f"- **Brier Score**: `{card.calibration.brier_score:.4f}`")
        if card.calibration.points:
            lines.append("| Predicted Prob Bin | Observed True Rate |")
            lines.append("|---|---|")
            for pt in card.calibration.points:
                lines.append(f"| {pt.prob_pred:.3f} | {pt.prob_true:.3f} |")
        lines.append("")

    if card.residuals:
        r = card.residuals
        lines.append("## Residuals Analysis")
        lines.append(f"- **MAE**: `{r.mae:.4f}`")
        lines.append(f"- **RMSE**: `{r.rmse:.4f}`")
        lines.append(f"- **R²**: `{r.r2:.4f}`")
        lines.append(f"- **Max Error**: `{r.max_error:.4f}`")
        lines.append("")

    if card.audit_summary:
        lines.append("## Pre-training Data Audit Summary")
        for item in card.audit_summary:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def _test_sample(
    ctx: RunContext, spec: TaskSpec, bundle: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Up to ``MAX_DIAGNOSTIC_ROWS`` rows of the held-out test split, prepared exactly as for serving."""
    if not (ctx.dataset_path.exists() and ctx.split and ctx.split.path.exists()):
        return None
    target = spec.target_column
    frame = (
        pl.concat(
            [pl.scan_parquet(ctx.dataset_path), pl.scan_parquet(ctx.split.path)], how="horizontal_extend"
        )
        .filter(pl.col(SPLIT_COL) == TEST)
        .head(MAX_DIAGNOSTIC_ROWS)
        .collect()
        .to_pandas()
    )
    frame = frame[frame[target].notna()]
    if bundle.get("classes") is not None:  # labels unseen in training cannot be scored
        frame = frame[frame[target].astype(str).isin([str(c) for c in bundle["classes"]])]
    if len(frame) < 5:
        return None
    X = validate_input(frame[[c for c in frame.columns if c in bundle["features"]]], bundle)
    return X, frame[target]


def _text_coefficients(bundle: dict[str, Any], top: int = 20) -> list[FeatureImportance]:
    """Most influential tokens of a TF-IDF linear text model (hashing models have no token names)."""
    steps = getattr(bundle["model"], "named_steps", {})
    vec, clf = steps.get("tfidf"), steps.get("model")
    coef = getattr(clf, "coef_", None)
    if vec is None or coef is None:
        return []
    weights = np.abs(np.asarray(coef)).max(axis=0)
    order = np.argsort(weights)[::-1][:top]
    total = float(weights[order].sum()) or 1.0
    names = vec.get_feature_names_out()
    return [
        FeatureImportance(
            feature=str(names[i]), importance=round(float(weights[i] / total), 4), method="coefficient"
        )
        for i in order
    ]


def _encode(y: pd.Series, bundle: dict[str, Any]) -> np.ndarray:
    """Training label codes (indices into the bundle's classes) for permutation importance."""
    classes = bundle.get("classes")
    lookup = {str(c): i for i, c in enumerate(classes if classes is not None else [])}
    return y.astype(str).map(lookup).to_numpy(dtype=int)


Diagnostics = tuple[
    list[FeatureImportance], ConfusionMatrixData | None, CalibrationReport | None, ResidualsData | None
]


def _residuals(y_true: np.ndarray, y_p: np.ndarray) -> ResidualsData:
    residuals = y_p - y_true
    q = np.percentile(residuals, [5, 25, 50, 75, 95])
    return ResidualsData(
        mae=round(float(mean_absolute_error(y_true, y_p)), 4),
        rmse=round(float(np.sqrt(mean_squared_error(y_true, y_p))), 4),
        r2=round(float(r2_score(y_true, y_p)), 4),
        max_error=round(float(max_error(y_true, y_p)), 4),
        quantiles={k: round(float(v), 4) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q, strict=True)},
        sample_residuals=[round(float(val), 4) for val in residuals[:50]],
    )


def _diagnose_forecaster(bundle: dict[str, Any], artifact_dir: Path) -> Diagnostics:
    """Importances of a LightGBM forecaster and residuals of its rolling-origin test backtest."""
    model = bundle["model"]
    regressor = model.get("regressor") if isinstance(model, dict) else None
    fi_list = extract_feature_importances(regressor, list(regressor.feature_name_)) if regressor else []
    res = None
    path = artifact_dir / "test_predictions.parquet"
    if path.exists():
        preds = pd.read_parquet(path)
        if len(preds) >= 5:
            res = _residuals(preds["y_true"].to_numpy(dtype=float), preds["y_pred"].to_numpy(dtype=float))
    return fi_list, None, None, res


def _diagnose(ctx: RunContext, spec: TaskSpec, bundle: dict[str, Any], artifact_dir: Path) -> Diagnostics:
    """Feature importances and test-split diagnostics for the selected model."""
    if spec.task_type == TaskType.time_series_forecasting:
        return _diagnose_forecaster(bundle, artifact_dir)
    sample = _test_sample(ctx, spec, bundle)
    model = bundle["model"]
    if spec.task_type == TaskType.text_classification:
        fi_list = _text_coefficients(bundle)
    elif sample is not None:
        X, y = sample
        y_fit = (
            y.to_numpy(dtype=float) if spec.task_type == TaskType.tabular_regression else _encode(y, bundle)
        )
        fi_list = extract_feature_importances(model, bundle["features"], X, y_fit)
    else:
        fi_list = extract_feature_importances(model, bundle["features"])
    if sample is None:
        return fi_list, None, None, None

    X, y = sample
    out = predict_dataframe(X, bundle)
    if spec.task_type == TaskType.tabular_regression:
        return (
            fi_list,
            None,
            None,
            _residuals(y.to_numpy(dtype=float), np.asarray(out["predictions"], dtype=float)),
        )

    y_true = y.astype(str).to_numpy()
    y_pred = np.asarray(out["predictions"], dtype=object).astype(str)
    labels = out.get("classes") or sorted(set(y_true) | set(y_pred))
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        lbl: {
            "precision": round(float(rep[lbl]["precision"]), 4),
            "recall": round(float(rep[lbl]["recall"]), 4),
            "f1-score": round(float(rep[lbl]["f1-score"]), 4),
            "support": int(rep[lbl]["support"]),
        }
        for lbl in labels
        if lbl in rep
    }
    cm = ConfusionMatrixData(
        labels=list(labels),
        matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        per_class=per_class,
    )
    cal = None
    if "probabilities" in out and len(labels) == 2:
        p_pos = np.asarray(out["probabilities"])[:, 1]
        y_pos = (y_true == labels[1]).astype(int)
        prob_true, prob_pred = calibration_curve(y_pos, p_pos, n_bins=10, strategy="uniform")
        cal = CalibrationReport(
            brier_score=round(float(brier_score_loss(y_pos, p_pos)), 4),
            points=[
                CalibrationPoint(prob_pred=round(float(pp), 4), prob_true=round(float(pt), 4))
                for pp, pt in zip(prob_pred, prob_true, strict=True)
            ],
        )
    return fi_list, cm, cal, None


class ModelCardHook(PipelineHooks):
    """Generates post-training model card artifacts on run completion."""

    async def on_run_finished(self, ctx: RunContext, result: PipelineResult) -> None:
        """Inspect the winning model, compute diagnostics, and write model card artifacts."""
        if not result.success or not result.task_spec or not result.metrics:
            return
        artifact_dir_str = result.metrics.get("artifact_dir")
        if not artifact_dir_str:
            return
        try:
            bundle = load_bundle(Path(artifact_dir_str) / "model.joblib")
        except InferenceError:
            return
        try:
            parts = await asyncio.to_thread(_diagnose, ctx, result.task_spec, bundle, Path(artifact_dir_str))
        except Exception:  # a model card must never fail a finished run
            log.warning("model card diagnostics failed", exc_info=True)
            parts = ([], None, None, None)
        await self._write_card(ctx, result, Path(artifact_dir_str), *parts)

    async def _write_card(
        self,
        ctx: RunContext,
        result: PipelineResult,
        artifact_dir: Path,
        fi_list: list[FeatureImportance],
        cm_data: ConfusionMatrixData | None,
        cal_data: CalibrationReport | None,
        res_data: ResidualsData | None,
    ) -> None:
        spec = result.task_spec
        assert spec is not None and result.metrics is not None

        # 3. Audit summary from context
        audit_report = ctx.state.get("audit_report")
        audit_summary = []
        if audit_report:
            if audit_report.dropped_columns:
                audit_summary.append(f"Audit dropped columns: {', '.join(audit_report.dropped_columns)}")
            if audit_report.train_test_duplicates > 0:
                audit_summary.append(
                    f"Flagged {audit_report.train_test_duplicates:,} train/test duplicate rows "
                    f"({audit_report.duplicate_pct:.2f}% of test split)"
                )
            for f in audit_report.findings:
                if f.severity in ("critical", "high") and f.column not in audit_report.dropped_columns:
                    audit_summary.append(f"{f.check}: {f.message}")

        # 4. "Why this model" section (synthesize narrative grounded only in measured facts)
        top_fi_str = ", ".join(f"{fi.feature} ({fi.importance:.1%})" for fi in fi_list[:3])
        score_val = result.metrics.get("score")
        score_repr = f"{score_val:.4f}" if isinstance(score_val, float) else str(score_val)
        family = result.plan.model_family if result.plan else "best candidate"

        narrative = (
            f"The pipeline selected {family}, which reached {spec.metric} "
            f"({score_repr}) on the held-out test split after grounded verification ranked the "
            "candidates on validation data. "
            f"Key contributing features are {top_fi_str or 'features across the dataset'}. "
        )
        if audit_summary:
            narrative += f"Pre-training audit: {'; '.join(audit_summary)}. "
        if cal_data:
            narrative += f"Probability calibration achieved a Brier score of {cal_data.brier_score:.4f}. "
        if res_data:
            narrative += f"Residual evaluation confirmed MAE={res_data.mae:.4f} and R²={res_data.r2:.4f}. "

        card = ModelCard(
            task_type=spec.task_type,
            model_family=family,
            primary_metric=spec.metric,
            primary_score=score_val if isinstance(score_val, float) else None,
            feature_importances=fi_list,
            confusion_matrix=cm_data,
            residuals=res_data,
            calibration=cal_data,
            audit_summary=audit_summary,
            why_this_model=narrative.strip(),
            metadata={"created_at": datetime.now(UTC).isoformat()},
        )

        # 5. Save model_card.json and model_card.md
        card_json = card.model_dump_json(indent=2)
        card_md = format_model_card_markdown(card)

        (artifact_dir / "model_card.json").write_text(card_json, encoding="utf-8")
        (artifact_dir / "model_card.md").write_text(card_md, encoding="utf-8")
        (ctx.workdir / "model_card.json").write_text(card_json, encoding="utf-8")
        (ctx.workdir / "model_card.md").write_text(card_md, encoding="utf-8")

        result.metrics["model_card"] = card.model_dump(mode="json")
        await ctx.emit(
            Stage.done,
            "model_card",
            f"Generated model card for {family} ({spec.metric}={score_repr})",
            kind="artifact",
            payload={"model_card": card.model_dump(mode="json")},
        )
