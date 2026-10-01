"""Optuna + LightGBM baseline script (rendered with CONFIG and run in the sandbox).

A strong, non-LLM AutoML baseline for tabular tasks: TPE search over LightGBM
hyperparameters on the same train/valid split as the agents, stopped by a
wall-clock budget, then the best configuration is scored on the same test split.
"""

import json
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

CONFIG = __CONFIG__

IS_CLF = CONFIG["task_type"] == "tabular_classification"
SPLITS = {"train": 0, "valid": 1, "test": 2}
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load(name):
    data = pl.scan_parquet(CONFIG["data_path"])
    lf = pl.concat([data, pl.scan_parquet(CONFIG["split_path"])], how="horizontal")
    lf = lf.filter(pl.col("__split") == SPLITS[name])
    cap = CONFIG.get("max_rows", {}).get(name)
    if cap:
        n = lf.select(pl.len()).collect().item()
        if n > cap:
            lf = lf.filter(pl.col("__r") < cap / n)
    excluded = {CONFIG["target"], *CONFIG.get("drop_columns", [])}
    cols = [c for c in data.collect_schema().names() if c not in excluded]
    df = lf.select([*cols, CONFIG["target"]]).collect()
    df = df.with_columns([pl.col(c).cast(pl.Categorical) for c, t in df.schema.items() if t == pl.String and c in cols])
    pdf = df.to_pandas()
    return pdf[cols], pdf[CONFIG["target"]]


def score(metric, y, pred, proba):
    if IS_CLF:
        values = {
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro"),
            "f1_weighted": f1_score(y, pred, average="weighted"),
        }
        try:
            values["roc_auc"] = (
                roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2 else roc_auc_score(y, proba, multi_class="ovr")
            )
        except ValueError:
            pass
    else:
        values = {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "mae": mean_absolute_error(y, pred),
                  "r2": r2_score(y, pred)}
        if (y >= 0).all():
            values["rmsle"] = float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2)))
    return {k: float(v) for k, v in values.items()}


def main():
    start = time.time()
    X_tr, y_tr = load("train")
    X_va, y_va = load("valid")
    X_te, y_te = load("test")
    for col in X_tr.select_dtypes("category").columns:
        for frame in (X_va, X_te):
            frame[col] = pd.Categorical(frame[col].astype(object), categories=X_tr[col].cat.categories)
    if IS_CLF:
        classes = sorted(pd.unique(y_tr.dropna()), key=str)
        lookup = {c: i for i, c in enumerate(classes)}
        y_tr, y_va, y_te = (y.map(lookup) for y in (y_tr, y_va, y_te))
        keep_va, keep_te = y_va.notna(), y_te.notna()
        X_va, y_va, X_te, y_te = X_va[keep_va], y_va[keep_va].astype(int), X_te[keep_te], y_te[keep_te].astype(int)
        y_tr = y_tr.astype(int)

    metric = CONFIG["metric"]
    higher = metric not in {"rmse", "mae", "mape", "rmsle"}
    Model = lgb.LGBMClassifier if IS_CLF else lgb.LGBMRegressor

    def fit(params):
        model = Model(n_estimators=2000, random_state=42, n_jobs=CONFIG.get("n_jobs", -1), verbose=-1, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return model

    def evaluate(model, X, y):
        proba = model.predict_proba(X) if IS_CLF else None
        return score(metric, y, model.predict(X), proba)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        value = evaluate(fit(params), X_va, y_va).get(metric, np.nan)
        return value if higher else -value

    budget = max(5.0, CONFIG["time_budget_s"] - (time.time() - start))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, timeout=budget * 0.8, n_trials=CONFIG.get("max_trials", 200))
    best = fit({**study.best_params, "subsample_freq": 1})
    metrics = evaluate(best, X_te, y_te)
    result = {
        "metric": metric,
        "score": metrics.get(metric),
        "metrics": metrics,
        "split": "test",
        "model_family": "lightgbm (optuna)",
        "n_trials": len(study.trials),
        "best_params": study.best_params,
        "wall_s": round(time.time() - start, 2),
    }
    with open("metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
