# Data audit and model card

*Issue #5.* Grounded verification picks the plan with the best **measured** score. That makes leakage more dangerous here than in the paper. A column that encodes the answer gives the best measured score, so it wins every time. The data audit removes such columns *before* any plan is trained. The model card then explains what the selected model relies on.

## Pre-training data audit

`tools/data_audit.py:run_data_audit` runs after the split is written (stage `prepare`). It is deterministic, uses no LLM, and runs on Polars.

| Check | Rule | Action |
|---|---|---|
| Identifier | Name looks like an id (`id`, `uuid`, `guid`, `key`, `hash`, `code`, ...) and ≥ 90% unique, or any discrete column ≥ 99% unique. Continuous numbers are never treated as ids | drop (high) |
| Target name | Column name contains the target's name plus an affix such as `pred`, `true`, `label`, `outcome`, `after`, `copy` | drop (critical) |
| Predictive leakage | One feature alone predicts the target with AUC, accuracy or R² ≥ 0.98 on a shuffled 80/20 holdout of up to 10k rows. Numbers use a depth-2 tree; categories use a per-category lookup, so a 40-level code that maps to the target is caught | drop (critical) |
| Train/test duplicates | Exact duplicate feature rows shared by train and test, found with 64-bit row hashes in Polars (5M rows in a few seconds) | flag: critical ≥ 10%, high ≥ 1% |
| Constant columns | One distinct value | drop |
| Missing values | ≥ 80% missing → drop; ≥ 50% → flag | drop / flag |
| Near-constant | ≥ 99.5% of rows share one value | flag |
| High cardinality | Categorical with > 200 levels and > 30% unique | flag |
| Class imbalance | Minority class < 1% (high) or < 5% (medium) | flag (stratify) |

The text column of a text task is never dropped.

`apply_audit_to_task_spec` merges the drops into `TaskSpec.drop_columns` and records each one as a visible assumption. The findings also become a knowledge item for the planners. The UI's audit panel lists every finding with its metric. Set `DATA_AUDIT=false` to switch the audit off, for example in an ablation. The benchmark records how many columns the audit dropped (`audit_dropped`) and whether that changed the outcome.

!!! example "On real data"
    On the 8.9M-row Microsoft Malware set, the audit takes about 25 s. It drops four columns that are between 83% and 99.97% empty (`PuaMode`, `Census_ProcessorClass`, `DefaultBrowsersIdentifier`, `Census_IsFlightingInternal`) and flags about 1,000 test rows that duplicate training rows. On the planted-leak sample (`data/samples/customer_churn_leaky.csv`), it drops `cancellation_confirmation` (AUC 1.0) and `account_guid`.

## Model card

`extensions/model_card.py:ModelCardHook` runs in `on_run_finished` for successful runs. It writes `model_card.json` and `model_card.md` next to the model, and they are included in the deployment bundle. It reloads the saved bundle and prepares up to 50,000 held-out test rows **through the serving module**, so the card describes exactly the model that `/predict` serves.

| Section | Content |
|---|---|
| Feature importances | Native gain (LightGBM, XGBoost, tree ensembles), otherwise permutation importance on the prepared test sample. For TF-IDF text models, the most influential tokens by coefficient |
| Classification | Confusion matrix, per-class precision, recall, F1 and support; for binary tasks a 10-bin reliability curve and Brier score |
| Regression | MAE, RMSE, R², max error, residual quantiles |
| Forecasting | Importances of the LightGBM forecaster; residuals of the rolling-origin test backtest |
| Audit summary | Dropped columns, duplicates, unresolved high-severity findings |
| Why this model | A narrative built only from measured numbers: family, test score, top features, audit, calibration or residuals |

Diagnostics run in a worker thread. A failure is logged and never fails a finished run.
