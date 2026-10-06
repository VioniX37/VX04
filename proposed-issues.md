# Proposed issues

Four features, each written as a GitHub issue. They are ordered by impact. Every one closes a gap that is already documented in the repo: the paper comparison, the proposal's "Future work" section, or an unused extension hook.

---

## 1. Model serving: prediction API, batch scoring and an exportable inference bundle

**Labels:** `enhancement`, `backend`, `frontend`, `paper-parity`

### Problem
A successful run ends with `model.joblib` in the run's workspace, and nothing can use it. The paper delivers "deployment-ready code + inference endpoint". [`docs/research/comparison.md`](docs/research/comparison.md) lists this as *"endpoint left as future work"*. Today a user can train a model but cannot get a prediction out of it without writing Python by hand.

### Proposal
- **Inference module.** Add an inference module (`execution/inference.py`) that loads the bundle written by the templates (`{"model", "classes", "features"}`) and applies **the same preprocessing** used in training. That covers column order, categorical dtypes and dropped columns from `TaskSpec.drop_columns`. Training and serving must share a single code path, so the template's preprocessing must be importable and not only rendered into the script.
- **REST endpoints:**
  - `POST /api/runs/{id}/predict` takes JSON records and returns predictions (plus probabilities for classification).
  - `POST /api/runs/{id}/predict/batch` takes a CSV or Parquet upload and returns a scored file. It must stream for large files, reusing the Parquet ingest path.
  - `GET /api/runs/{id}/artifacts/bundle` returns a zip with `model.joblib`, `predict.py`, `requirements.txt` (pinned versions), `schema.json` (input columns and dtypes) and `metrics.json`.
- **CLI:** `automl-agent predict --run <id> --data new.csv --out scored.parquet`.
- **UI:** a "Use this model" panel on a finished run. It should offer a small form generated from `schema.json` for single predictions, a file drop for batch scoring, and a bundle download button.
- **Validation:** reject input with missing or extra columns or the wrong dtypes. The error should name the column and must not return a 500.

### Acceptance criteria
- [ ] For each sample dataset (tabular classification, regression and text), predictions from the endpoint match the predictions the training script made on the test split.
- [ ] Batch scoring of a 1M-row file completes with memory bounded by `EXEC_MAX_MEM_MB`.
- [ ] The exported bundle runs in a fresh venv using only its own `requirements.txt`.
- [ ] Tests cover schema mismatches and calls on a run that failed or has no model.
- [ ] REST, CLI and UI docs are updated, and the comparison table row is updated.

### Pointers
`execution/templates/tabular.py:252` (bundle format), `execution/templates/text_classification.py`, `api/runs.py`, `cli.py`, `tools/ingest.py`

---

## 2. Human-in-the-loop run control: plan approval, cancel and resume

**Labels:** `enhancement`, `backend`, `frontend`, `ux`

### Problem
After a run starts, the user can only watch it. They cannot veto a plan they know is wrong, for example a model family they cannot deploy or a feature they know leaks. They also cannot stop a run that is burning free-tier quota. `RunStatus` has only `pending/running/succeeded/failed`. The proposal lists "human-in-the-loop plan approval" as future work, and the `on_plans_ranked` hook in `extensions/__init__.py` was designed for this but nothing uses it.

### Proposal
- **New run states:** `awaiting_input` and `cancelled`. This needs a DB migration through the existing forward-migration helper in `storage/db.py`.
- **Approval mode** (`RunCreate.approval: "auto" | "plans" | "plans+code"`, default `auto` so CLI and evaluation behaviour do not change):
  - `plans`: after grounded verification ranks the plans, the run pauses and the UI shows the ranked plans with their **observed** scores. The user can approve the top plan, pick a different one, or edit it (drop a model family, add a constraint, change a hyperparameter range) and send it back.
  - `plans+code`: the user also reviews the generated training script as a diff against the template before it runs.
  - If nobody responds before a configurable timeout, the run falls back to `auto` so unattended runs never hang forever.
- **Cancel:** `POST /api/runs/{id}/cancel`. It kills the sandbox subprocess tree, stops LLM calls at the next await point, persists partial observations, and emits a `run_cancelled` event.
- **Resume:** a paused run must survive a backend restart. Build on the orphan-recovery logic from commit `238cb83`.
- **Memory:** record whether the human overrode the agent's choice. This becomes a useful signal for the experience memory.

### Acceptance criteria
- [ ] A run in `plans` mode pauses, appears as "Needs your input" in the run list, and continues correctly after approval, pick or edit.
- [ ] Cancel ends a running training subprocess within a few seconds, and no orphan processes are left (checked on Windows and Linux).
- [ ] A backend restart while a run is `awaiting_input` does not lose the run.
- [ ] `auto` mode behaves exactly as before, and the existing tests pass unchanged.
- [ ] The event schema docs cover the new events.

### Pointers
`extensions/__init__.py` (`on_plans_ranked`), `agents/manager.py:419`, `schemas/run.py`, `services/run_service.py`, `execution/sandbox.py`, `docs/reference/event-schema.md`

---

## 3. Data audit and trust report: leakage detection before training, model card after

**Labels:** `enhancement`, `backend`, `frontend`, `research`

### Problem
Grounded verification selects the plan with the best **observed** validation score. That makes the system *more* vulnerable to target leakage than the paper's approach. A leaked column, such as a post-outcome field, an ID correlated with the label, or duplicate rows across splits, produces the highest score and wins successive halving every time. Right now the only defence is the LLM remembering to put the column in `drop_columns`. After training, the user gets one score and no explanation of what the model relies on.

### Proposal
**A. Pre-training data audit** (deterministic, uses Polars, runs after `Prepare`):
- **Leakage checks.** Check single-feature predictive power against the target using a fast univariate model on a sample, and flag near-perfect predictors. Also flag near-unique ID-like columns, columns whose names match the target, and exact or near-duplicate rows shared between the train and test splits.
- **Quality checks.** Check class imbalance severity, constant and near-constant columns, high-cardinality categoricals, and the share of missing values in each column.
- **Findings.** Each finding is a structured `AuditFinding` with a severity and a suggested action. High-severity findings feed into the task spec (auto-drop plus a surfaced assumption) and into planning knowledge. They appear in the UI before planning starts.

**B. Post-training model card**, generated through the `on_run_finished` hook:
- Feature importances: native gain for LightGBM and XGBoost, and permutation importance on a test sample as a model-agnostic fallback.
- Per-class metrics and a confusion matrix, or a residual plot for regression.
- Probability calibration (reliability curve and Brier score).
- A Markdown and JSON `model_card` artifact that is included in the bundle from issue #3.
- A "Why this model" section that combines the audit, the grounded observations and the importances. It is LLM-written but cites only measured numbers.

### Acceptance criteria
- [ ] A synthetic dataset with a planted leaky column is flagged, and with the audit on, the winning model does not use that column. Add the dataset to `data/samples/`.
- [ ] Train/test duplicate detection works on the 5M-row synthetic set within a few seconds.
- [ ] The model card renders in the UI for every supported task type.
- [ ] The audit can be turned off (`DATA_AUDIT=false`) for ablations, and the evaluation harness reports how often it changed the outcome.

### Pointers
`tools/dataset_profiler.py`, `tools/splits.py`, `schemas/task_spec.py:49` (`drop_columns`), `verification/request.py`, `extensions/__init__.py`, `evaluation/`

---

## 4. New modality: time-series forecasting

**Labels:** `enhancement`, `backend`, `research`, `paper-parity`

### Problem
The project supports only tabular classification, tabular regression and text classification. The paper evaluates 7 task types, and time-series forecasting is the one that best fits this project's strengths: tabular, large and CPU-friendly. Without it, RQ4 (matching the paper's results) covers only part of the benchmark. Forecasting is also one of the most common real-world requests. ADR 0004 states that adding a modality should be "a local change", and this issue tests that claim.

### Proposal
- **Task type:** `TaskType.time_series_forecasting`. The `TaskSpec` gets new fields: `time_column`, `horizon`, `frequency`, an optional `series_id_columns` for many-series panels, and metrics `mae`, `rmse`, `mape` and `smape` (default `smape`).
- **Splits:** a temporal split mode in `tools/splits.py`. Train, valid and test are contiguous time blocks, never shuffled. The nested subsamples used by grounded verification should be **suffix windows** (the most recent data) rather than random rows, so that the low-fidelity rungs stay meaningful.
- **Template:** `execution/templates/time_series.py` with:
  - seasonal-naive and ETS baselines, which always run and act as the floor;
  - LightGBM on lag, rolling-window and calendar features (the global model across all series);
  - direct multi-horizon forecasting.
- **Profiler:** detect datetime columns, infer frequency and gaps, and report the number of series and their lengths.
- **Prompt Agent:** extract the horizon and frequency from phrases like "forecast next 14 days of sales per store".
- **Leakage guard:** lag features must be built strictly from the past. Add a test that would catch a look-ahead bug.
- **Knowledge base:** add a forecasting entry to the local KB so retrieval-augmented planning has something to draw on.

### Acceptance criteria
- [ ] A sample dataset (synthetic multi-store daily sales with seasonality) runs end to end from the UI and the CLI.
- [ ] The winning model beats seasonal-naive on the test block, or the run reports clearly that it did not.
- [ ] Grounded verification works with suffix-window rungs.
- [ ] One public forecasting dataset is added to the evaluation fetcher, and results appear in `docs/research/results.md`.
- [ ] The comparison table's modalities row is updated, along with the user-guide docs on writing forecasting requests.

### Pointers
`schemas/task_spec.py`, `execution/model_registry.py`, `execution/templates/`, `tools/splits.py`, `tools/dataset_profiler.py`, `agents/prompt_agent.py`, `docs/development/adr/0004-template-grounded-codegen.md`

---

### Suggested order and dependencies
- **#3 (serving)** and **#4 (HITL)** are independent and touch different layers. Two people can start on them in parallel.
- **#5 (audit and model card)** puts its model card into #3's bundle. Build the audit first and the card later.
- **#6 (forecasting)** is the largest. It should land after #5's leakage checks, because temporal leakage is the main risk in forecasting.
