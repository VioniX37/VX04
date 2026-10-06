# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Time-series forecasting.** New task type `time_series_forecasting` with `time_column`, `horizon`, `frequency` and `series_id_columns` in the task spec, extracted from requests such as "forecast the next 14 days of sales per store"; profiler detection of datetime columns, frequency, gaps and series; date-aligned temporal splits (every series shares the block boundaries) with suffix-window grounding rungs; a shared forecasting module (`execution/forecasting.py`) with a global direct multi-horizon LightGBM model and seasonal-naive and ETS baselines, all scored with a rolling-origin backtest; forecasts served from each series' recent history through the same REST endpoints and bundle; `store_sales` sample, `air_passengers` in the dataset fetcher and `evaluation.forecasting_benchmark`.
- **Human-in-the-loop run control.** Approval modes per run (`auto`, `plans`, `plans+code`): the run pauses as `awaiting_input` after grounded verification so the user can approve, pick or edit a plan, and optionally review the generated script; unanswered approvals continue with the top-ranked plan after `APPROVAL_TIMEOUT_S` (one hour by default). `POST /runs/{id}/cancel` kills the sandbox process tree and keeps partial observations (`cancelled` status). Paused runs survive a backend restart and resume onto the plans the user reviewed. Human overrides are recorded in experience memory.
- **Pre-training data audit.** Deterministic target leakage and quality checks using Polars (single-feature predictive power, identifier detection, target name matching, and high-performance train/test duplicate row detection scaling to 5M rows in seconds). High-severity findings automatically populate `drop_columns`, surface assumptions, and inform planning knowledge. Toggleable via `DATA_AUDIT`.
- **Post-training model card.** Automated diagnostic model card (`model_card.json` and `model_card.md`) generated via the `on_run_finished` hook. Includes native gain and permutation feature importances, confusion matrix and per-class metrics, residual analysis for regression, probability calibration curves with Brier score, and a measured-evidence "Why this model" narrative.
- **Web UI audit and model card panels.** Pre-training audit panel displaying leakage alerts and auto-dropped features; post-training model card view with interactive feature importance bars, diagnostic matrices, and reliability tables.
- **Planted leaky benchmark sample.** Synthetic dataset (`data/samples/customer_churn_leaky.csv`) for leakage detection validation and benchmark ablation tracking.
- **Gemini-only LLM layer.** Role routing (smart/fast models), native JSON-schema output, a per-model rate limiter, a global concurrency cap, backoff honouring `retryDelay`, an on-disk response cache, startup validation of model ids, and Google Search grounding for knowledge retrieval. Optional fused Data+Model analysis (`AGENT_FUSION`).
- **Large-data pipeline.** Streaming ingest of CSV/TSV/Parquet/JSONL (including `.gz`) to Parquet; Polars profiling with scale tiers; fixed stratified train/valid/test splits with nested subsampling; LightGBM, XGBoost and HistGB templates with native categoricals and early stopping; an out-of-core hashing text model; a supervised sandbox with memory watchdog and live progress; registration by server path or URL; a 5 GB upload limit.
- **`automl-agent` CLI** (`ingest`, `run`, `datasets`, `models`) for headless use on Colab and Kaggle.
- **Grounded verification.** Successive halving over nested data subsamples replaces LLM-predicted plan ranking (paper behaviour kept as `VERIFICATION_MODE=pseudo`). Run budgets for wall time, LLM calls and tokens. Every prediction and observation is persisted (`PlanObservation`, `GET /api/runs/{id}/observations`).
- **Experience memory.** Past runs on similar datasets (meta-feature k-NN) are recalled as planning knowledge; error→fix pairs are reused while debugging; leave-one-dataset-out guard for evaluation.
- **Evaluation harness.** Config-driven ablation matrix; SR/NPS/CS as in the paper; calibration of pseudo-execution (Spearman ρ, top-1 hit, MAE); zero-shot Gemini and Optuna+LightGBM baselines; dataset fetcher; analysis script producing tables and figures.
- **Web UI.** Upload progress; register-by-path/URL and registered-dataset tabs; scale tier and size in the profile; grounded-verification table; budget meters; planning knowledge with memory recalls and web sources; run configuration chips; live training progress.
- **Inference endpoints & deployment bundle (paper parity).** A serving module (`execution/inference.py`) that replays the training dtypes and category levels recorded in `model.joblib`, so served predictions reproduce the training script's test score; REST endpoints `GET /runs/{id}/schema`, `POST /runs/{id}/predict`, `POST /runs/{id}/predict/batch` and `GET /runs/{id}/artifacts/bundle`; a deployment zip (`model.joblib`, `automl_inference.py`, `predict.py`, `requirements.txt` pinned with `==`, `schema.json`, `metrics.json`, model card) that runs in a clean Python environment; CLI `automl-agent predict`; and a "Use this model" panel with a schema-generated prediction form, batch file scoring and bundle download.
- **Documentation.** MkDocs Material site with getting-started guides, user guide, concepts, generated configuration and Python API references, research pages and decision records; Google-style docstrings enforced by Ruff; a LaTeX paper skeleton; CI workflow; contribution guide and templates.

### Changed

- The project is proprietary: all rights reserved to the contributors (see `LICENSE`).
- The default LLM provider is now `gemini`; `fake` remains for offline use.
- Final scores are computed on a held-out test split that is never used for selection.
- Default execution timeout raised to 30 minutes for full-data training.

### Removed

- OpenAI, Anthropic and OpenAI-compatible adapters.

## [0.1.0] - 2026-09-24

### Added

- Initial implementation of the AutoML-Agent pipeline: Prompt, Manager, Data, Model and Operation agents; request/execution/implementation verification; retrieval-augmented planning over a local knowledge base; template-grounded code generation with a debug loop.
- FastAPI backend with SSE event streaming, SQLite storage and a Next.js web UI.
- Offline fake LLM, test suite and sample datasets.
