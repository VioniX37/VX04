# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Gemini-only LLM layer.** Role routing (smart/fast models), native JSON-schema output, a per-model rate limiter, a global concurrency cap, backoff honouring `retryDelay`, an on-disk response cache, startup validation of model ids, and Google Search grounding for knowledge retrieval. Optional fused Data+Model analysis (`AGENT_FUSION`).
- **Large-data pipeline.** Streaming ingest of CSV/TSV/Parquet/JSONL (including `.gz`) to Parquet; Polars profiling with scale tiers; fixed stratified train/valid/test splits with nested subsampling; LightGBM, XGBoost and HistGB templates with native categoricals and early stopping; an out-of-core hashing text model; a supervised sandbox with memory watchdog and live progress; registration by server path or URL; a 5 GB upload limit.
- **`automl-agent` CLI** (`ingest`, `run`, `datasets`, `models`) for headless use on Colab and Kaggle.
- **Grounded verification.** Successive halving over nested data subsamples replaces LLM-predicted plan ranking (paper behaviour kept as `VERIFICATION_MODE=pseudo`). Run budgets for wall time, LLM calls and tokens. Every prediction and observation is persisted (`PlanObservation`, `GET /api/runs/{id}/observations`).
- **Experience memory.** Past runs on similar datasets (meta-feature k-NN) are recalled as planning knowledge; error→fix pairs are reused while debugging; leave-one-dataset-out guard for evaluation.
- **Evaluation harness.** Config-driven ablation matrix; SR/NPS/CS as in the paper; calibration of pseudo-execution (Spearman ρ, top-1 hit, MAE); zero-shot Gemini and Optuna+LightGBM baselines; dataset fetcher; analysis script producing tables and figures.
- **Web UI.** Upload progress; register-by-path/URL and registered-dataset tabs; scale tier and size in the profile; grounded-verification table; budget meters; planning knowledge with memory recalls and web sources; run configuration chips; live training progress.
- **Documentation.** MkDocs Material site with getting-started guides, user guide, concepts, generated configuration and Python API references, research pages and decision records; Google-style docstrings enforced by Ruff; a LaTeX paper skeleton; CI workflow; contribution guide and templates.

### Changed

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
