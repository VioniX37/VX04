# Comparison with the paper and the official code

This project was written from the paper; no code from the official repository (CC BY-NC 4.0) was copied. The table compares behaviour, with pointers into our code.

| Aspect | Paper / official implementation | This project | Code |
|---|---|---|---|
| Orchestration | State machine INIT → PLAN → ACT → PRE_EXEC → EXEC → POST_EXEC → REV → END; `n_revise = 3` | Ten explicit stages; `MAX_REVISIONS` (default 2); budget-aware stopping | `agents/manager.py` |
| Backbone LLM | GPT-4o for all agents except the Prompt Agent (fine-tuned Mixtral via vLLM) | Gemini only; separate *smart* and *fast* models per role | `llm/` |
| Prompt Agent | LoRA fine-tuning on ~2.3k instruction pairs; six-part JSON | Prompted model with native JSON-schema output (`TaskSpec`), surfaced assumptions, one repair round | `agents/prompt_agent.py` |
| Request verification | LLM judges clarity; may ask the user | Deterministic checks against the dataset profile + repair round | `verification/request.py` |
| Retrieval (RAP) | arXiv, web search, Kaggle, HF, Papers with Code; static "past experience cases" | Curated local KB + Gemini Google Search grounding (cited URLs) + **experience memory learned from past runs** | `planning/`, `memory/` |
| Plans | $P = 3$, generated independently | `N_PLANS` (default 3) in one structured call; model families filtered by data size | `AgentManager.generate_plans` |
| Plan decomposition | LLM prompt | Deterministic from the structured plan (saves one call per plan) | `planning/decomposition.py` |
| Data / Model agents | Pseudo-execution; training-free HPO; predicted metrics | Same, run concurrently; optional fused single call (`AGENT_FUSION`) | `agents/data_agent.py`, `agents/model_agent.py`, `agents/plan_analyst.py` |
| Execution verification | LLM judges pass/fail on *predicted* results | **Successive halving with real runs on nested subsamples**, budget-aware; paper mode kept as `VERIFICATION_MODE=pseudo` | `verification/grounding.py` |
| Operation Agent | Skeleton code + LLM code generation + revision | LLM edits a working template; debug loop with **past fixes from memory**; template fallback | `agents/operation_agent.py` |
| Implementation verification | LLM judges the executed results | Checks `metrics.json` from the **held-out test split** against targets and time limits | `verification/implementation.py` |
| Data handling | In-memory, small benchmark datasets; 8×A100 | Parquet ingest, Polars profiling, fixed stratified splits, LightGBM/XGBoost, out-of-core text; 5M rows on a laptop CPU | `tools/`, `execution/` |
| Execution | Subprocess | Supervised subprocess with memory watchdog, timeout and live progress | `execution/sandbox.py` |
| Modalities | Image, text, tabular (classification/regression/clustering), time series, graph | Tabular classification/regression, text classification | `execution/model_registry.py` |
| Deployment | Deployment-ready code + inference endpoint | Trained model artifact (`model.joblib`); endpoint left as future work | — |
| Interface | Python API | Web UI with live event stream, REST API, CLI | `frontend/`, `api/`, `cli.py` |
| Reproducibility | — | Seeded splits, on-disk LLM response cache, per-run configuration and observation logs | `llm/cache.py`, `storage/db.py` |
| Evaluation | SR / NPS / CS over 14 datasets; AutoGluon, DS-Agent, GPT-4 zero-shot, SELA | Same metrics + calibration (RQ1) + cost; zero-shot Gemini and Optuna+LightGBM baselines; ablation matrix | `evaluation/` |

## What we deliberately did not reproduce

- **Fine-tuning the Prompt Agent.** A strong instruction-following model with a strict JSON schema makes fine-tuning unnecessary for our task set, and the free tier does not allow fine-tuning.
- **Image, graph and time-series tasks.** Scope was traded for scale. The template/registry design makes adding a modality a local change (see [ADR 0004](../development/adr/0004-template-grounded-codegen.md)).
- **LLM-judged verification.** Where a measurable check exists (columns exist, metrics met on test data), we use the measurement instead of asking the model.
