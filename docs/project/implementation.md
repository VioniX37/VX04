# Implementation guide

This page follows one run from request to served model, naming the module and function that does each step. The concept pages explain *why* each part works the way it does. This page explains *where* it is and *how* the parts connect. Paths are relative to `backend/src/automl_agent/` unless stated otherwise.

## Repository map

```text
backend/src/automl_agent/
  main.py, __main__.py, cli.py     FastAPI app factory, `python -m automl_agent`, CLI commands
  config.py                         Settings (every environment variable), workspace layout
  api/                              REST routers: datasets, runs, inference, health
  services/                         run_service (run lifecycle), dataset_service, event_bus (SSE)
  storage/db.py                     SQLModel tables and forward migrations
  schemas/                          Pydantic models: TaskSpec, Plan, events, runs, audit, model card
  llm/                              Gemini client, fake client, role router, rate limiter, cache
  agents/                           Manager, Prompt, Data, Model, Plan analyst, Operation agents; budgets
  prompts/                          System prompts per agent (Markdown)
  planning/                         Local knowledge base, Gemini Search retrieval, plan decomposition
  verification/                     Request, execution (pseudo ranking), grounding, implementation checks
  tools/                            Ingest, profiler, splits, data audit, heuristics
  execution/                        Model registry, renderer, sandbox, templates, inference, forecasting
  memory/                           Meta-features, experience store, retriever, hooks
  extensions/                       Hook interface, approval (human in the loop), model card
  knowledge/ml_practices.json       Curated planning knowledge
backend/evaluation/                 Benchmark harness, baselines, metrics, analysis, dataset fetcher
backend/tests/                      Offline test suite (fake LLM), 150 tests
frontend/src/                       Next.js 16 app: pages, components, run model, API client
docs/                               This site (MkDocs Material)
paper/                              LaTeX paper
```

## The life of a run

```mermaid
sequenceDiagram
    participant UI as Web UI / CLI
    participant API as api/runs.py
    participant RS as services/run_service.py
    participant M as agents/manager.py
    participant SB as execution/sandbox.py
    participant DB as storage (SQLite)
    UI->>API: POST /api/runs {dataset_id, prompt, approval}
    API->>RS: create_run()
    RS->>DB: RunRecord(status=pending)
    RS-->>UI: 201 run id (pipeline continues as an asyncio task)
    UI->>API: GET /api/runs/{id}/events (SSE)
    RS->>M: AgentManager(ctx).run()
    M->>M: parse → verify → prepare → audit → retrieve
    loop each revision
        M->>M: plan → analyse → ground
        M->>SB: grounding runs on subsamples
        M-->>UI: (optional) awaiting_input, plans for approval
        M->>SB: final run of the chosen plan
        M->>M: implementation verification
    end
    M->>M: hooks: model card, experience memory
    RS->>DB: status, metrics, code, usage, observations
    RS-->>UI: done event; UI shows results and "Use this model"
```

### 1. Starting the run

`api/runs.py:start_run` validates the dataset id and calls `services/run_service.py:create_run`. That function inserts a `RunRecord` and starts `execute_run` as an asyncio task. The task is registered in `_active_tasks`, so cancel can find it. `execute_run` builds a `RunContext` (`agents/context.py`) that holds:

- the run id, prompt and dataset path;
- the profile and settings;
- the LLM router;
- the event bus;
- the approval mode;
- a `state` dict that hooks use to share data (the audit report, human decisions).

Every step reports progress through `ctx.emit(stage, agent, message, kind, payload)`. `services/event_bus.py` appends each event to `runs/<id>/events.jsonl` and fans it out to SSE subscribers. Reconnecting clients replay the log first, which is how a page reload loses nothing.

### 2. Understanding the request

- **`AgentManager.parse_and_verify`.** The Prompt Agent (`agents/prompt_agent.py`, prompt in `prompts/prompt_agent.md`) turns the request into a `TaskSpec` (`schemas/task_spec.py`). The spec holds the task type, target, metric and target value, text column, time column, horizon, frequency, series ids, dropped columns and assumptions. Gemini's native JSON-schema output makes the spec well-typed by construction. With the fake LLM, `tools/heuristics.py:guess_task_spec` derives it from the profile.
- **`verification/request.py:verify_request`** checks the spec against the data profile. It checks that columns exist, that the target is numeric for regression and forecasting, that the number of classes is plausible, and that text and time columns are present. On failure the Prompt Agent gets one repair round with the listed issues.

### 3. Preparing the data

- **`AgentManager.prepare_data`.** `tools/ingest.py:ingest_to_parquet` converts CSV/TSV/JSON/JSONL/Parquet (optionally gzipped) into one Parquet file by streaming. Polars never loads the whole file: the 4.4 GB, 8.9M-row Malware CSV converts in about 18 s.
- **`tools/dataset_profiler.py:profile_lazy`** computes column kinds, null counts, exact or approximate unique counts and top values on a sample. For time series it also infers the datetime column, frequency, gaps and series.
- **`tools/splits.py:ensure_split`** writes a split file next to the data, with one row per data row:
  - `__split`: 0 = train, 1 = valid, 2 = test.
  - `__r`: a per-row draw used for subsampling.

  Classification splits are stratified. Forecasting splits are contiguous date blocks (`_assign_temporal`), so all series share the boundaries. For time series, `__r` ranks rows from the most recent, so low-fidelity rungs train on recent history. The split is cached by target and seed, so every plan and every run on a dataset uses identical rows.

### 4. Auditing the data

`AgentManager.audit_data` runs `tools/data_audit.py:run_data_audit` in a worker thread. The checks are:

- identifier-like discrete columns;
- column names that echo the target;
- constant columns, mostly-missing columns and high cardinality;
- class imbalance;
- single-feature predictive power (shuffled 80/20 holdout; a depth-2 tree for numbers, a per-category lookup for categoricals);
- exact train/test duplicates, found by hashing rows in Polars.

`apply_audit_to_task_spec` adds high-severity columns to `drop_columns` and records each drop as an assumption. Findings also become a knowledge item, so the planners are told about them. Set `DATA_AUDIT=false` to switch it off for ablations. See [Data audit and model card](../concepts/data-audit.md).

### 5. Knowledge and planning

- **`AgentManager.retrieve_knowledge`** gathers knowledge items from three sources:
  - the curated knowledge base (`planning/retrieval.py`, `knowledge/ml_practices.json`);
  - Gemini Google Search grounding with cited URLs (`planning/gemini_search.py`);
  - experience memory (`memory/retriever.py`): the k nearest past runs by meta-feature distance (`memory/meta_features.py`), with their plans, observed scores and fixes.
- **`AgentManager.generate_plans`** asks the Manager for `N_PLANS` diverse plans in one structured call. Model families are limited to those the registry allows at this data size (`execution/model_registry.py:supported_models`). On a revision, the history of earlier attempts and their verification issues goes back into the prompt.
- **`planning/decomposition.py:decompose`** splits each plan deterministically into data and model sub-tasks. The paper uses an LLM call per plan for this.
- **`AgentManager.evaluate_plan`** runs the Data and Model agents concurrently for each plan. With `AGENT_FUSION`, a single Plan analyst call does both. They return preprocessing steps, hyperparameters and a *predicted* score: the paper's pseudo-execution.

### 6. Selecting a plan

- **Paper mode** (`VERIFICATION_MODE=pseudo`): `verification/execution.py:rank_plans` sorts plans by predicted score.
- **Grounded mode** (the default): `AgentManager.ground` calls `verification/grounding.py`. Every surviving plan is rendered and *run* on a nested subsample, starting at `GROUNDING_MIN_ROWS` rows and growing by `GROUNDING_GROWTH` per rung. Each run is scored on a validation slice capped at `GROUNDING_VALID_ROWS`. Only the best 1/`GROUNDING_ETA` of plans continue to the next rung. Every prediction and observation is stored as a `PlanObservation`, which is the data behind the calibration study (RQ1). `agents/budget.py` enforces the wall-time, LLM-call and token budgets and can cut grounding short.
- **Hooks then run.** `on_plans_ranked` lets extensions reorder or edit the ranking. `AgentManager._validated_choice` maps the chosen model family to a supported one, in case a human edit named an unknown model.

### 7. Human in the loop (optional)

If the run was created with `approval="plans"` or `"plans+code"`, `extensions/approval.py:ApprovalHooks` is active:

1. `on_plans_ranked` writes `pause_state.json`, sets the run to `awaiting_input` and waits on an asyncio future.
2. `POST /runs/{id}/approve` (`run_service.approve_run`) resolves the future with an approve, pick or edit decision.
3. With `plans+code`, `on_code_generated` pauses again so the user can review the script.
4. If nobody answers within `APPROVAL_TIMEOUT_S`, the run continues with the top plan.
5. After a server restart there is no future. The decision is written into `pause_state.json`, the run is re-executed, and the hook replays the decision onto the plans saved in that file.
6. `POST /runs/{id}/cancel` kills the sandbox process tree (`sandbox.kill_run_processes`), cancels the task and keeps partial observations.

See [Human in the loop](../concepts/human-in-the-loop.md).

### 8. Implementation

`agents/operation_agent.py:OperationAgent.implement` turns the chosen plan into a script and runs it:

1. **`base_code`.** `execution/renderer.py:render_template` fills the task's template with a `CONFIG` dict: paths, target, family, hyperparameters, fidelity, eval split and seed. The templates are `templates/tabular.py`, `text_classification.py` and `time_series.py`.
2. **Optional LLM edit.** With `CODEGEN_MODE=llm`, the Operation Agent edits the working template rather than writing from scratch ([ADR 0004](../development/adr/0004-template-grounded-codegen.md)).
3. **Running.** `execution/sandbox.py:run_script` runs the script in a subprocess with a timeout, a memory watchdog (`EXEC_MAX_MEM_MB`) and process-tree kill. It streams `PROGRESS {json}` lines to the UI.
4. **Debugging.** On failure, the error signature and up to three past fixes from memory go to the LLM for a repair. After `MAX_DEBUG_ATTEMPTS`, the pristine template runs as a safety net.

Each template has the same contract:

- it trains on the train split and evaluates on the test split;
- it writes `metrics.json`, with the primary metric under `score` and the validation metrics under `metrics_valid`;
- it writes `model.joblib`, a dict bundle with the model and everything serving needs to reproduce the feature preparation.

### 9. Verification and revision

`verification/implementation.py:verify_implementation` checks four things:

- a numeric score exists;
- the metric target is met;
- the time limits are respected;
- for forecasting, the model beats seasonal-naive.

If it fails and revisions remain, the Manager re-plans with the issues as feedback. Attempts are compared on **validation** scores. The test score is reported only for the chosen attempt and never used for selection.

### 10. After the run

`on_run_finished` hooks run in order:

- **`ModelCardHook`** (`extensions/model_card.py`) builds `model_card.json` and `model_card.md`. It reloads the bundle and prepares up to 50k test rows *through the serving module*, then computes:
  - importances: native gain, permutation, or TF-IDF coefficients;
  - a confusion matrix and per-class metrics, calibration and Brier score for classifiers;
  - residuals for regression, or the backtest residuals for forecasting;
  - a narrative that cites only measured numbers.
- **`MemoryHooks`** (`memory/hooks.py`) stores an `ExperienceRecord`: meta-features, the winning plan, observed scores, error→fix pairs and whether a human overrode the choice.

`run_service.execute_run` then persists the status, metrics, code, LLM usage and observations, and emits the final event.

### 11. Serving

- **`execution/inference.py`** is a standalone module (NumPy, pandas, Polars, joblib) shared by the API, the CLI and the exported bundle:
  - `load_bundle` loads the model;
  - `validate_input` rejects missing or unexpected columns and replays the training dtypes and category levels (`prepare_features`);
  - `predict_dataframe` returns predictions, class probabilities, or forecasts with their dates;
  - `score_parquet_chunked` scores files 100k rows at a time.
- **`execution/forecasting.py`** holds the forecasting core used by the forecasting template, the API and the bundle: features, the LightGBM/seasonal-naive/ETS models as plain data, the rolling-origin backtest and history-based forecasting.
- **`api/inference.py`** exposes `GET /schema`, `POST /predict`, `POST /predict/batch` (streamed upload, unique temp directory) and `GET /artifacts/bundle`. The bundle zip contains:
  - `model.joblib`;
  - the two serving modules (as `automl_inference.py` and `automl_forecasting.py`);
  - `predict.py`;
  - `requirements.txt` pinned to the training versions;
  - `schema.json`, the metrics and the model card.
- **`cli.py:cmd_predict`** scores a file using the run's selected model.

See [Model serving](../concepts/model-serving.md) and [Time-series forecasting](../concepts/forecasting.md).

## Cross-cutting pieces

### LLM layer

`llm/factory.py:create_llm` returns an `LLMRouter` (`llm/router.py`) with a *smart* and a *fast* client. Agents choose a role through `model_role`. `llm/gemini.py` handles the provider side:

- structured output against Pydantic schemas;
- per-model RPM limits (`llm/rate_limit.py`) and a global concurrency cap;
- backoff that honours the server's `retryDelay`;
- fallback across `GEMINI_FALLBACK_MODELS` on overload or quota errors.

Every response is cached on disk (`llm/cache.py`), keyed by model, prompt and `LLM_CACHE_NAMESPACE`. This makes reruns free and reproducible. `llm/fake.py` is a deterministic offline backend used by every test. See [LLM layer](../concepts/llm-layer.md).

### Storage

`storage/db.py` defines four SQLite tables:

| Table | Holds |
|---|---|
| `DatasetRecord` | path, profile |
| `RunRecord` | status, approval mode and human decision, spec, plan, metrics, code, usage, config |
| `PlanObservation` | predicted versus observed scores per plan and fidelity |
| `ExperienceRecord` | the memory |

`init_db` creates the tables, and `_add_missing_columns` adds columns that were introduced later to existing databases. `run_service.recover_interrupted_runs` marks runs whose process died as failed. It leaves paused runs alone.

### Extension hooks

`extensions/__init__.py:PipelineHooks` defines `on_task_parsed`, `on_knowledge_retrieved`, `on_plans_generated`, `on_plans_ranked`, `on_code_generated` and `on_run_finished`. Approval, memory and the model card are all built on these hooks. A new extension subclasses `PipelineHooks` and calls `register_hooks(...)`.

### Configuration

`config.py:Settings` reads every option from the environment or `backend/.env`. `scripts/gen_config_reference.py` generates [the configuration reference](../reference/configuration.md), and CI fails if that page is out of date.

## Frontend

`frontend/src` is a Next.js 16 app (React 19, TypeScript, Tailwind v4):

- **`app/page.tsx`**: the landing page with the new-run form (`components/NewRunForm.tsx`). It supports upload with progress, register by path or URL, and picking an existing dataset, and you choose the approval mode there.
- **`app/runs/page.tsx`**: run history (`RunList.tsx`), with "Needs your input" badges.
- **`app/runs/[id]/page.tsx`** → `components/RunView.tsx`: the mission-control view. `lib/run-model.ts:buildRunModel` derives everything from the event stream:

  | View | Shows |
  |---|---|
  | `viz/PipelineGraph` | stage states |
  | `viz/AgentGantt` | timing |
  | `viz/GroundingChart` | predicted against observed scores |
  | `viz/BudgetGauges` | budgets |
  | `viz/LlmActivity` | LLM calls |
  | `viz/ResourceMonitor` | CPU and RAM telemetry |

  Feature panels add `ApprovalCard` (human in the loop), `AuditPanel`, `ModelCardPanel` and `UseModelPanel` (a schema-generated prediction form, batch scoring and bundle download).
- **`lib/api.ts`** is the typed REST client plus the SSE subscription. `lib/types.ts` mirrors the backend schemas.

## Evaluation harness

`backend/evaluation/run_benchmark.py` runs a config (`evaluation/configs/*.json`): datasets with ground-truth specs and prompts, variants (setting overrides) and seeds.

- Each variant gets its own workspace, so memory only accumulates within a variant.
- The LLM cache is shared, so identical calls are paid for once.
- It writes `results.csv` (SR, NPS, CS, wall time, LLM calls, tokens, memory hits, audit drops), `observations.csv` and `summary.json`.
- Baselines (`evaluation/baselines/`): Gemini zero-shot (one script, one call) and Optuna + LightGBM at an equal wall-clock budget.
- `evaluation/analysis.py` produces the paper's tables and figures, and `evaluation/forecasting_benchmark.py` scores the forecasting families.

See the [evaluation protocol](../research/evaluation-protocol.md).

## Testing and CI

The offline suite (`backend/tests`, run with the fake LLM) covers:

- schemas and tools;
- the sandbox;
- the LLM layer (retries, fallbacks, caching);
- pipelines end to end for every task type;
- grounding, memory and budgets;
- recovery after a restart;
- approval, cancellation and migrations;
- the audit and the model card;
- training/serving parity, the bundle running in isolation, and the CLI;
- forecasting: the leakage guard, splits, serving and a bundle without the package.

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, the configuration-reference check and `pytest`. It also runs the frontend's `npm run lint` and `npm run build`, and `mkdocs build --strict`.
