# Event schema

Every run produces an ordered stream of `AgentEvent`s:

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | 1-based position in the run's stream |
| `run_id` | string | Run id |
| `ts` | datetime (UTC) | Emission time |
| `stage` | enum | One of the [pipeline stages](../concepts/pipeline-stages.md), or `done` |
| `agent` | string | `manager`, `prompt_agent`, `data_agent`, `model_agent`, `plan_analyst`, `operation_agent`, `system` |
| `kind` | enum | `status` (progress), `info`, `llm` (raw structured output), `artifact` (a result), `warning`, `error` |
| `message` | string | Human-readable summary |
| `payload` | object \| null | Structured data (below) |

## Payloads the UI relies on

| Stage | Key | Content |
|---|---|---|
| any (`kind=llm`) | `output` | The agent's validated JSON answer |
| `verify_request` | `task_spec` | The verified `TaskSpec` |
| `prepare` | `split` | `{train, valid, test}` row counts |
| `retrieve` | `knowledge` | `[{id, title, source, urls}]`; `source` is `local-kb`, `google-search` or `memory:<run id>` |
| `ground` (status) | `schedule`, `n_train`, `predicted` | Rows per rung (`null` = all), training size, predicted score per plan id |
| `ground` (artifact) | `rung` | `{rows, results: [{plan_id, model_family, fidelity_rows, score, ok, duration_s, error}]}` |
| `select` | `ranked`, `selected`, `verification_mode` | `PlanEvaluation[]` (with observations), selected plan id |
| `implement` (status) | `attempt`, `code` | Script about to run |
| `implement` (info) | `attempt`, `progress` | A `PROGRESS` line printed by the script: `{stage, ...}` |
| `implement` (artifact/warning) | `attempt`, `result` | `ExecutionResult`: `ok`, `returncode`, `duration_s`, `stdout`, `stderr`, `metrics`, `timed_out`, `memory_exceeded`, `peak_memory_mb` |
| `verify_impl` | `issues`, `metrics`, `budget` | Unmet requirements, test metrics, budget snapshot |
| `done` | `success`, `metrics`, `target_met` | Final outcome |

## Persistence

Events are appended to `workspace/runs/<id>/events.jsonl` as they are published. After a server restart, finished runs are replayed from this file, so the UI can always rebuild a run page.
