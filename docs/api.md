# Backend API

Base URL: `http://localhost:8000/api`. Interactive docs are at `http://localhost:8000/docs`.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/health` | | `{status, version, llm_provider, llm_model, codegen_mode}` |
| POST | `/datasets` | multipart `file` (.csv / .tsv, ≤200 MB) | `Dataset {id, filename, created_at, profile}` (201) |
| GET | `/datasets` | | `Dataset[]` |
| GET | `/datasets/{id}` | | `Dataset` |
| POST | `/runs` | `{dataset_id, prompt}` | `Run` (201); the pipeline starts in the background |
| GET | `/runs` | | `Run[]` |
| GET | `/runs/{id}` | | `Run {status, task_spec, plan, metrics, code, error, llm_usage, ...}` |
| GET | `/runs/{id}/events/history` | | `AgentEvent[]` |
| GET | `/runs/{id}/events` | `?after=<seq>` | **SSE** stream |

## SSE stream

Each event looks like this:

```
event: agent_event
id: 12
data: {"seq":12,"run_id":"…","ts":"…","stage":"select","agent":"manager","kind":"artifact","message":"Selected plan r1p1: …","payload":{…}}
```

When the run finishes, the stream sends `event: end` and closes. Events already sent are replayed, so a reconnect can pass `?after=<last seq>`.

Payloads the UI relies on:

| stage | payload key | content |
|---|---|---|
| `verify_request` | `task_spec` | the verified `TaskSpec` |
| `retrieve` | `knowledge` | `[{id, title, source}]` |
| `select` | `ranked`, `selected` | `PlanEvaluation[]`, selected plan id |
| `implement` | `code`, `result` | script being run / `ExecutionResult` |
| `verify_impl` | `metrics`, `issues` | `metrics.json` content, unmet requirements |
| `done` | `success`, `metrics`, `target_met` | final outcome |

TypeScript mirrors of every schema are in `frontend/src/lib/types.ts`. Keep them in sync with `backend/src/automl_agent/schemas/`.
