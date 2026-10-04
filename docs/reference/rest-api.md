# REST API

Base URL: `http://localhost:8000/api`. The live OpenAPI schema and a try-it-out console are served at `http://localhost:8000/docs`.

## Health

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status, version, llm_provider, llm_model, models: {smart, fast}, models_available, codegen_mode}` |

`models_available` maps each configured Gemini model id to whether the key can use it (checked at startup).

## Datasets

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/datasets` | multipart `file` (CSV/TSV/Parquet/JSONL, optionally `.gz`; ≤ `MAX_UPLOAD_MB`) | `Dataset` (201) |
| POST | `/datasets/register` | `{"path": "..."}` **or** `{"url": "https://..."}`, optional `"name"` | `Dataset` (201) |
| GET | `/datasets` | | `Dataset[]`, newest first |
| GET | `/datasets/{id}` | | `Dataset` |

```json
{
  "id": "3f2a9c1b7e44",
  "filename": "customer_churn.csv",
  "source": "upload",
  "created_at": "2026-10-01T10:00:00Z",
  "profile": {
    "n_rows": 1500, "n_cols": 9, "scale_tier": "small", "size_bytes": 81234,
    "memory_estimate_mb": 0.2, "approximate_counts": false,
    "guessed_target": "churn", "text_columns": [],
    "columns": [{"name": "churn", "dtype": "String", "kind": "categorical", "n_unique": 2,
                 "n_missing": 0, "sample_values": ["no", "yes"], "top_values": {"no": 0.61, "yes": 0.39}}]
  }
}
```

Errors: `400` for unreadable or unsupported files and bad paths/URLs, `413` for uploads that are too large, `422` when both or neither of `path`/`url` are given.

## Runs

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/runs` | `{"dataset_id": "...", "prompt": "..."}` | `Run` (201); the pipeline starts in the background |
| GET | `/runs` | | `Run[]`, newest first |
| GET | `/runs/{id}` | | `Run` |
| GET | `/runs/{id}/events` | `?after=<seq>` | **Server-Sent Events** stream (below) |
| GET | `/runs/{id}/events/history` | | `AgentEvent[]` |
| GET | `/runs/{id}/observations` | | Predicted-vs-observed rows (`PlanObservation[]`) |

`Run` fields: `id`, `dataset_id`, `prompt`, `status` (`pending`/`running`/`succeeded`/`failed`), timestamps, `task_spec`, `plan` (selected, with final hyperparameters), `metrics` (final `metrics.json` plus `target_met`, `attempts`, `stop_reason`, `budget`, `artifact_dir`), `code`, `error`, `llm_usage` (`calls`, `cache_hits`, token counts) and `config` (the experimental condition: models, verification mode, memory, grounding, budgets).

## Event stream

```text
event: agent_event
id: 12
data: {"seq":12,"run_id":"…","ts":"…","stage":"ground","agent":"manager","kind":"artifact",
       "message":"Rung at 20,000 rows: r1p1=0.7283, r1p2=0.7163, r1p3=0.7496","payload":{"rung":{…}}}
```

Events are replayed from the start (or from `after`), then streamed live. When the run ends, the server sends `event: end` and closes the stream. Payload shapes per stage are listed in the [event schema](event-schema.md).

TypeScript mirrors of all schemas are in `frontend/src/lib/types.ts`.

## Inference (model serving)

These endpoints are available for runs that have `status == "succeeded"`.

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/runs/{id}/predict` | JSON array of row objects | `PredictResult` |
| POST | `/runs/{id}/predict/batch` | multipart `file` (CSV/TSV/Parquet/JSONL) | scored Parquet (`application/octet-stream`) |
| GET | `/runs/{id}/artifacts/bundle` | | zip with model, script, requirements, schema, metrics |

### POST `/runs/{id}/predict`

Score one or more rows supplied as a JSON array. Returns predictions and, for classifiers, per-class probabilities.

```json
// Request
[{"age": 34, "income": 65000, "tenure": 2}]

// Response
{
  "run_id": "8c855bb87153",
  "n": 1,
  "predictions": ["yes"],
  "classes": ["no", "yes"],
  "probabilities": [[0.27, 0.73]]
}
```

### POST `/runs/{id}/predict/batch`

Upload a file to score. The scored result is returned as a Parquet file with a `prediction` column (and `prob_<class>` columns for classifiers). Reuses the Parquet ingest path, so multi-GB files are handled with bounded memory.

### GET `/runs/{id}/artifacts/bundle`

Download a zip with everything needed to run the model in a fresh Python environment:

| File | Contents |
|---|---|
| `model.joblib` | Trained model and preprocessing state |
| `predict.py` | Standalone CLI scoring script |
| `requirements.txt` | Pinned dependency versions |
| `schema.json` | Input columns, dtypes, task type, target, drop_columns |
| `metrics.json` | Test-split performance metrics |

The bundle runs without the AutoML-Agent package: `pip install -r requirements.txt && python predict.py --input new.csv --output scored.parquet`.

### Error codes

All three endpoints return `404` when the run is not found and `422` (never `500`) for:

- Run has not yet succeeded (`status` ≠ `succeeded`)
- No model saved (run used `save_model=false` or failed mid-training)
- Schema mismatch (the `422` body is `{"error": "…", "column": "<name>"}`)

