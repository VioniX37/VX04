# Testing

The whole test suite runs **offline**. `LLM_PROVIDER=fake` swaps Gemini for a deterministic heuristic backend, and every test uses a temporary workspace.

```bash
cd backend
pytest                      # full suite
pytest tests/test_grounding.py -q
pytest -k memory
```

## Layout

| File | Covers |
|---|---|
| `test_llm.py` | JSON extraction and repair, response cache (hits, eviction of invalid answers), rate limiter, role routing, Gemini retries on 429 honouring `retryDelay`, non-retryable errors, schema fallback |
| `test_data_pipeline.py` | Ingest of CSV/TSV/Parquet/JSONL/gzip, late-schema recovery, profiling, scale tiers, stratified/nested/reused splits |
| `test_sandbox.py` | Script contract, failures, timeouts, memory ceiling, live progress streaming |
| `test_schemas_and_tools.py` | TaskSpec normalisation, heuristic parsing, request verification |
| `test_grounding.py` | Fidelity schedules, successive halving follows observed not predicted scores, lower-is-better metrics, failures, budget stop, budget tracker |
| `test_memory.py` | Meta-features and fingerprints, storage and recall, task-type isolation, leave-one-out guard, disabled memory, error signatures |
| `test_pipeline_e2e.py` | End-to-end runs for all task types, paper-faithful vs grounded modes, fusion call savings, budget-stopped revisions |
| `test_api.py` | Health, upload, registration by path, run lifecycle, SSE stream, observations endpoint |
| `test_cli.py` | `ingest`, `run` and `datasets` commands |
| `test_evaluation.py` | SR/NPS/CS, calibration, a full benchmark smoke run including both baselines and the analysis figures |
| `test_docs.py` | The configuration reference matches `Settings` |

## Scale test

This is not part of the default suite, because it takes about 2 minutes and 2 GB of RAM:

```bash
python ../data/samples/generate_large.py --rows 5000000
LLM_PROVIDER=fake automl-agent --set CODEGEN_MODE=template run \
  --data ../data/samples/large_conversion.parquet --prompt "Predict whether the customer converted. Optimise ROC AUC."
```

Check the wall time in the output and `peak_memory_mb` in the run's `events.jsonl`.

## Testing against Gemini

Use a separate workspace so experiments stay out of your main database:

```bash
WORKSPACE_DIR=/tmp/gemini-check automl-agent run --data ../data/samples/customer_churn.csv --prompt "Predict churn"
```

Responses are cached, so repeating the command costs no quota.

## Writing tests

- Use the `settings` fixture (temporary workspace, fake LLM, no revisions) and `sample_csvs` (small generated datasets).
- When adding an agent or schema, add the matching `_answer_<Schema>` handler to `llm/fake.py`.
- Do not call the network in tests; inject fake SDK clients as `test_llm.py` does.
