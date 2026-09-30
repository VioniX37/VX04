# Reading results

## The run page

| Panel | What it tells you |
|---|---|
| **Configuration chips** | Verification mode, memory on/off, fused agents, the Gemini model per role |
| **Stepper** | The current stage; failed stages turn red |
| **Result** | Final metrics on the held-out **test** split, the model family, training time, and whether all requirements were met |
| **Grounded verification** | One row per candidate plan: the score the Model Agent *predicted*, then the *observed* validation score at each data size. The best score per column is green; plans dropped by successive halving show "eliminated" |
| **Candidate plans** | Rationale, preprocessing steps, data risks, predicted and observed scores |
| **Generated code** | The exact script that produced the model; copy it to reuse outside the platform |
| **Task specification** | Target, metric, target value, dropped columns, split sizes and the Prompt Agent's assumptions |
| **Budget** | Wall-clock time, LLM calls and tokens against the configured budgets, plus LLM cache hits |
| **Planning knowledge** | Knowledge used for planning: past runs recalled from experience memory, web sources (with links) and the local knowledge base |
| **Agent activity** | The full event stream; tick *Show raw LLM outputs* to inspect every structured answer |

## Success and requirements

A run **succeeds** when at least one implementation trained and produced a numeric score. The **requirements met** badge additionally means every stated constraint held on the test split, such as a metric target or a training-time limit. If a constraint is missed, the Manager revises its plans up to `MAX_REVISIONS` times and keeps the best result.

## Artifacts

```text
backend/workspace/runs/<run id>/
  events.jsonl                  # the full event log (replayed by the UI)
  ground_r<k>/<plan>_<rows>/    # grounding runs: train.py + metrics.json
  attempt_<k>/                  # final implementation of revision k
      train.py                  # the script
      metrics.json              # metric, score, all metrics, split, timings
      model.joblib              # fitted model (+ label classes and feature list)
```

Load a trained model:

```python
import joblib
bundle = joblib.load("backend/workspace/runs/<run id>/attempt_1/model.joblib")
model, classes = bundle["model"], bundle["classes"]
```

## Programmatic access

`GET /api/runs/{id}` returns the outcome. `GET /api/runs/{id}/observations` returns every predicted-versus-observed score, which is useful for your own analysis. See the [REST API reference](../reference/rest-api.md).
