# Pipeline stages

A run moves through ten stages. Each stage emits [events](../reference/event-schema.md) that the UI renders live and that are persisted to `events.jsonl`.

| # | Stage | What happens | Code |
|---|---|---|---|
| 1 | `parse` | Prompt Agent: request → `TaskSpec` | `agents/prompt_agent.py` |
| 2 | `verify_request` | Check columns, task/metric consistency and data size; one repair round; surface assumptions | `verification/request.py` |
| 3 | `prepare` | Ensure Parquet; create the fixed stratified train/valid/test split for the target | `tools/ingest.py`, `tools/splits.py` |
| 4 | `retrieve` | Retrieval-augmented planning: local knowledge base, Google Search grounding, experience memory | `planning/`, `memory/retriever.py` |
| 5 | `plan` | Manager proposes `N_PLANS` diverse plans (scale-filtered model families) | `AgentManager.generate_plans` |
| 6 | `execute_plans` | Decompose each plan; Data and Model agents (or the fused Plan Analyst) pseudo-execute in parallel | `planning/decomposition.py` |
| 7 | `ground` | *Grounded mode only.* Successive halving: surviving plans train on nested subsamples of growing size, scored on validation data | `verification/grounding.py` |
| 8 | `select` | Rank plans: on observed scores (grounded) or predicted scores (pseudo) | `verification/` |
| 9 | `implement` | Operation Agent writes, runs and debugs the full-data script; the score is computed on the **test** split | `agents/operation_agent.py`, `execution/` |
| 10 | `verify_impl` | Check test-split metrics against the user's constraints; revise (back to 5) if unmet | `verification/implementation.py` |

A final `done` event carries the outcome.

## Revisions

Stages 5–10 repeat up to `MAX_REVISIONS` extra times while a constraint is unmet. Each round's plans receive feedback listing the earlier plans, their predicted and observed scores, and the issues found. The best working implementation across rounds is kept. The loop also stops early when a [budget](#budgets) is exhausted; the run's `stop_reason` records why (`target_met`, `max_revisions`, `time_budget`, `llm_calls_budget`, `tokens_budget`).

## Budgets

| Setting | Meaning |
|---|---|
| `BUDGET_WALL_S` | Wall-clock seconds per run |
| `BUDGET_LLM_CALLS` | Gemini calls per run (cache hits are free) |
| `BUDGET_TOKENS` | Input + output tokens per run |

Budgets are soft limits checked between stages. Grounding also uses the remaining time to decide whether another rung fits. The planner sees the remaining budget in its context.

## Failure handling

| Failure | Behaviour |
|---|---|
| Invalid `TaskSpec` twice | Run fails at `verify_request` with the issues listed |
| Gemini `429`/`5xx` | Retried with exponential backoff, honouring the server's `retryDelay` |
| A model stays overloaded (`503`) or out of quota (`429`) | After 3 tries, falls back along `GEMINI_FALLBACK_MODELS`; usage records the model that answered |
| Google Search grounding quota exhausted | One attempt only; web search is paused for 10 minutes and planning continues with the other knowledge sources |
| Server restarts mid-run | The run is marked failed ("Interrupted") at the next startup and its event stream is closed; re-run it (cached answers are free) |
| Invalid JSON from the model | Error fed back for repair (up to 2 times); bad answers are evicted from the cache |
| A plan fails during grounding | It ranks last; if every plan fails, ranking falls back to the order reached |
| Script error | Operation Agent repairs it (with past fixes from memory); then template fallback |
| Timeout or memory ceiling | Process tree killed; reported in `stderr` with a hint |
| Unexpected exception | Run marked `failed` with the error; the server keeps running |
