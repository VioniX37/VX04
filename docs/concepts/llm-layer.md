# LLM layer

All model access goes through `automl_agent.llm`. Agents never call an SDK directly.

## Roles

| Role | Setting | Default | Used by |
|---|---|---|---|
| smart | `GEMINI_MODEL_SMART` | `gemini-3.8-flash` | Manager (planning), Operation Agent (code), web search |
| fast | `GEMINI_MODEL_FAST` | `gemini-3.1-flash-lite` | Prompt, Data and Model agents, Plan Analyst |

`LLMRouter` holds one client per role. The clients share a usage counter, so a run reports total calls, tokens and cache hits. Model ids are validated at startup against the ids visible to your key (`automl-agent models` lists them). Gemini releases new models regularly, so check the ids before long experiments.

## Structured output

Agents call `complete_json(messages, Schema)`. The Gemini client sends the Pydantic model's JSON schema as `response_json_schema`, so the model is constrained to valid output. If a model rejects a schema construct (HTTP 400), the client falls back to describing the schema in the prompt. Any output that fails validation is sent back with the error for repair, up to two times.

## Free-tier operation

| Mechanism | Setting | Purpose |
|---|---|---|
| Per-model sliding-window limiter | `GEMINI_RPM_SMART`, `GEMINI_RPM_FAST` | Stay under requests-per-minute quotas |
| Global concurrency cap | `GEMINI_MAX_CONCURRENCY` | Avoid bursts from parallel agents |
| Retries with backoff | `GEMINI_MAX_RETRIES` | Recover from `429`/`5xx`; honours the server's `retryDelay` |
| Model fallback chain | `GEMINI_FALLBACK_MODELS` | Another model answers when one is overloaded or out of quota |
| Overload patience | `GEMINI_OVERLOAD_WAIT_S` | Rides out demand spikes that affect every model in the chain |
| Response cache | `LLM_CACHE`, `LLM_CACHE_ROOT`, `LLM_CACHE_NAMESPACE` | Identical requests are free and reproducible |
| Fused analysis | `AGENT_FUSION` | One call per plan instead of two |

The limiter and semaphore are per process, because quota belongs to the API key rather than to a run.

!!! warning "One key per deployment"
    The client deliberately supports a single API key. Spreading requests over several keys to get past rate limits would break section 2(d) of the Google APIs Terms of Service, which says limits must not be circumvented. It would not help with `503` overloads anyway: those are capacity limits on Google's side, shared by every key. For more throughput, enable billing on the project (a paid tier) or use Vertex AI.

## Knowledge retrieval with Google Search

With `SEARCH_GROUNDING=true`, the smart model answers one question per run with the Google Search tool enabled: which approaches work best for this task, data size and metric? The answer and its cited URLs become a knowledge item (`source: google-search`), and the UI shows the links. This stands in for the paper's separate arXiv, Kaggle and Papers-with-Code integrations, using a single grounded call. Failures degrade gracefully to no web knowledge.

## Offline backend

`LLM_PROVIDER=fake` swaps in `FakeLLM`, a deterministic heuristic that answers every schema from the context block. It powers the test suite and CI, and lets you try the UI without a key.

## Vertex AI

Set `GEMINI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and authenticate with Application Default Credentials. The same client code is used.
