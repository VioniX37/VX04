# Extension ideas (to decide as a team)

The core pipeline follows the paper. Our contribution should plug in through `automl_agent.extensions.PipelineHooks` and/or new UI, so we can compare against the baseline pipeline with the extension switched on and off.

| # | Idea | Hook / area | Effort | Why it's interesting |
|---|---|---|---|---|
| 1 | **Human-in-the-loop plan approval.** The run pauses after ranking; the user picks or edits a plan in the UI, then implementation continues. | `on_plans_ranked` waits on an `asyncio.Event`, plus a new `POST /api/runs/{id}/decision` and a UI panel | M | Uses the frontend well. We can measure how often users override the agent and whether that helps. |
| 2 | **Plan memory / experience reuse.** Store winning plans with dataset meta-features; retrieve similar past plans as extra knowledge. | `on_run_finished` stores; a new `Retriever` retrieves | M | Should raise the success rate and cut LLM calls on repeated task types. Easy to evaluate with the benchmark harness. |
| 3 | **Cost-aware planning.** A user budget (tokens, $, CPU-seconds) becomes a constraint; plans are ranked by expected score per cost. | `on_plans_ranked`, TaskSpec field, usage tracker | S–M | Reports a cost/accuracy Pareto front, which the paper doesn't analyze. |
| 4 | **Explainability report.** Auto-generate permutation importance / SHAP, a confusion matrix, and a model card after training. | `on_run_finished` + a new template section + UI tab | S | A visible, practical deliverable. |
| 5 | **Small-open-model study.** Measure SR and score for GPT-class vs local Ollama models, with and without templates. | Benchmark harness only | S | Empirical contribution: "can AutoML-Agent run on a laptop?" |
| 6 | **New modality: time series or images.** | New templates + model registry entries | M–L | Closer to the paper's breadth. |

We suggest **1 + 5**, or **2 + 5**: one system contribution plus one empirical study.
