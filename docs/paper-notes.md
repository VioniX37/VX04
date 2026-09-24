# Paper notes: AutoML-Agent (arXiv:2410.02958, ICML 2025)

## Summary

- **Problem.** Earlier LLM-for-AutoML systems each automate one piece of the pipeline, such as HPO, model selection or feature engineering. AutoML-Agent covers the whole pipeline, from a natural-language request through data retrieval and preprocessing, model design and HPO, to a deployable model.
- **Agents.** Agent Manager, Prompt Agent (an instruction-tuned LLM that parses requests into JSON), Data Agent, Model Agent, and Operation Agent.
- **Retrieval-augmented planning (RAP).** The Manager retrieves external knowledge (web search, papers, model hubs) and generates several diverse plans instead of one. This widens exploration.
- **Plan decomposition + parallel pseudo-execution.** Each plan is split into data and model sub-tasks that specialist agents "execute" by reasoning, without running code. This makes it cheap to compare plans before paying for real execution.
- **Multi-stage verification.** Request verification (is the request clear and valid?), execution verification (do the pseudo-execution results satisfy the requirements, and which plan is best?), and implementation verification (does the deployed code actually meet the requirements?).
- **Evaluation.** 7 downstream tasks (image, text and tabular classification, tabular regression, graph node classification, time-series forecasting, and others), 14 datasets. Metrics are success rate (SR), normalized performance score (NPS), and a comprehensive score (CS) combining the two.

## What we implement (for now)

| Paper component | Ours | Notes |
|---|---|---|
| Prompt Agent (LoRA-tuned Mixtral) | Prompted general LLM + Pydantic `TaskSpec` + one self-repair round | We don't fine-tune. Schema validation does the enforcement. |
| RAP retrieval (web, arXiv, Kaggle, HF) | Local curated knowledge base with lexical ranking | `Retriever` protocol ready for web and HF-Hub retrievers |
| N plans | `N_PLANS` (default 3), diversity required by the prompt | |
| Plan decomposition (LLM) | Deterministic split from the structured `Plan` | Saves one LLM call per plan |
| Data/Model pseudo-execution | Same idea, in parallel with `asyncio.gather` | |
| Execution verification (LLM) | Deterministic ranking on predicted score, time budget, and speed | Could be made LLM-based later |
| Operation Agent | LLM edits a working template, then a debug loop, then falls back to the template | Templates make local/small LLMs viable |
| Implementation verification | Checks `metrics.json` against the metric target and time budget; revises with feedback | |
| Modalities | Tabular classification and regression, text classification | Image, graph and time series are future work |
| Deployment (Gradio app) | Not yet; the model is saved as `model.joblib` | |

## Evaluation plan

Use `backend/evaluation/run_benchmark.py` to reproduce the paper's metrics on our task set:

- **SR**: the fraction of runs that produce a working implementation (`success`).
- **Target-met rate**: the fraction of runs that satisfy the user's constraint.
- **Score**: the achieved metric, normalized against a simple baseline to get an NPS-style score.
- **Cost**: LLM calls and tokens per run.
