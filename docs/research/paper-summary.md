# The original paper

> Patara Trirat, Wonyong Jeong, Sung Ju Hwang. **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML.** ICML 2025. [arXiv:2410.02958](https://arxiv.org/abs/2410.02958) · [official code](https://github.com/DeepAuto-AI/automl-agent) (CC BY-NC 4.0)

## Problem

Earlier LLM-based AutoML systems automate one piece of the pipeline, such as hyperparameter optimisation, model selection or feature engineering. AutoML-Agent aims at the **full pipeline**: from a natural-language request, through data retrieval and preprocessing, model design and HPO, to a deployment-ready model, driven by a team of LLM agents.

## Method

| Component | Description |
|---|---|
| **Agent Manager** | Central coordinator; plans, delegates, verifies and revises |
| **Prompt Agent** | Mixtral-8x7B fine-tuned with LoRA on ~2.3k EvolInstruct pairs; parses requests into a JSON spec with six sections (user, problem, dataset, model, knowledge, service) |
| **Retrieval-augmented planning (RAP)** | The Manager retrieves knowledge (arXiv, web search, Kaggle, Hugging Face, Papers with Code, past experience cases) and generates $P = 3$ diverse end-to-end plans |
| **Plan decomposition** | Each plan is split into data sub-tasks and model sub-tasks |
| **Data Agent / Model Agent** | Specialists that *pseudo-execute* their sub-tasks by reasoning, without running code. The Model Agent performs training-free HPO and predicts performance for its top-$k$ ($k=3$) candidates |
| **Multi-stage verification** | *Request* verification (is the request clear?), *execution* verification (do the pseudo-execution results meet the requirements, and which plan is best?), *implementation* verification (does the executed code meet them?) |
| **Operation Agent** | Writes and runs code for the selected plan, starting from task-specific skeleton scripts |
| **Revision** | Failed verification triggers re-planning informed by the failure (up to 3 revisions in the official code) |

## Evaluation

- **Tasks.** 7 downstream task types over 14 datasets: image, text and tabular classification, tabular regression and clustering, time-series forecasting and graph node classification. Each dataset has a constraint-free and a constraint-aware instruction (28 instances).
- **Metrics.** Graded success rate (SR), normalized performance score $\text{NPS} = 1/(1+s)$ for loss $s$, and comprehensive score $\text{CS} = 0.5\,\text{SR} + 0.5\,\text{NPS}$.
- **Baselines.** Human models, AutoGluon, DS-Agent, GPT-3.5/GPT-4 zero-shot, SELA (MCTS).
- **Results.** Higher SR and CS than the baselines; about 525 s and \$0.30 per run with GPT-4o; about 8× faster than SELA at comparable performance.

## Stated limitations

1. Dependence on skeleton code for new task types (risk of hallucinated code).
2. Strong sensitivity to the backbone LLM: attempts with smaller open models failed (truncated code, hallucinations).
3. Limited scope: no reinforcement learning or recommender pipelines.
4. Literature retrieval could be improved (e.g. PaperQA).

Our extensions address limitation 2 directly: plan selection no longer depends on the LLM's predictions ([grounded verification](../concepts/grounded-verification.md)), and past outcomes supplement the LLM's knowledge ([experience memory](../concepts/experience-memory.md)). They also add something the paper does not evaluate: operation on **large datasets** under explicit budgets.
