# GroundML

*Multi-agent AutoML that verifies plans on real data.* CSE311 Artificial Intelligence project, IIIT Kottayam.

[![CI](https://github.com/VioniX37/VX04/actions/workflows/ci.yml/badge.svg)](https://github.com/VioniX37/VX04/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![LLM](https://img.shields.io/badge/LLM-Gemini-4285F4)
[![License: proprietary](https://img.shields.io/badge/license-proprietary-red)](LICENSE)

**Multi-agent LLM AutoML that verifies its plans on real data, learns from experience and scales to millions of rows on a laptop.**

Upload a dataset (or point to one on disk or at a URL), describe the goal in plain language, and a team of Gemini agents does the rest:

1. They parse the request into a task specification.
2. They plan several candidate pipelines.
3. They test the candidates on growing samples of the data.
4. They write, run and debug the winner's training code.
5. They report the result on a held-out test split.

A web UI streams every step live.

This project re-implements, and extends:

> Patara Trirat, Wonyong Jeong, Sung Ju Hwang. **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML.** ICML 2025. [arXiv:2410.02958](https://arxiv.org/abs/2410.02958)

## What is new

| | The paper | This project |
|---|---|---|
| **Plan selection** | LLM *predicts* each plan's score; never checked | **Grounded verification**: successive halving with real runs on nested subsamples, under a time/token budget |
| **Knowledge** | Static retrieval | **Experience memory**: plans, observed scores and bug fixes from past runs on similar datasets |
| **Scale** | Small benchmark datasets, 8×A100 | **Large data** on one CPU machine: Parquet ingest, fixed splits, LightGBM/XGBoost, out-of-core text; tested end to end on 8.9M rows with a 16 GB laptop |
| **Backbone** | GPT-4o (+ fine-tuned Mixtral) | **Gemini free tier**: role-routed models, structured output, rate limiting, response cache, Google Search grounding |
| **Tasks** | Tabular, text, time series, image, graph | Tabular classification/regression, text classification and **time-series forecasting** (direct multi-horizon LightGBM, seasonal-naive and ETS baselines, rolling-origin backtests) |
| **Trust** | — | **Data audit** before training (target leakage, identifier columns, train/test duplicates) and a **model card** after it (importances, per-class metrics, calibration, residuals) |
| **Control** | Fully automatic | **Human in the loop**: approve, pick or edit a plan and review the generated script; cancel a run; paused runs survive restarts |
| **Deployment** | Inference endpoint | **Model serving** with the training feature preparation: REST predict and batch scoring, a prediction form in the UI, and a standalone bundle (`predict.py`, pinned requirements, schema, model card) |
| **Interface** | Python | Web UI with live event stream, REST API and CLI (Colab/Kaggle friendly) |
| **Evaluation** | SR / NPS / CS | The same metrics, plus calibration of LLM predictions, cost, baselines and an ablation harness |

See the [comparison with the paper](docs/research/comparison.md) and the [research proposal](docs/research/proposal.md).

## Quickstart

```bash
# Backend (Python 3.11+)
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,eval]"
cp ../.env.example .env              # set GEMINI_API_KEY (or LLM_PROVIDER=fake to try it offline)
automl-agent serve --reload           # API on http://localhost:8000

# Web UI (Node 20+), in a second terminal
cd frontend
npm install
npm run dev                          # http://localhost:3000
```

Or headless:

```bash
automl-agent run --data ../data/samples/customer_churn.csv --prompt "Predict churn. Optimise macro F1."
```

Full instructions: [Installation](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) · [Colab and Kaggle](docs/getting-started/colab-kaggle.md).

## How it works

```mermaid
flowchart LR
    U[Request + dataset] --> P[Prompt Agent] --> V{Request<br/>verification} --> PR[Prepare<br/>Parquet + splits]
    PR --> DA[Data audit<br/>leakage · quality] --> R[Retrieve<br/>KB · web · memory] --> PL[Manager<br/>N plans] --> A[Data + Model<br/>agents]
    A --> G[Grounded verification<br/>successive halving] --> H{Human approval<br/>optional} --> O[Operation Agent<br/>code · run · debug]
    O --> V3{Implementation<br/>verification}
    V3 -- revise --> PL
    V3 -- done --> MC[Model card] --> M[(Experience memory)]
    MC --> S[Serving<br/>API · UI · bundle]
```

## Results

Large-data study with the Gemini free tier on a laptop (Intel Core i5-1235U, 16 GB RAM, no GPU), one seed per configuration:

| Variant | Malware (8.9M rows): test ROC AUC | Conversion (5M rows): test ROC AUC | LLM requests |
|---|---|---|---|
| `paper` (LLM-predicted scores select the plan) | 0.7288 | 0.7772 | 8 / 6 |
| `grounded` (successive halving selects the plan) | **0.7302** | 0.7771 | 6 / 6 |
| `full` (grounded + experience memory) | 0.7293 | **0.7775** | 6 / 5 |
| Optuna + LightGBM (no LLM) | 0.7360 | 0.7784 | 0 / 0 |

- **LLM score predictions are over-optimistic.** All 12 predicted validation scores were above the measured ones, by +0.104 ROC AUC on average, and within a task they did not rank the plans (Spearman ρ ≈ 0).
- **Grounding corrects the choice cheaply.** It replaced the LLM's favourite plan (XGBoost) with the measured best (LightGBM) in two of the four runs that used it, at 6–30% of the compute of one full training run.
- **Measurements need guards.** The data audit removed a planted leak that otherwise scored a false 1.000 ROC AUC (honest score: 0.761), and a full-data check rejects LLM-edited scripts that train on a subsample.

Details: [Results](docs/research/results.md) · result set and provenance in [`backend/evaluation/results/local_large/`](backend/evaluation/results/local_large/) · [paper](paper/).

## Repository layout

```text
backend/                 Python package `automl_agent`, FastAPI app, CLI, tests, evaluation harness
  src/automl_agent/      agents · llm · planning · verification · memory · execution · tools · api
  evaluation/            benchmark runner, metrics, baselines, dataset fetcher, analysis
  tests/                 offline test suite (fake LLM)
frontend/                Next.js 16 · TypeScript · Tailwind v4 web UI
data/samples/            synthetic demo datasets and a large-data generator
docs/                    documentation site (MkDocs Material)
paper/                   LaTeX paper skeleton; figures generated by the evaluation harness
```

## Documentation

The documentation is built with MkDocs: `pip install -e "backend[docs]"`, then `mkdocs serve`. Highlights:

- Project: [Implementation guide](docs/project/implementation.md) · [Timeline and team](docs/project/timeline.md)
- Concepts: [Architecture](docs/concepts/architecture.md) · [Agents](docs/concepts/agents.md) · [Pipeline stages](docs/concepts/pipeline-stages.md) · [Grounded verification](docs/concepts/grounded-verification.md) · [Experience memory](docs/concepts/experience-memory.md) · [Large data](docs/concepts/large-data.md) · [LLM layer](docs/concepts/llm-layer.md) · [Data audit and model card](docs/concepts/data-audit.md) · [Human in the loop](docs/concepts/human-in-the-loop.md) · [Model serving](docs/concepts/model-serving.md) · [Time-series forecasting](docs/concepts/forecasting.md)
- Reference: [Configuration](docs/reference/configuration.md) · [REST API](docs/reference/rest-api.md) · [CLI](docs/reference/cli.md) · [Event schema](docs/reference/event-schema.md)
- Research: [Original paper](docs/research/paper-summary.md) · [Proposal](docs/research/proposal.md) · [Evaluation protocol](docs/research/evaluation-protocol.md)
- Development: [Contributing](docs/development/contributing.md) · [Testing](docs/development/testing.md) · [Decision records](docs/development/adr/index.md) · [Changelog](CHANGELOG.md)

## Development

```bash
cd backend && ruff check . && pytest            # offline suite
cd frontend && npm run lint && npm run build
mkdocs build --strict
```

> **Safety:** generated training code runs in a supervised subprocess with time and memory limits. That is process isolation, not a security sandbox. Run the system only on machines and data you trust.

## Team

Built at the **Indian Institute of Information Technology Kottayam** by:

| Member | Roll no. | Main contribution |
|---|---|---|
| Abhinandh A | 2024BCD0009 | Architecture, core pipeline, research extensions, integration, experiments |
| Jason Bobby | 2024BCD0013 | Human-in-the-loop run control |
| Pranav Hariharan | 2024BCD0005 | Time-series forecasting |
| Sajish Sajan Varghese | 2024BCS0001 | Data audit and model card |
| Aniketh S | 2024BCS0049 | Model serving |

How the project came together, phase by phase: [Timeline](docs/project/timeline.md).

## Relationship to the original code

This implementation was written from the paper. The authors' repository ([DeepAuto-AI/automl-agent](https://github.com/DeepAuto-AI/automl-agent), CC BY-NC 4.0) was consulted only as a reference for the paper's design; none of its code is included.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the original work:

```bibtex
@inproceedings{trirat2025automlagent,
  title     = {AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML},
  author    = {Trirat, Patara and Jeong, Wonyong and Hwang, Sung Ju},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

## License

Copyright (c) 2026 GroundML contributors. **All rights reserved.**
This is proprietary software: no permission is granted to use, copy, modify or distribute it without prior written consent. See [LICENSE](LICENSE).
