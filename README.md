# AutoML-Agent: a re-implementation with extensions

Semester project: our own implementation of

> Patara Trirat, Wonyong Jeong, Sung Ju Hwang. **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML.** ICML 2025. [arXiv:2410.02958](https://arxiv.org/abs/2410.02958)

It adds a web UI and a hook system for our own extension (see [docs/extension-ideas.md](docs/extension-ideas.md)).

A user uploads a dataset and describes the goal in plain language. A team of LLM agents then handles the rest:

1. They parse the request into a structured task spec and verify it.
2. They retrieve ML knowledge and plan several candidate pipelines.
3. Data and Model agents evaluate the plans in parallel, and the best one is selected.
4. The Operation agent writes the training code, runs it, and debugs it.
5. The Manager checks the result against the user's requirements, and revises if needed.

The UI streams each step live.

```
┌──────────────┐  REST + SSE  ┌─────────────────────── FastAPI backend ───────────────────────┐
│  Next.js UI  │ ───────────▶ │ Manager ─▶ Prompt Agent ─▶ request verification               │
│  (TS + TW)   │ ◀─────────── │    │       knowledge retrieval ─▶ N plans                      │
└──────────────┘  live events │    ├──▶ Data Agent ┐  (parallel, per plan)                     │
                              │    ├──▶ Model Agent┘ ─▶ execution verification (rank)          │
                              │    └──▶ Operation Agent ─▶ sandboxed run ─▶ impl. verification │
                              │  LLM layer: OpenAI · Anthropic · Gemini · Groq · Ollama · Fake │
                              └────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
backend/                 Python 3.11+ · FastAPI · scikit-learn
  src/automl_agent/
    agents/              Manager, Prompt, Data, Model, Operation agents + RunContext
    prompts/             role prompts for each agent (markdown)
    planning/            knowledge retrieval (RAP) + plan decomposition
    knowledge/           curated local ML knowledge base (json)
    verification/        request / execution / implementation verification
    execution/           code templates, model registry, subprocess sandbox
    llm/                 provider-agnostic client + adapters + offline FakeLLM
    tools/               dataset profiler, heuristic parser
    schemas/             Pydantic models (TaskSpec, Plan, AgentEvent, ...)
    services/            event bus (SSE) + background run service
    storage/             SQLite (SQLModel) for datasets and runs
    api/                 REST routes: /api/health, /api/datasets, /api/runs
    extensions/          PipelineHooks: where our extension plugs in
  tests/                 unit + end-to-end tests (offline, FakeLLM)
  evaluation/            benchmark harness (success rate, score, tokens)
frontend/                Next.js 16 (App Router) · TypeScript · Tailwind v4
  src/app/               /  (new run) · /runs · /runs/[id] (live view)
  src/components/        RunView, AgentTimeline, PlanCards, CodeViewer, ...
  src/lib/               typed API client, TS mirrors of backend schemas
data/samples/            synthetic demo datasets + generator
docs/                    architecture, paper notes, API, extension ideas
```

## Quick start

Prerequisites: Python 3.11+ and Node 20+.

```bash
# backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # cmd: .venv\Scripts\activate.bat · macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp ../.env.example .env            # LLM_PROVIDER=fake works offline; set a real provider + key later
python ../data/samples/generate_samples.py
python -m uvicorn automl_agent.main:app --reload --app-dir src     # http://localhost:8000/docs

# frontend (second terminal)
cd frontend
npm install
npm run dev                                              # http://localhost:3000
```

> **`ModuleNotFoundError` (e.g. `sqlmodel`)?** The venv isn't active and a global `uvicorn` is being used. Activate it (your prompt should show `(.venv)`), or run `.\.venv\Scripts\python.exe -m uvicorn ...` directly. If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Open http://localhost:3000, upload `data/samples/customer_churn.csv`, pick an example prompt, and start a run.

### Choosing an LLM

Set these in `backend/.env`:

| Provider | `LLM_PROVIDER` | Needs |
|---|---|---|
| Offline heuristic (tests/demo) | `fake` | nothing |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Local Ollama | `ollama` | Ollama running, e.g. `ollama pull qwen2.5-coder:7b` |
| vLLM / any OpenAI-compatible server | `openai_compatible` | `LLM_BASE_URL`, `LLM_MODEL` |

`LLM_MODEL` overrides the default model (defaults are in `backend/src/automl_agent/llm/factory.py`).

## Development

```bash
cd backend && pytest && ruff check .                  # 19 tests, fully offline
python evaluation/run_benchmark.py --provider fake     # benchmark → evaluation_results/
cd frontend && npm run lint && npm run build
```

> **Safety:** the Operation Agent runs LLM-generated Python in a subprocess, with a timeout and a separate working directory. That is not a security sandbox, so run it only on your own machine or in a container.

## Relationship to the original code

The authors' repository ([DeepAuto-AI/automl-agent](https://github.com/DeepAuto-AI/automl-agent), CC BY-NC 4.0) was used only as a reference for the paper's design. None of its code is copied here. Differences are listed in [docs/paper-notes.md](docs/paper-notes.md).

## Citation

```bibtex
@inproceedings{trirat2025automlagent,
  title     = {AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML},
  author    = {Trirat, Patara and Jeong, Wonyong and Hwang, Sung Ju},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
