# Architecture

The system has three deployable parts: a **FastAPI backend** that hosts the agent pipeline, a **Next.js web UI**, and the **`automl-agent` CLI** for headless use. They share one workspace directory, which holds a SQLite database plus data and run artifacts.

```mermaid
flowchart TB
    subgraph Clients
      UI[Next.js UI<br/>TypeScript · Tailwind]
      CLI[automl-agent CLI<br/>notebooks · servers]
      BENCH[Benchmark harness<br/>evaluation/]
    end
    subgraph Backend[FastAPI backend]
      API[REST + SSE<br/>api/]
      RS[Run service<br/>services/run_service.py]
      DS[Dataset service<br/>ingest · profile]
      BUS[Event bus<br/>services/event_bus.py]
      subgraph Pipeline[Agent pipeline]
        MGR[Manager]
        AG[Prompt · Data · Model ·<br/>Plan Analyst · Operation agents]
        VER[Verification<br/>request · grounding · implementation]
        MEM[Experience memory]
      end
      LLM[LLM router<br/>Gemini smart/fast · cache · rate limits]
      SBX[Sandbox<br/>time + memory limits]
    end
    subgraph Workspace[Workspace]
      DB[(SQLite<br/>datasets · runs · observations · experience)]
      FS[(Parquet data · splits ·<br/>scripts · models · event logs)]
    end
    UI --> API
    CLI --> RS
    BENCH --> RS
    API --> RS & DS
    RS --> MGR
    MGR --> AG & VER & MEM
    AG --> LLM
    VER --> SBX
    AG --> SBX
    MGR --> BUS --> API
    RS --> DB
    DS --> FS
    SBX --> FS
    MEM --> DB
```

## Backend packages

| Package | Responsibility |
|---|---|
| `agents/` | The Manager (orchestration, grounding, revisions) and the specialist agents; `RunContext`; budgets |
| `prompts/` | One role prompt per agent (Markdown with a header documenting inputs and output schema) |
| `llm/` | Gemini client, offline `FakeLLM`, role router, response cache, rate limiter |
| `planning/` | Local knowledge base retrieval, Google Search retrieval, plan decomposition |
| `verification/` | Request verification, pseudo ranking (paper), grounded successive halving, implementation verification |
| `memory/` | Experience memory: meta-features, store, retriever, pipeline hooks |
| `execution/` | Model registry, code templates, renderer, supervised sandbox |
| `tools/` | Ingest to Parquet, Polars profiling, fixed splits, heuristics |
| `schemas/` | Pydantic models shared by agents, API and UI |
| `services/` | Event bus, dataset registration, run execution |
| `storage/` | SQLModel tables and lightweight migrations |
| `api/` | REST and SSE endpoints |
| `extensions/` | `PipelineHooks`, the extension point memory is built on |

## Design principles

- **Agents reason, code executes.** LLM agents only produce structured JSON (validated with Pydantic). Everything that touches data runs as generated Python in the sandbox, against a fixed contract: read the Parquet + split files, write `metrics.json`.
- **Ground truth beats prediction.** Plans are selected on measured validation scores (see [Grounded verification](grounded-verification.md)). The final score is always computed on a test split that no agent has seen.
- **Every run is an experiment.** Configuration, predicted and observed scores, budgets and LLM usage are recorded per run, so the web UI and the benchmark harness read the same data.
- **Reproducible and quota-friendly.** Responses are cached on disk, splits are seeded, and every threshold is a setting.
- **Faithful baseline preserved.** `VERIFICATION_MODE=pseudo` and `MEMORY_ENABLED=false` reproduce the paper's behaviour for ablations.

## Request lifecycle

```mermaid
sequenceDiagram
  participant UI as Web UI
  participant API as FastAPI
  participant M as Manager
  participant A as Agents (Gemini)
  participant S as Sandbox
  participant DB as SQLite

  UI->>API: POST /api/datasets (or /register)
  API->>API: ingest to Parquet, profile
  API-->>UI: Dataset + profile
  UI->>API: POST /api/runs {dataset_id, prompt}
  API->>M: start in background
  UI->>API: GET /api/runs/{id}/events (SSE)
  M->>A: parse request -> TaskSpec
  M->>M: verify request, create split
  M->>A: retrieve knowledge (KB, web, memory), plan N candidates
  par each plan
    M->>A: Data + Model analysis (pseudo-execution)
  end
  loop successive halving rungs
    M->>S: train surviving plans on r rows, score on valid
  end
  M->>A: Operation Agent writes final script
  M->>S: full-data training, score on test
  M->>M: implementation verification (revise if needed)
  M->>DB: run outcome, observations, experience record
  API-->>UI: events streamed throughout
```

## Runtime layout

```text
backend/workspace/
  automl.db                       SQLite: datasets, runs, plan observations, experience records
  datasets/<id>/data.parquet      converted dataset
  datasets/<id>/splits/*.parquet  fixed train/valid/test assignment per target
  runs/<id>/events.jsonl          event log (replayed by the UI after restarts)
  runs/<id>/ground_r<k>/...       grounding runs
  runs/<id>/attempt_<k>/          final script, metrics.json, model.joblib
  llm_cache/<namespace>/          cached Gemini responses
```
