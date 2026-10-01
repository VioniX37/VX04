# Installation

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 or newer | Backend, CLI and experiments |
| Node.js | 20 or newer | Web UI only |
| Gemini API key | — | Free key from [Google AI Studio](https://aistudio.google.com/apikey); not needed for the offline `fake` backend |
| RAM | 8 GB+ | ~2 GB peak for a 5M-row tabular dataset |

## Backend

=== "Windows (PowerShell)"

    ```powershell
    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e ".[dev,eval]"
    ```

    If PowerShell refuses to run `Activate.ps1`, allow local scripts once:
    `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

=== "macOS / Linux"

    ```bash
    cd backend
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,eval]"
    ```

The extras are optional: `dev` (pytest, ruff), `eval` (Optuna, matplotlib and SciPy for the benchmark harness) and `docs` (this documentation site).

### Configure Gemini

Copy the example environment file and add your key:

```bash
cp ../.env.example .env
```

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
```

Check that the configured models are available to your key:

```bash
automl-agent models
```

!!! tip "No key yet?"
    Set `LLM_PROVIDER=fake` to run the whole pipeline offline with a deterministic heuristic backend. It is useful for trying the UI, for development and for tests.

### Start the API

```bash
automl-agent serve --reload
```

The API listens on <http://localhost:8000>. Interactive OpenAPI docs are at <http://localhost:8000/docs>.

!!! warning "Don't use plain `uvicorn --reload`"
    `--reload` without restrictions watches the whole `backend/` folder, including the workspace. When a run writes its training script, the server restarts and the run is killed. `automl-agent serve --reload` watches only the source code. The equivalent raw command is
    `python -m uvicorn automl_agent.main:app --reload --reload-dir src/automl_agent --app-dir src`.
    Runs interrupted by a restart are marked as failed at the next startup, and re-running them is cheap thanks to the LLM cache.

!!! note "`ModuleNotFoundError` when starting?"
    The virtual environment is not active, so a global `uvicorn` is being used. Activate the venv (your prompt shows `(.venv)`), or call the venv's interpreter directly: `.venv/Scripts/python -m uvicorn ...`.

## Web UI

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. If the API runs elsewhere, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

## Sample data

```bash
python data/samples/generate_samples.py            # three small CSVs (churn, house prices, reviews)
python data/samples/generate_large.py --rows 5000000   # 5M-row Parquet for scale tests (~370 MB)
```

## Documentation site

```bash
pip install -e "backend[docs]"
mkdocs serve        # from the repository root; opens on http://localhost:8000 (stop the API first, or use -a :8001)
```
