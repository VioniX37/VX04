# automl-agent (backend)

Python backend of **GroundML**: the multi-agent pipeline, the FastAPI server, the
`automl-agent` CLI and the evaluation harness.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
automl-agent serve --reload
```

See the repository [README](../README.md) and the documentation site (`mkdocs serve` from the
repository root) for installation, configuration and usage.
