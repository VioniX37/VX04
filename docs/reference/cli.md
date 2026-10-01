# CLI

The `automl-agent` command is installed with the backend package (`pip install -e backend`). `python -m automl_agent` is equivalent.

```text
automl-agent [--set KEY=VALUE ...] {ingest,run,datasets,models} ...
```

`--set` overrides any [setting](configuration.md) for one command and can be repeated. It must come **before** the sub-command:

```bash
automl-agent --set VERIFICATION_MODE=pseudo --set N_PLANS=2 run --data data.csv --prompt "..."
```

## `ingest`

Register a dataset from a path or URL, convert it to Parquet and profile it.

| Option | Description |
|---|---|
| `--data PATH_OR_URL` | CSV/TSV/Parquet/JSONL file (optionally `.gz`), or an `http(s)://` URL |

Prints `{dataset_id, rows, cols, scale_tier, guessed_target}`.

## `run`

Run the full pipeline and stream stage events to the terminal.

| Option | Description |
|---|---|
| `--data PATH_OR_URL` | Register this file first, then run on it |
| `--dataset-id ID` | Use an already registered dataset |
| `--prompt TEXT` | The task in plain language (required) |

Ends with a JSON summary: `run_id`, `success`, `target_met`, `metric`, `score`, `model_family`, `artifacts`. The exit code is 0 on success.

## `datasets`

List registered datasets: id, name, rows, scale tier and source.

## `models`

List the Gemini models visible to the configured key and report whether `GEMINI_MODEL_SMART` and `GEMINI_MODEL_FAST` are available.

## Benchmark commands

The evaluation harness has its own entry points (run from `backend/`):

| Command | Purpose |
|---|---|
| `python -m evaluation.run_benchmark --config evaluation/configs/<name>.json` | Run an ablation matrix; see [Evaluation protocol](../research/evaluation-protocol.md) |
| `python -m evaluation.analysis <results dir> [--figures DIR] [--markdown FILE]` | Tables and figures |
| `python -m evaluation.datasets.fetch --list \| --group paper \| NAME ...` | Download benchmark datasets |
| `python scripts/gen_config_reference.py [--check]` | Regenerate (or check) the configuration reference |
