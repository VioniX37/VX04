# Colab and Kaggle

Large datasets are awkward to push through a browser. On notebook platforms, use the headless CLI and register data by **path** or **URL**.

## Kaggle notebook

```python
!git clone https://github.com/VioniX37/VX04.git
%cd VX04/backend
!pip install -q -e .
```

Add your Gemini key as a Kaggle secret named `GEMINI_API_KEY`, then:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
os.environ["WORKSPACE_DIR"] = "/kaggle/working/automl-workspace"
```

Register a competition or dataset file directly from `/kaggle/input` (no copy through the browser):

```python
!automl-agent ingest --data /kaggle/input/playground-series-s3e16/train.csv
!automl-agent run --dataset-id <id printed above> \
    --prompt "Predict the crab's Age from its measurements. Report RMSE."
```

## Google Colab

```python
!git clone https://github.com/VioniX37/VX04.git
%cd VX04/backend
!pip install -q -e .

from google.colab import userdata
import os
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

Files from Google Drive are registered by path after mounting:

```python
from google.colab import drive
drive.mount("/content/drive")
!automl-agent run --data /content/drive/MyDrive/data/transactions.parquet \
    --prompt "Predict whether a transaction is fraudulent. Optimise ROC AUC."
```

Or directly from a URL (the file is streamed to disk, then converted):

```python
!automl-agent run --data https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet \
    --prompt "Predict the fare amount of a taxi trip. Report RMSE."
```

## Tips for free-tier keys

- Keep `LLM_CACHE=true` (the default). Re-running an experiment replays cached answers and uses no quota.
- `--set AGENT_FUSION=true` halves the number of agent calls per plan.
- Lower `GEMINI_RPM_SMART` / `GEMINI_RPM_FAST` if you still see `429` errors. The client already retries with backoff.
- Set a budget, e.g. `--set BUDGET_WALL_S=1800 --set BUDGET_LLM_CALLS=60`, to cap a single run.

## Using the web UI on a remote machine

Run the API on the VM and expose port 8000 (for example through an SSH tunnel). Then set `NEXT_PUBLIC_API_URL` for a locally running UI, and add the UI origin to `CORS_ORIGINS`.
