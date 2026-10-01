# Quickstart

This walkthrough trains a churn model from the bundled sample data, first in the web UI and then from the command line.

## In the web UI

1. Start the API and the UI (see [Installation](installation.md)).
2. Open <http://localhost:3000>.
3. Under **1 · Dataset**, drop `data/samples/customer_churn.csv` onto the upload area. The dataset is converted to Parquet and profiled. You get a column table with types, missing values and a guessed target.
4. Under **2 · Describe the task**, pick the example *"Predict whether a customer will churn. The classes are imbalanced, so optimise macro F1."*
5. Click **Start AutoML run**.

The run page updates live:

- **Stepper.** Shows which of the ten stages is active.
- **Task specification.** What the Prompt Agent understood, including any assumptions it had to make.
- **Grounded verification.** Each candidate plan's *predicted* score next to its *observed* validation scores at each data size. Weak plans are eliminated.
- **Result.** The final model's metrics on the held-out **test** split, and whether your requirements were met.
- **Generated code.** The exact training script that produced the model.
- **Budget** and **Planning knowledge.** Time, LLM calls and tokens used, and which knowledge (including experience memory) informed the plans.

Artifacts (script, `metrics.json`, `model.joblib`) are saved under `backend/workspace/runs/<run id>/attempt_<n>/`.

## From the command line

The same pipeline runs headless, which is handy on servers and notebooks:

```bash
cd backend
automl-agent run --data ../data/samples/customer_churn.csv \
  --prompt "Predict whether a customer will churn. Optimise macro F1."
```

Output streams stage by stage and ends with a JSON summary:

```text
[       prepare] + manager: Split 1,050 train / 225 valid / 225 test rows (test rows are held out ...)
[        ground] + manager: Rung at all rows: r1p1=0.6802, r1p2=0.6580, r1p3=0.7020
[        select] + manager: Selected plan r1p3: ... (best observed validation score)
...
{
  "run_id": "8c855bb87153",
  "success": true,
  "metric": "f1_macro",
  "score": 0.702,
  ...
}
```

CLI runs are stored in the same database as the UI, so they also appear on the **Runs** page.

## Next steps

- Try a large dataset: [Colab and Kaggle](colab-kaggle.md) and [Large data](../concepts/large-data.md).
- Learn how to phrase requests and constraints: [Writing requests](../user-guide/writing-requests.md).
- Compare the paper-faithful mode with grounded verification: add `--set VERIFICATION_MODE=pseudo` to a run.
