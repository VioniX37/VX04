# Writing requests

The Prompt Agent turns your request into a structured [task specification](../reference/python-api.md#automl_agent.schemas.task_spec.TaskSpec). Clear requests give better specifications. Anything ambiguous is resolved with a stated assumption, which appears on the run page.

## What to include

| Element | Example | Effect |
|---|---|---|
| **What to predict** | "predict whether a customer will churn" | Sets the target column and task type |
| **The metric** | "optimise macro F1", "report RMSE" | Chooses the metric used for ranking and for the final score |
| **A target value** | "at least 85% accuracy", "RMSE below 2.2" | Becomes a constraint checked on the held-out test split; unmet targets trigger revisions |
| **A time budget** | "training must take under 60 seconds" | Recorded as `max_train_time_s` and checked by implementation verification |
| **Columns to avoid** | "ignore the customer id" | Added to `drop_columns` |
| **Domain** | "for a retail bank" | Helps knowledge retrieval |

## Supported tasks

| Task type | When it is chosen | Metrics |
|---|---|---|
| `tabular_classification` | The target is a label or category | `accuracy`, `f1_macro`, `f1_weighted`, `roc_auc`, `balanced_accuracy` |
| `tabular_regression` | The target is a continuous quantity | `rmse`, `mae`, `r2`, `mape` (RMSLE is also reported when the target is non-negative) |
| `text_classification` | The signal lives in a free-text column | Classification metrics |

## Examples

```text
Predict which customers will churn next month. The classes are imbalanced, so optimise macro F1.
```

```text
Estimate the sale price of a house from its features. RMSE must be below 30,000.
```

```text
Classify support tickets (column "body") into their category with at least 90% accuracy.
```

!!! tip "Check the assumptions"
    If the Prompt Agent guessed the wrong target or metric, the **Task specification** card shows it immediately. Rephrase the request and start a new run.
