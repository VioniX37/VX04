# Model serving

*Issue #3.* A run is only useful if its model can be used. Every successful run can be queried over REST, scored in bulk, used from the web UI and the CLI, and exported as a self-contained bundle.

## One code path for training and serving

The most common way a served model goes wrong is *training/serving skew*: the API prepares features slightly differently from the training script. Here, categories get different codes, a float becomes a string, or a column arrives in a different order. Predictions then look plausible and are quietly wrong. This project avoids that in two ways.

- **The training templates record their feature preparation in `model.joblib`.** The bundle stores:

  | Key | Holds |
  |---|---|
  | `features` | the feature list, in training order |
  | `dtypes` | the training dtype per column (`float32` for floats, `int8` for booleans, `category` for strings) |
  | `categories` | the category levels seen in training |
  | `ordinal_columns` | columns HistGradientBoosting encoded as integer codes (more than 250 levels) |
  | `drop_columns`, `ignored_columns` | columns that were excluded or present but unused |
  | `task_type`, `target`, `config` | the task type, the target and the full training configuration |

- **`execution/inference.py` replays it.**
  - `validate_input` drops the target, the excluded columns and the unused training columns. It rejects any other missing or unexpected column with a 422 that names the column.
  - `prepare_features` casts each column to its training dtype. Categories are encoded with the training levels, and unseen values become missing.
  - A non-numeric value in a numeric column is rejected with its column name. A numeric string from a web form is accepted.

The parity test (`tests/test_inference.py::test_endpoint_predictions_reproduce_training_test_score`) sends the whole held-out test split through `POST /predict` for a classification, a regression and a text dataset. It checks that the served predictions reproduce the score the training script reported, to 1e-6.

Text models get the list of texts, through the separate vectorizer for the hashing model. Forecasting models are served from each series' recent history; see [Time-series forecasting](forecasting.md).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/runs/{id}/schema` | Input columns with their kind (`numeric`, `category`, `text`) and category levels. The UI builds its prediction form from this |
| `POST /api/runs/{id}/predict` | Score a JSON array of records. Returns predictions, plus class probabilities for classifiers or forecast dates for forecasters |
| `POST /api/runs/{id}/predict/batch` | Upload CSV, TSV, Parquet or JSONL. The upload is streamed to a unique temporary directory, converted to Parquet, and scored 100k rows at a time. Returns Parquet in input row order |
| `GET /api/runs/{id}/artifacts/bundle` | Download the deployment bundle |

The model served is always the one the run **selected**. The run records that attempt's directory in `metrics.artifact_dir`, which isn't necessarily the latest attempt.

## Deployment bundle

| File | Contents |
|---|---|
| `model.joblib` | The trained model and its preparation state |
| `automl_inference.py` | The serving module itself, the same file the API uses |
| `automl_forecasting.py` | The forecasting core, for time-series models |
| `predict.py` | CLI: `python predict.py --input new.csv --output scored.parquet [--format csv]` |
| `requirements.txt` | Dependencies pinned (`==`) to the versions used in training, plus LightGBM or XGBoost when needed |
| `schema.json` | Same content as `GET /schema` |
| `metrics.json`, `model_card.md`, `model_card.json` | Test metrics and the model card |

The serving modules import only NumPy, pandas, Polars and joblib (plus LightGBM for forecasting). The bundle therefore runs in a fresh environment without the AutoML-Agent package, which two tests check: `test_bundle_scores_in_isolation` and `test_forecasting_bundle_loads_without_the_package`.

## CLI and UI

- `automl-agent predict --run <id> --data new.csv --out scored.parquet` scores a file with the run's selected model and refuses runs that did not succeed.
- The "Use this model" panel on a finished run has three parts:
  - a prediction form generated from the schema, with dropdowns for categories, a number field for numbers and a text area for text;
  - a result card showing the prediction, the class probabilities or the forecast date;
  - batch scoring with a download, and the bundle download.
