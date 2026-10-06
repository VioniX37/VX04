# Time-series forecasting

*Issue #6.* Forecasting is the paper's task type that best fits this project's strengths: tabular data, large volumes, and CPU-friendly models. The guiding rule is that **nothing from the future may reach a forecast**, in training, in evaluation or in serving.

## Task specification

`TaskType.time_series_forecasting` adds these fields to the `TaskSpec`:

| Field | Meaning |
|---|---|
| `time_column` | The timestamp column (detected by the profiler if not given) |
| `horizon` | How many steps ahead to forecast |
| `frequency` | Pandas frequency (`D`, `H`, `W`, `MS`, ...), inferred from the data |
| `series_id_columns` | Columns that identify individual series in panel data (for example `store_id`) |

The metrics are `smape` (the default), `mae`, `rmse` and `mape`. The Prompt Agent extracts all of these from requests such as *"Forecast the next 14 days of sales per store. Optimise sMAPE."* The profiler (`tools/dataset_profiler.py`) detects datetime columns, the frequency, gaps, the number of series and their lengths.

## Temporal split

Rows are never shuffled. `tools/splits.py:_assign_temporal` cuts the **distinct timestamps** into contiguous train (70%), validation (15%) and test (15%) blocks. Every series therefore shares the same boundary dates, and no date falls into two blocks. For grounding, `__r` is the share of more recent rows in the block. A fidelity of *r* rows keeps the most recent whole dates (*suffix windows*), so low-fidelity rungs train on recent history rather than on random rows.

## Models

All forecasting logic lives in `execution/forecasting.py`. The training template, the API and the exported bundle share this one module. Models are stored as plain data, so a bundle unpickles anywhere.

| Family | Model |
|---|---|
| `lightgbm` | One global LightGBM regressor across all series and all steps (*direct* multi-horizon forecasting): it predicts `y(t+h)` from features known at the origin `t` |
| `seasonal_naive` | The latest value from the same seasonal phase |
| `ets` | Additive Holt-Winters (level, trend, seasonality) |

The LightGBM features at origin `t` for step `h` are:

- lags `y(t)`, `y(t-1)`, `y(t-6)` and `y(t-13)`;
- rolling means and standard deviations over the last 7 and 14 values;
- the seasonal lag, the latest value at or before `t` in the same seasonal phase as `t+h`;
- calendar features of the target date `t+h` (day of week, month, day, weekend);
- the step `h`;
- the series code.

Every value feature uses only data at or before `t`. `test_leakage_guard_strictly_past_features` changes every value after an index and checks that no feature of an earlier origin changes.

The season length follows the frequency: 7 for daily, 24 for hourly, 52 for weekly, 12 for monthly, 60 for minutes.

## Evaluation: rolling-origin backtest

A forecast is only meaningful from a specific origin. `forecasting.backtest` walks the evaluation block in windows of `horizon` steps. Each window is forecast from **all actual values before it**: the history plus the evaluation block so far. No forecast is made more than `horizon` steps ahead. The seasonal-naive and ETS baselines are always scored the same way, on the same block, as the floor. `metrics.json` reports:

- the model's scores;
- `baseline_scores`;
- `beats_seasonal_naive` (implementation verification flags a model that does not beat the floor);
- `metrics_valid`, the validation backtest used to compare revision attempts.

## Serving

`forecasting.serving_state` attaches each series' most recent history to the model: the last values and the last date. `POST /predict` takes the series id columns and `horizon_step` (1 to the horizon). It returns the forecast for that many steps past the end of the data, plus its `forecast_dates`. An unknown series or a step outside the horizon returns 422.

```json
[{"store_id": "store_3", "horizon_step": 1}, {"store_id": "store_3", "horizon_step": 7}]
```

The test `test_time_series_pipeline_end_to_end` checks that served forecasts follow each series' recent level and are not flat.

## Results

On the `store_sales` sample (5 stores × 180 days), LightGBM reaches 8.57% sMAPE, against 8.59% for ETS and 9.79% for seasonal-naive, on the held-out test block. Reproduce with `python -m evaluation.forecasting_benchmark`; see [Results](../research/results.md).
