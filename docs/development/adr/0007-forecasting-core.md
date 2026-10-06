# ADR 0007: A shared forecasting core with rolling-origin evaluation

- Status: Accepted
- Date: 2026-10-06

## Context

The first forecasting implementation had three problems:

- **Wrapped-horizon evaluation.** A test block longer than the horizon was scored as one forecast from the end of history, with the step counter wrapping back to 1 every `horizon` rows. Later rows became "1-step" forecasts made from weeks-old lags, which inflated the scores (7.95% vs 13.89% sMAPE reported; 8.57% vs 9.79% measured correctly).
- **No history at serving time.** The saved model held no history, so serving filled every lag with 0 and returned flat, wrong forecasts.
- **Script-local classes.** Models were pickled as classes defined in the training script's `__main__`, so loading them needed regex-scraping `train.py` and injecting classes from the package. The standalone bundle could not do either.

## Decision

- All forecasting logic lives in one module (`execution/forecasting.py`): features, models, backtest, serving state and forecasting. The training template, the API and the bundle (`automl_forecasting.py`) all use it.
- Models are plain data (dicts, arrays, a fitted `LGBMRegressor`), never script-local classes.
- Every family, the baselines included, is evaluated with a **rolling-origin backtest**: one forecast of up to `horizon` steps per window, made from all actual values before it.
- The serving state stores each series' recent history and last date. Requests ask for `(series, horizon_step)` and get forecasts past the end of the data, with their dates.
- Temporal splits cut on timestamps, so all series share the block boundaries.

## Alternatives considered

- *Evaluate only the first `horizon` steps of the test block.* This is simpler, but it wastes most of the block and gives noisy estimates.
- *Recursive one-step forecasting.* This needs only one model, but errors compound. The direct global model is the common strong baseline for panels.

## Consequences

- Reported scores are comparable to standard forecasting practice, and the baselines are scored on equal terms.
- Forecasts are consistent between evaluation and serving, because both run the same code.
- Forecasts beyond the trained horizon are rejected rather than extrapolated.
