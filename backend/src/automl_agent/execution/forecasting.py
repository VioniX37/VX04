"""Forecasting core shared by the time-series training template and the serving layer.

One code path for training, evaluation and serving:

* the training template (``templates/time_series.py``) fits a model with :func:`fit`,
  scores it with :func:`backtest` and saves :func:`serving_state`;
* the API and the exported bundle forecast with :func:`forecast` from that state.

Models are stored as plain data (dicts, NumPy arrays and, for LightGBM, the fitted
``LGBMRegressor``), never as classes defined in a generated script, so a bundle unpickles
anywhere. Like ``inference.py`` this module imports only NumPy, pandas and LightGBM and
is shipped verbatim inside the deployment bundle.

Forecasts are *direct* multi-horizon: one global model per family predicts ``y(t + h)``
for ``h = 1..H`` from features known at the forecast origin ``t``. Every feature is
computed strictly from values at or before the origin, so there is no look-ahead.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

FAMILIES = ("lightgbm", "seasonal_naive", "ets")
FEATURES = [
    "series_code",
    "horizon_step",
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "seasonal_lag",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "dayofweek",
    "month",
    "day",
    "is_weekend",
]
MIN_HISTORY = 14
ETS_PARAMS = {"alpha": 0.2, "beta": 0.05, "gamma": 0.1}


# --------------------------------------------------------------------------- #
# Calendar helpers                                                             #
# --------------------------------------------------------------------------- #


def season_length(freq: str | None) -> int:
    """Length of the dominant seasonal cycle for a pandas frequency string."""
    f = (freq or "D").upper().lstrip("0123456789")
    if f.startswith(("MIN", "T")):
        return 60
    if f.startswith("H"):
        return 24
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith(("Q", "BQ")):
        return 4
    if f.startswith(("Y", "A")):
        return 1
    return 7  # daily (and anything unknown): weekly cycle


def date_offset(freq: str | None) -> pd.DateOffset:
    """The step between consecutive observations."""
    f = freq or "D"
    if f.upper() == "M":
        f = "MS"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.tseries.frequencies.to_offset(f)
    except ValueError:
        return pd.tseries.frequencies.to_offset("D")


def _calendar(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    dow = dates.dayofweek.to_numpy()
    return {
        "dayofweek": dow,
        "month": dates.month.to_numpy(),
        "day": dates.day.to_numpy(),
        "is_weekend": (dow >= 5).astype(int),
    }


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #


def _origin_frame(values: np.ndarray) -> pd.DataFrame:
    """Per-position features known at origin ``t`` (row ``t`` uses ``values[..t]`` only)."""
    s = pd.Series(values, dtype=float)
    return pd.DataFrame(
        {
            "lag_1": s,
            "lag_2": s.shift(1).fillna(s),
            "lag_7": s.shift(6).bfill(),
            "lag_14": s.shift(13).bfill(),
            "rolling_mean_7": s.rolling(7, min_periods=1).mean(),
            "rolling_std_7": s.rolling(7, min_periods=1).std(ddof=0),
            "rolling_mean_14": s.rolling(14, min_periods=1).mean(),
            "rolling_std_14": s.rolling(14, min_periods=1).std(ddof=0),
        }
    )


def _seasonal_index(origin: np.ndarray, h: np.ndarray, season: int) -> np.ndarray:
    """Index of the latest value at or before the origin in the same seasonal phase as ``origin + h``."""
    return origin + h - season * np.ceil(h / season).astype(int)


def _rows(
    values: np.ndarray,
    origins: np.ndarray,
    steps: np.ndarray,
    target_dates: pd.DatetimeIndex,
    code: int,
    season: int,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows for forecasting ``values`` from each ``origins[i]`` ``steps[i]`` steps ahead.

    ``frame`` is ``_origin_frame(values)``, passed in when the caller reuses it across calls.
    """
    frame = _origin_frame(values) if frame is None else frame
    base = frame.iloc[origins].reset_index(drop=True)
    s_idx = _seasonal_index(origins, steps, season)
    seasonal = np.where(s_idx >= 0, values[np.clip(s_idx, 0, None)], values[origins])
    rows = base.assign(series_code=code, horizon_step=steps, seasonal_lag=seasonal, **_calendar(target_dates))
    return rows[FEATURES]


def training_examples(
    series: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    codes: dict[str, int],
    horizon: int,
    season: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Supervised (X, y) pairs from every origin of every series and every step ``1..horizon``."""
    frames, targets = [], []
    for sid, (dates, values) in series.items():
        n = len(values)
        origin_frame = _origin_frame(values)
        for h in range(1, horizon + 1):
            origins = np.arange(MIN_HISTORY - 1, n - h)
            if len(origins) == 0:
                continue
            steps = np.full(len(origins), h)
            frames.append(
                _rows(values, origins, steps, dates[origins + h], codes.get(sid, -1), season, origin_frame)
            )
            targets.append(values[origins + h])
    if not frames:
        return pd.DataFrame(columns=FEATURES), np.array([])
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


# --------------------------------------------------------------------------- #
# Exponential smoothing                                                        #
# --------------------------------------------------------------------------- #


def _ets_forecasts(
    values: np.ndarray, season: int, origins: dict[int, int], params: dict[str, float] | None = None
) -> dict[int, np.ndarray]:
    """Additive Holt-Winters run once over ``values``; returns forecasts made at each origin.

    ``origins`` maps an origin index ``t`` to the number of steps to forecast from it.
    Forecasts at ``t`` use only ``values[..t]``.
    """
    p = {**ETS_PARAMS, **(params or {})}
    a, b, g = p["alpha"], p["beta"], p["gamma"]
    s_len = max(season, 2)
    n = len(values)
    out: dict[int, np.ndarray] = {}
    if n < 2 * s_len:  # not enough history for seasonality: damped level only
        for t, k in origins.items():
            out[t] = np.full(k, float(values[max(0, min(t, n - 1))]) if n else 0.0)
        return out
    level = float(np.mean(values[:s_len]))
    trend = float((np.mean(values[s_len : 2 * s_len]) - level) / s_len)
    seas = values[:s_len].astype(float) - level
    for t in range(n):
        y = float(values[t])
        prev_level = level
        level = a * (y - seas[t % s_len]) + (1 - a) * (prev_level + trend)
        trend = b * (level - prev_level) + (1 - b) * trend
        seas[t % s_len] = g * (y - level) + (1 - g) * seas[t % s_len]
        if t in origins:
            h = np.arange(1, origins[t] + 1)
            out[t] = level + h * trend + seas[(t + h) % s_len]
    return out


# --------------------------------------------------------------------------- #
# Fit / forecast                                                               #
# --------------------------------------------------------------------------- #


def fit(
    family: str,
    train: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    *,
    horizon: int,
    freq: str | None,
    valid: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Fit a forecaster on the training block and return its (history-free) model state."""
    if family not in FAMILIES:
        family = "lightgbm"
    season = season_length(freq)
    model: dict[str, Any] = {
        "family": family,
        "horizon": horizon,
        "frequency": freq,
        "season": season,
        "codes": {sid: i for i, sid in enumerate(sorted(train))},
    }
    if family == "ets":
        model["params"] = {k: float(v) for k, v in (params or {}).items() if k in ETS_PARAMS}
    if family != "lightgbm":
        return model

    import lightgbm as lgb

    X, y = training_examples(train, model["codes"], horizon, season)
    lgb_params = {
        "objective": "regression_l1",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    allowed = {*lgb_params, "max_depth", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda"}
    lgb_params.update({k: v for k, v in (params or {}).items() if k in allowed})
    regressor = lgb.LGBMRegressor(**lgb_params)
    fit_kwargs: dict[str, Any] = {"categorical_feature": ["series_code"]}
    if valid:
        X_val, y_val, _ = backtest_examples(model, train, valid)
        if len(X_val):
            fit_kwargs.update(eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])
    regressor.fit(X, y, **fit_kwargs)
    model["regressor"] = regressor
    return model


def _forecast_from(
    model: dict[str, Any],
    sid: str,
    values: np.ndarray,
    origin: int,
    steps: np.ndarray,
    dates: pd.DatetimeIndex,
    frame: pd.DataFrame | None = None,
) -> np.ndarray:
    """Forecast ``steps`` ahead from ``values[..origin]`` (``dates`` are the target dates)."""
    season = model["season"]
    if model["family"] == "seasonal_naive":
        idx = _seasonal_index(np.full(len(steps), origin), steps, season)
        return np.where(idx >= 0, values[np.clip(idx, 0, None)], values[origin]).astype(float)
    if model["family"] == "ets":
        k = int(steps.max())
        path = _ets_forecasts(values[: origin + 1], season, {origin: k}, model.get("params"))[origin]
        return path[steps - 1]
    X = _rows(values, np.full(len(steps), origin), steps, dates, model["codes"].get(sid, -1), season, frame)
    return np.asarray(model["regressor"].predict(X), dtype=float)


def backtest_examples(
    model: dict[str, Any],
    history: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    block: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[str, int]]]:
    """LightGBM feature rows for a rolling-origin pass over ``block`` (used for early stopping)."""
    frames, targets, keys = [], [], []
    for sid, (dates, values) in block.items():
        h_dates, h_values = history.get(sid, (pd.DatetimeIndex([]), np.array([])))
        all_values = np.concatenate([h_values, values])
        all_dates = h_dates.append(dates)
        origin_frame = _origin_frame(all_values)
        for start in range(0, len(values), model["horizon"]):
            origin = len(h_values) + start - 1
            if origin < 0:
                continue
            steps = np.arange(1, min(model["horizon"], len(values) - start) + 1)
            frames.append(
                _rows(all_values, np.full(len(steps), origin), steps, all_dates[origin + steps],
                      model["codes"].get(sid, -1), model["season"], origin_frame)
            )  # fmt: skip
            targets.append(values[start : start + len(steps)])
            keys.extend((sid, start + i) for i in range(len(steps)))
    if not frames:
        return pd.DataFrame(columns=FEATURES), np.array([]), []
    return pd.concat(frames, ignore_index=True), np.concatenate(targets), keys


def backtest(
    model: dict[str, Any],
    history: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    block: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Rolling-origin evaluation of ``model`` over ``block``.

    Every ``horizon`` steps a new forecast is made from all actual values before it (the
    history plus the block so far), for at most ``horizon`` steps. Returns per-series
    predictions aligned with the block's values.
    """
    preds: dict[str, np.ndarray] = {}
    horizon = model["horizon"]
    for sid, (dates, values) in block.items():
        h_dates, h_values = history.get(sid, (pd.DatetimeIndex([]), np.array([])))
        all_values = np.concatenate([h_values, values])
        all_dates = h_dates.append(dates)
        out = np.full(len(values), np.nan)
        origins = {
            len(h_values) + s - 1: min(horizon, len(values) - s) for s in range(0, len(values), horizon)
        }
        origins = {t: k for t, k in origins.items() if t >= 0}
        is_ets = model["family"] == "ets"
        ets = _ets_forecasts(all_values, model["season"], origins, model.get("params")) if is_ets else {}
        origin_frame = _origin_frame(all_values) if model["family"] == "lightgbm" else None
        for origin, k in origins.items():
            start = origin + 1 - len(h_values)
            steps = np.arange(1, k + 1)
            if is_ets:
                out[start : start + k] = ets[origin]
            else:
                out[start : start + k] = _forecast_from(
                    model, sid, all_values, origin, steps, all_dates[origin + steps], origin_frame
                )
        preds[sid] = out
    return preds


def serving_state(
    model: dict[str, Any], full: dict[str, tuple[pd.DatetimeIndex, np.ndarray]]
) -> dict[str, Any]:
    """Attach the most recent history of every series so the model can forecast past the end of the data."""
    keep = max(2 * model["season"], MIN_HISTORY + model["season"], 28)
    state = dict(model)
    state["history"] = {
        sid: {
            # ETS replays its recursion over the whole series; the other families need only the recent window
            "values": (values if model["family"] == "ets" else values[-keep:]).astype(float).tolist(),
            "last_date": dates[-1].isoformat(),
        }
        for sid, (dates, values) in full.items()
        if len(values)
    }
    return state


def forecast(
    state: dict[str, Any], series_ids: Sequence[str], steps: Sequence[int]
) -> tuple[np.ndarray, list[str]]:
    """Forecast ``steps[i]`` steps past the end of the data for ``series_ids[i]``.

    Returns the forecasts and their target dates (ISO strings).

    Raises:
        KeyError: An unknown series id.
        ValueError: A step outside ``1..horizon``.
    """
    horizon = state["horizon"]
    offset = date_offset(state.get("frequency"))
    preds = np.empty(len(series_ids))
    dates_out: list[str] = [""] * len(series_ids)
    steps_arr = np.asarray(steps, dtype=int)
    if len(steps_arr) and (steps_arr.min() < 1 or steps_arr.max() > horizon):
        raise ValueError(f"horizon_step must be between 1 and {horizon}")
    by_series: dict[str, list[int]] = {}
    for i, sid in enumerate(series_ids):
        if sid not in state["history"]:
            raise KeyError(sid)
        by_series.setdefault(sid, []).append(i)
    for sid, rows in by_series.items():
        hist = state["history"][sid]
        values = np.asarray(hist["values"], dtype=float)
        last = pd.Timestamp(hist["last_date"])
        s = steps_arr[rows]
        target = pd.DatetimeIndex([last + int(k) * offset for k in s])
        preds[rows] = _forecast_from(state, sid, values, len(values) - 1, s, target)
        for j, r in enumerate(rows):
            dates_out[r] = target[j].isoformat()
    return preds, dates_out


# --------------------------------------------------------------------------- #
# Data helpers                                                                 #
# --------------------------------------------------------------------------- #


def series_key(frame: pd.DataFrame, series_columns: Sequence[str]) -> pd.Series:
    """Composite series id (``"series_0"`` for a single series)."""
    if not series_columns:
        return pd.Series("series_0", index=frame.index)
    return frame[list(series_columns)].astype(str).agg("-".join, axis=1)


def to_series(
    frame: pd.DataFrame, time_column: str, target: str, series_columns: Sequence[str]
) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    """Split a long frame into time-ordered ``(dates, values)`` per series, dropping missing values."""
    df = pd.DataFrame(
        {
            "sid": series_key(frame, series_columns),
            "dt": pd.to_datetime(frame[time_column], errors="coerce"),
            "y": pd.to_numeric(frame[target], errors="coerce"),
        }
    ).dropna()
    df = df.sort_values(["sid", "dt"], kind="stable")
    return {
        sid: (pd.DatetimeIndex(g["dt"]), g["y"].to_numpy(dtype=float))
        for sid, g in df.groupby("sid", sort=True)
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Forecast accuracy: sMAPE and MAPE (in percent), MAE and RMSE."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if not len(yt):
        return {"smape": math.nan, "mae": math.nan, "rmse": math.nan, "mape": math.nan}
    denom = np.abs(yt) + np.abs(yp)
    smape = np.where(denom > 1e-8, 200.0 * np.abs(yt - yp) / np.where(denom > 1e-8, denom, 1.0), 0.0)
    return {
        "smape": float(np.mean(smape)),
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mape": float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-8)) * 100.0),
    }
