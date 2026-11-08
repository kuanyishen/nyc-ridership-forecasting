"""Forecasters. Each takes the history up to the forecast origin and returns a Series for the next ``horizon`` days.

Baselines come first because every model has to beat them:
* naive            - repeat the last value (ignores the weekly cycle, so it is a floor, not a contender)
* seasonal_naive   - same weekday last week (or the most recent same weekday, for horizons beyond 7 days)
* seasonal_median4 - median of the last four same weekdays (robust to one odd week)
Then Prophet and a direct multi-horizon XGBoost, and their average.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import calendar

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def _future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _same_weekday_lags(y: np.ndarray, o: int, h: int, n: int = 4) -> list[float]:
    """Values at the n most recent days that share the weekday of day o+h and are not after the origin o."""
    k = -(-h // 7)                                            # ceil(h / 7)
    return [y[o + h - 7 * (k + j)] if o + h - 7 * (k + j) >= 0 else np.nan for j in range(n)]


def naive(history: pd.Series, horizon: int, **_) -> pd.Series:
    return pd.Series(history.iloc[-1], index=_future_index(history, horizon))


def seasonal_naive(history: pd.Series, horizon: int, **_) -> pd.Series:
    y, o = history.values, len(history) - 1
    return pd.Series([_same_weekday_lags(y, o, h, 1)[0] for h in range(1, horizon + 1)], index=_future_index(history, horizon))


def seasonal_median4(history: pd.Series, horizon: int, **_) -> pd.Series:
    y, o = history.values, len(history) - 1
    return pd.Series([np.nanmedian(_same_weekday_lags(y, o, h, 4)) for h in range(1, horizon + 1)], index=_future_index(history, horizon))


# ------------------------------------------------------------------------------------------------ XGBoost (direct)

WX_COLS = ["prcp", "snow", "snwd", "tmax", "tmin"]


def _features_for(y: np.ndarray, dates: pd.DatetimeIndex, origins: np.ndarray, h: int, wx: pd.DataFrame | None = None,
                  wx_max_h: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Features for forecasting day origin+h from information available at ``origins`` (array of positions).

    Returns (features, base) where base is the seasonal level the model's log-ratio output is relative to.

    ``wx`` (observed weather by date) adds the weather OF THE TARGET DAY. That is information a real forecast does not
    have 14 days ahead, so it is only for oracle experiments: ``wx_max_h`` keeps it for horizons up to that many days
    (mimicking a reliable short-range weather forecast) and sets it to NaN beyond.
    """
    k = -(-h // 7)
    n = len(y)
    pad = np.concatenate([np.full(7 * (k + 4) + 70, np.nan), y])      # so negative positions read NaN
    off = 7 * (k + 4) + 70
    t = origins + h
    lags = np.stack([pad[off + t - 7 * (k + j)] for j in range(4)], axis=1)
    with np.errstate(all="ignore"):
        base = np.nanmedian(lags, axis=1)
        y_o = pad[off + origins]
        m7 = np.nanmean(np.stack([pad[off + origins - i] for i in range(7)], axis=1), axis=1)
        m28 = np.nanmean(np.stack([pad[off + origins - i] for i in range(28)], axis=1), axis=1)
        m28p = np.nanmean(np.stack([pad[off + origins - 28 - i] for i in range(28)], axis=1), axis=1)
    target_dates = pd.DatetimeIndex(dates[0] + pd.to_timedelta(t, unit="D"))
    cal = calendar.features(target_dates)
    lag_dates = [pd.DatetimeIndex(dates[0] + pd.to_timedelta(t - 7 * (k + j), unit="D")) for j in range(4)]
    hol = set(calendar.holiday_dates())
    f = pd.DataFrame({
        "h": h, "log_lag1": np.log(lags[:, 0] / base), "log_lag2": np.log(lags[:, 1] / base), "log_lag3": np.log(lags[:, 2] / base),
        "log_yo": np.log(y_o / base), "log_m7": np.log(m7 / base), "trend28": np.log(m28 / m28p), "ratio7_28": np.log(m7 / m28),
        **{f"lag{j + 1}_is_hol": np.array([d in hol for d in lag_dates[j]], dtype=int) for j in range(4)},
    })
    f = pd.concat([f, cal.reset_index(drop=True)], axis=1)
    if wx is not None:
        use = wx_max_h is None or h <= wx_max_h
        vals = wx.reindex(target_dates)[WX_COLS].values if use else np.full((len(target_dates), len(WX_COLS)), np.nan)
        for j, c in enumerate(WX_COLS):
            f[f"wx_{c}"] = vals[:, j]
    f.index = origins
    return f.replace([np.inf, -np.inf], np.nan), base


def xgboost_forecast(history: pd.Series, horizon: int, *, train_start: str | None = None, n_estimators: int = 300, seed: int = 0,
                     weather: pd.DataFrame | None = None, weather_max_h: int | None = None, **_) -> pd.Series:
    """One model for all horizons: predict log(y_T / base) where base = median of the last four same-weekday values."""
    from xgboost import XGBRegressor

    h_all = history.loc[train_start:] if train_start else history
    y, dates = h_all.values.astype(float), h_all.index
    X, Y = [], []
    for h in range(1, horizon + 1):
        origins = np.arange(35, len(y) - h)
        if len(origins) == 0:
            continue
        f, base = _features_for(y, dates, origins, h, weather, weather_max_h)
        tgt = np.log(y[origins + h] / base)
        keep = np.isfinite(tgt) & np.isfinite(base)
        X.append(f[keep]), Y.append(tgt[keep])
    X, Y = pd.concat(X), np.concatenate(Y)
    model = XGBRegressor(n_estimators=n_estimators, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, objective="reg:squarederror", n_jobs=1, random_state=seed, verbosity=0)
    model.fit(X, Y)
    o = np.array([len(y) - 1])
    preds = []
    for h in range(1, horizon + 1):
        f, base = _features_for(y, dates, o, h, weather, weather_max_h)
        preds.append(float(base[0] * np.exp(model.predict(f)[0])))
    return pd.Series(preds, index=_future_index(history, horizon))


# ------------------------------------------------------------------------------------------------ Prophet

def prophet_forecast(history: pd.Series, horizon: int, *, train_start: str | None = None, **_) -> pd.Series:
    from prophet import Prophet

    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    h_all = history.loc[train_start:] if train_start else history
    df = pd.DataFrame({"ds": h_all.index, "y": h_all.values}).dropna()
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.1, holidays=calendar.holiday_frame(), uncertainty_samples=0)
    m.fit(df)
    fut = pd.DataFrame({"ds": _future_index(history, horizon)})
    return pd.Series(m.predict(fut)["yhat"].values, index=fut["ds"])


def xgboost_plain(history: pd.Series, horizon: int, *, weather: pd.DataFrame | None = None, **kw) -> pd.Series:
    """The production XGBoost: it must NEVER see weather, even when a backtest hands weather to every model."""
    kw.pop("weather_max_h", None)
    return xgboost_forecast(history, horizon, **kw)


def xgboost_wx(history: pd.Series, horizon: int, *, weather: pd.DataFrame | None = None, **kw) -> pd.Series:
    """ORACLE: XGBoost that is told the actual weather of every target day."""
    kw.pop("weather_max_h", None)
    return xgboost_forecast(history, horizon, weather=weather, **kw)


def xgboost_wx3(history: pd.Series, horizon: int, *, weather: pd.DataFrame | None = None, **kw) -> pd.Series:
    """ORACLE: told the actual weather only for the next 3 days (a perfect short-range forecast); blind beyond that."""
    kw.pop("weather_max_h", None)
    return xgboost_forecast(history, horizon, weather=weather, weather_max_h=3, **kw)


BASE_MODELS = {"naive": naive, "seasonal_naive": seasonal_naive, "seasonal_median4": seasonal_median4,
               "prophet": prophet_forecast, "xgboost": xgboost_plain, "xgboost_wx": xgboost_wx, "xgboost_wx3": xgboost_wx3}


def run_models(history: pd.Series, horizon: int, names: list[str], **kw) -> dict[str, pd.Series]:
    """Forecast with every requested model; 'ensemble' is the mean of Prophet and XGBoost."""
    need = set(names) | ({"prophet", "xgboost"} if "ensemble" in names else set())
    out = {n: BASE_MODELS[n](history, horizon, **kw) for n in need if n in BASE_MODELS}
    if "ensemble" in names:
        out["ensemble"] = (out["prophet"] + out["xgboost"]) / 2
    return {n: out[n] for n in names}
