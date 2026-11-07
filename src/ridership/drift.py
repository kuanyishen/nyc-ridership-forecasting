"""Drift monitoring.

1. SERIES SHIFT (level change): each day's "surprise" is log(actual / median of the last four same weekdays), with days
   around holidays masked. A two-sided CUSUM on the surprise, scaled by its own recent robust spread, raises an alarm
   when ridership drifts away from its weekly pattern. Detection delay and false-alarm rate are measured by injecting
   known level shifts into real history, not assumed.
2. FORECAST QUALITY: the recent WAPE of live forecasts is compared with the distribution of WAPEs the backtest saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calendar


def surprise(y: pd.Series) -> pd.Series:
    """log(y_t / median(y_{t-7}, ..., y_{t-28})); NaN on and next to holidays (their patterns differ by design)."""
    base = pd.concat([y.shift(7 * k) for k in (1, 2, 3, 4)], axis=1).median(axis=1)
    s = np.log(y / base)
    f = calendar.features(y.index)
    masked = ((f["to_next_hol"] <= 1) | (f["since_prev_hol"] <= 1)).values
    s[masked] = np.nan
    return s


def robust_scale(s: pd.Series, window: int = 365, min_periods: int = 90) -> pd.Series:
    """Rolling 1.4826 x MAD of PAST surprises only (shifted one day so today's value never sets its own yardstick)."""
    def mad(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan

    return s.shift(1).rolling(window, min_periods=min_periods).apply(mad, raw=True)


def cusum(z: pd.Series, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """Two-sided CUSUM. Resets after an alarm. NaN days (masked or missing) carry the statistic forward unchanged."""
    pos = neg = 0.0
    rows = []
    for d, v in z.items():
        alarm = ""
        if np.isfinite(v):
            pos, neg = max(0.0, pos + v - k), max(0.0, neg - v - k)
            if pos > h:
                alarm, pos, neg = "up", 0.0, 0.0
            elif neg > h:
                alarm, pos, neg = "down", 0.0, 0.0
        rows.append((d, pos, neg, alarm))
    return pd.DataFrame(rows, columns=["date", "s_pos", "s_neg", "alarm"]).set_index("date")


def detect(y: pd.Series, k: float = 0.5, h: float = 5.0, scale: pd.Series | None = None) -> pd.DataFrame:
    """Alarms for a series; ``scale`` may be supplied (e.g. from clean history) instead of being recomputed."""
    s = surprise(y)
    sc = robust_scale(s) if scale is None else scale
    out = cusum(s / sc, k, h)
    return out[out["alarm"] != ""]


def tune_h(y: pd.Series, k: float, tune_start: str, tune_end: str, target_per_year: float, grid=np.arange(3.0, 15.1, 0.5)) -> float:
    """Smallest threshold whose false-alarm rate on a CLEAN tuning window is at most ``target_per_year``."""
    s = surprise(y)
    z = (s / robust_scale(s)).loc[tune_start:tune_end]
    years = (pd.Timestamp(tune_end) - pd.Timestamp(tune_start)).days / 365.25
    for h in grid:
        if len(cusum(z, k, h).query("alarm != ''")) / years <= target_per_year:
            return float(h)
    return float(grid[-1])


def injection_test(y: pd.Series, delta: float, starts: list[pd.Timestamp], k: float, h: float, max_delay: int = 28) -> dict:
    """Multiply the series by (1+delta) from each start date onward and measure whether/when the detector fires."""
    s_clean = surprise(y)
    scale = robust_scale(s_clean)               # yardstick from history the shift has not touched
    delays = []
    for d in starts:
        y_mod = y.copy()
        y_mod.loc[d:] *= 1 + delta
        z = (surprise(y_mod) / scale).loc[d - pd.Timedelta(days=0): d + pd.Timedelta(days=max_delay)]
        al = cusum(z, k, h).query("alarm != ''")
        delays.append((al.index[0] - d).days if len(al) else np.nan)
    arr = np.array(delays, dtype=float)
    return {"delta": delta, "n": len(arr), "detected_share": float(np.mean(~np.isnan(arr))),
            "median_delay_days": float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else None}


def performance_status(recent_wape: float, backtest_wapes: pd.Series, quantile: float = 0.9) -> dict:
    """Flag live forecast quality that is worse than what the backtest ever (almost) saw."""
    threshold = float(backtest_wapes.quantile(quantile))
    return {"recent_wape": float(recent_wape), "threshold": threshold, "alert": bool(recent_wape > threshold),
            "backtest_median": float(backtest_wapes.median())}
