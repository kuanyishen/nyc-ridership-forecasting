"""Empirical prediction intervals that use ONLY forecasts made at earlier origins (no peeking).

For each model and horizon bucket, the relative errors (actual / forecast - 1) of all earlier origins give the
quantiles that widen the new point forecast into an interval. Coverage is then checked honestly in the backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = [(1, 7), (8, 14)]


def _quantiles(rel: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(rel, a)), float(np.quantile(rel, 1 - a))


def add_intervals(bt: pd.DataFrame, level: float = 0.8, min_prior: int = 4) -> pd.DataFrame:
    """Add ``lo`` / ``hi`` columns to a backtest table (NaN until ``min_prior`` earlier origins exist)."""
    out = bt.copy()
    out["lo"], out["hi"] = np.nan, np.nan
    out["rel_err"] = out["y_true"] / out["y_pred"] - 1
    for (series, model), g in out.groupby(["series", "model"]):
        origins = sorted(g["origin"].unique())
        for i, o in enumerate(origins):
            if i < min_prior:
                continue
            prior = g[g["origin"] < o]
            now = g.index[g["origin"] == o]
            for lo_h, hi_h in BUCKETS:
                rel = prior.loc[(prior["h"] >= lo_h) & (prior["h"] <= hi_h), "rel_err"].dropna().values
                sel = now[(out.loc[now, "h"] >= lo_h) & (out.loc[now, "h"] <= hi_h)]
                if len(rel) >= 20 and len(sel):
                    q_lo, q_hi = _quantiles(rel, level)
                    out.loc[sel, "lo"] = out.loc[sel, "y_pred"] * (1 + q_lo)
                    out.loc[sel, "hi"] = out.loc[sel, "y_pred"] * (1 + q_hi)
    return out


def final_interval_quantiles(bt: pd.DataFrame, level: float = 0.8) -> dict:
    """Relative-error quantiles from ALL backtest origins, for intervals on the live forecast."""
    rel = (bt["y_true"] / bt["y_pred"] - 1).rename("rel").to_frame().join(bt[["series", "model", "h"]])
    out = {}
    for (series, model), g in rel.groupby(["series", "model"]):
        for lo_h, hi_h in BUCKETS:
            r = g.loc[(g["h"] >= lo_h) & (g["h"] <= hi_h), "rel"].dropna().values
            if len(r) >= 20:
                out[f"{series}|{model}|{lo_h}-{hi_h}"] = _quantiles(r, level)
    return out


def apply_final(forecast: pd.DataFrame, quantiles: dict) -> pd.DataFrame:
    """Attach lo/hi to a live forecast table with columns series, model, h, forecast."""
    out = forecast.copy()
    lo, hi = [], []
    for _, r in out.iterrows():
        bucket = next((f"{a}-{b}" for a, b in BUCKETS if a <= r["h"] <= b), None)
        q = quantiles.get(f"{r['series']}|{r['model']}|{bucket}")
        lo.append(r["forecast"] * (1 + q[0]) if q else np.nan)
        hi.append(r["forecast"] * (1 + q[1]) if q else np.nan)
    out["lo"], out["hi"] = lo, hi
    return out
