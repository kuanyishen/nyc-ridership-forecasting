"""Forecast accuracy metrics and a bootstrap that resamples forecast ORIGINS.

The 14 forecasts made from one origin share the same history and the same shocks, so they are not independent. All
uncertainty statements therefore resample whole origins, never individual forecast days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calendar


def wape(df: pd.DataFrame) -> float:
    """Weighted absolute percentage error: sum|error| / sum(actual). Robust, and unlike MAPE not dominated by small days."""
    return float((df["y_pred"] - df["y_true"]).abs().sum() / df["y_true"].sum())


def annotate(bt: pd.DataFrame) -> pd.DataFrame:
    """Add day-type columns used for slicing the results."""
    f = calendar.features(pd.DatetimeIndex(bt["date"]))
    out = bt.copy()
    out["dow"] = f["dow"].values
    out["is_holiday"] = f["is_holiday"].values
    out["near_holiday"] = ((f["to_next_hol"] <= 1) | (f["since_prev_hol"] <= 1)).values.astype(int)
    out["day_type"] = np.where(out["near_holiday"] == 1, "holiday period", np.where(out["dow"] >= 5, "weekend", "weekday"))
    out["h_bucket"] = pd.cut(out["h"], [0, 3, 7, 14], labels=["1-3 days", "4-7 days", "8-14 days"])
    return out


def summary(bt: pd.DataFrame, reference: str, by: list[str] | None = None) -> pd.DataFrame:
    """Per series x model (x extra grouping): WAPE, MAE, bias, and skill = 1 - MAE / MAE of the reference model."""
    by = by or []
    keys = ["series", *by]
    rows = []
    for k, g in bt.groupby(keys + ["model"], observed=True):
        k = k if isinstance(k, tuple) else (k,)
        ref = bt
        for col, val in zip(keys, k[:-1]):
            ref = ref[ref[col] == val]
        ref = ref[ref["model"] == reference]
        mae = (g["y_pred"] - g["y_true"]).abs().mean()
        ref_mae = (ref["y_pred"] - ref["y_true"]).abs().mean()
        rows.append({**dict(zip(keys + ["model"], k)), "n": len(g), "wape": wape(g), "mae": mae,
                     "bias": float(((g["y_pred"] - g["y_true"]) / g["y_true"]).mean()), "skill_vs_ref": 1 - mae / ref_mae})
    return pd.DataFrame(rows)


def origin_bootstrap(bt: pd.DataFrame, series: str, a: str, b: str, *, n_boot: int = 4000, seed: int = 0) -> dict:
    """WAPE(a) - WAPE(b) for one series with a 95% CI from resampling origins (negative = a is better)."""
    sub = bt[(bt["series"] == series) & bt["model"].isin([a, b])].copy()
    sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()
    per = sub.groupby(["origin", "model"]).agg(err=("abs_err", "sum"), y=("y_true", "sum")).unstack("model")
    ea, eb, ya, yb = per[("err", a)].values, per[("err", b)].values, per[("y", a)].values, per[("y", b)].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(n_boot, len(ea)))
    diff = ea[idx].sum(axis=1) / ya[idx].sum(axis=1) - eb[idx].sum(axis=1) / yb[idx].sum(axis=1)
    est = ea.sum() / ya.sum() - eb.sum() / yb.sum()
    lo, hi = np.quantile(diff, [0.025, 0.975])
    p = float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return {"series": series, "a": a, "b": b, "wape_diff": float(est), "lo": float(lo), "hi": float(hi), "p_value": min(p, 1.0), "n_origins": int(len(ea))}


def origin_wapes(bt: pd.DataFrame, series: str, model: str) -> pd.Series:
    sub = bt[(bt["series"] == series) & (bt["model"] == model)]
    g = sub.assign(abs_err=(sub["y_pred"] - sub["y_true"]).abs()).groupby("origin")
    return g["abs_err"].sum() / g["y_true"].sum()


def interval_metrics(bt: pd.DataFrame, level: float) -> pd.DataFrame:
    """Coverage (share of actuals inside the interval; target = level) and relative interval width."""
    d = bt.dropna(subset=["lo", "hi"]).copy()
    d["inside"] = (d["y_true"] >= d["lo"]) & (d["y_true"] <= d["hi"])
    d["rel_width"] = (d["hi"] - d["lo"]) / d["y_pred"]
    return d.groupby(["series", "model"]).agg(n=("inside", "size"), coverage=("inside", "mean"), rel_width=("rel_width", "mean")).reset_index().assign(nominal=level)


def origin_ci(bt: pd.DataFrame, series: str, model: str, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """WAPE of one model on one series with a 95% CI from resampling origins."""
    sub = bt[(bt["series"] == series) & (bt["model"] == model)].copy()
    sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()
    per = sub.groupby("origin").agg(err=("abs_err", "sum"), y=("y_true", "sum"))
    e, y = per["err"].values, per["y"].values
    idx = np.random.default_rng(seed).integers(0, len(e), size=(n_boot, len(e)))
    boot = e[idx].sum(axis=1) / y[idx].sum(axis=1)
    return float(e.sum() / y.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
