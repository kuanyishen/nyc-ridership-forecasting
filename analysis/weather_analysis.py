"""Weather analysis: how much do storms move ridership, which drift alarms were weather, and does weather help?

    python analysis/weather_analysis.py        # writes docs/assets/weather.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ridership import data, drift, metrics, weather
from ridership.run import load_config

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
COLS = ["snow_light", "snow_heavy", "snwd", "prcp", "cold", "heat"]
FIT_DESCRIBE = ("2021-06-01", None)             # post-collapse period, for the descriptive effect sizes
FIT_MONITOR = ("2020-03-01", "2024-06-30")      # fitted ONLY on data before the monitor's evaluation period


def fit(s: pd.Series, wdiff: pd.DataFrame, window: tuple[str, str | None]):
    sub = s.loc[window[0]:window[1]]
    X = wdiff.reindex(sub.index)[COLS]
    ok = sub.notna() & X.notna().all(axis=1)
    beta, cov = weather.ols_robust(sub[ok].values, X[ok].values, return_cov=True)
    return beta, cov, int(ok.sum())


def snow_effect(beta, cov, inches: float) -> dict:
    """Predicted % change in ridership on a day with ``inches`` of new snow (other weather at its baseline), with a 95% CI."""
    x = np.zeros(len(beta)); x[1], x[2] = min(inches, 3.0), max(inches - 3.0, 0.0)
    est, se = float(x @ beta), float(np.sqrt(x @ cov @ x))
    return {"inches": inches, "pct": float(np.expm1(est)), "lo": float(np.expm1(est - 1.96 * se)), "hi": float(np.expm1(est + 1.96 * se))}


def main() -> None:
    cfg = load_config()
    wide = data.to_wide(data.load_raw())
    w = weather.load()
    wf = weather.features(w)
    wdiff = weather.baseline_difference(wf)
    k, tune_start, tune_end, oos_start = 0.5, "2023-01-01", "2024-06-30", "2024-07-01"
    base_monitor = json.loads((ASSETS / "monitor.json").read_text())
    out: dict = {"weather": {"station": weather.STATION, "start": str(w.index[0].date()), "end": str(w.index[-1].date()), "n_days": len(w),
                             "snow_days_1in": int((w["snow"] >= 1).sum()), "storm_days_6in": int((w["snow"] >= 6).sum())},
                 "effects": {}, "storms": [], "alarms": [], "monitor": {}}

    # ---- storms: model-free, ridership vs the same weekday over the previous four weeks ------------------------------
    storm_days = w[w["snow"] >= 4].index
    for d in storm_days:
        row = {"date": str(d.date()), "weekday": d.day_name(), "snow": float(w.loc[d, "snow"]), "snwd": float(w.loc[d, "snwd"]), "pct": {}}
        for name in cfg["series"]:
            y = data.get_series(wide, name)
            if d in y.index and d >= y.index[0] + pd.Timedelta(days=28):
                base = np.nanmedian([y.get(d - pd.Timedelta(days=7 * j), np.nan) for j in (1, 2, 3, 4)])
                row["pct"][name] = float(y[d] / base - 1) if d in y.index and pd.notna(y[d]) else None
        out["storms"].append(row)

    # ---- effect sizes per series + the weather-aware monitor -------------------------------------------------------------
    for name in cfg["series"]:
        y = data.get_series(wide, name)
        s = drift.surprise(y)
        beta, cov, n = fit(s, wdiff, FIT_DESCRIBE)
        se = np.sqrt(np.diag(cov))
        out["effects"][name] = {"n_days": n, "snow_effects": [snow_effect(beta, cov, i) for i in (2.0, 6.0, 11.0)],
                                "per_inch_rain": {"pct": float(np.expm1(beta[4])), "se": float(se[4])},
                                "per_degree_below_25F": {"pct": float(np.expm1(beta[5])), "se": float(se[5])}}
        beta_m, _, _ = fit(s, wdiff, FIT_MONITOR)
        effect = pd.Series(wdiff.reindex(y.index)[COLS].fillna(0.0).values @ beta_m[1:], index=y.index)
        for d in pd.to_datetime(["2026-01-25", "2026-01-26", "2026-02-22", "2026-02-23"]):         # out-of-sample check of the weather model
            if d in s.index and pd.notna(s.loc[d]):
                out.setdefault("storm_fit", []).append({"series": name, "date": str(d.date()), "actual": float(np.expm1(s.loc[d])), "predicted": float(np.expm1(effect.loc[d]))})
        h0 = base_monitor["series"][name]["h"]
        al0 = drift.detect(y, k, h0).loc[oos_start:]
        h1 = drift.tune_h(y, k, tune_start, tune_end, 1.0, weather_effect=effect)
        al1 = drift.detect(y, k, h1, weather_effect=effect).loc[oos_start:]
        years = (y.index[-1] - pd.Timestamp(oos_start)).days / 365.25
        starts = list(pd.date_range("2024-09-02", y.index[-1] - pd.Timedelta(days=35), periods=24).normalize())
        inj = [drift.injection_test(y, d, starts, k, h1, weather_effect=effect) for d in (-0.2, -0.1, -0.05, 0.1)]
        out["monitor"][name] = {"h_before": h0, "h_after": h1, "alarms_before": [str(d.date()) for d in al0.index], "alarms_after": [str(d.date()) for d in al1.index],
                                "rate_before": len(al0) / years, "rate_after": len(al1) / years, "injection_after": inj}
        for d, r in al0.iterrows():                                          # attribute every original alarm to weather (or not)
            wd = w.loc[max(d - pd.Timedelta(days=1), w.index[0]):d]
            weather_hit = bool((wd["snow"] >= 1).any() or w.loc[d, "snwd"] >= 3 or (wd["prcp"] >= 1.5).any())
            out["alarms"].append({"series": name, "date": str(d.date()), "direction": r["alarm"], "snow_today": float(w.loc[d, "snow"]), "snow_yesterday": float(w.loc[d - pd.Timedelta(days=1), "snow"]),
                                  "snow_depth": float(w.loc[d, "snwd"]), "prcp": float(w.loc[d, "prcp"]), "tmax": float(w.loc[d, "tmax"]), "weather_event": weather_hit,
                                  "removed_by_weather_adjustment": str(d.date()) not in [str(x.date()) for x in al1.index]})
        print(f"{name:7s} snow effect: 2in {out['effects'][name]['snow_effects'][0]['pct']*100:+.0f}%  6in {out['effects'][name]['snow_effects'][1]['pct']*100:+.0f}%  11in {out['effects'][name]['snow_effects'][2]['pct']*100:+.0f}% | "
              f"alarms {len(al0)} -> {len(al1)}  ({len(al0)/years:.1f}/yr -> {len(al1)/years:.1f}/yr), h {h0} -> {h1}")

    # ---- oracle forecasting experiment (needs the weather backtest) ------------------------------------------------------
    bw = ROOT / "data" / "processed" / "backtest_weather.csv"
    if bw.exists():
        bt = metrics.annotate(pd.read_csv(bw, parse_dates=["origin", "date"]))
        bad = (w["snow"] >= 1) | (w["snwd"] >= 3) | (w["prcp"] >= 1.0)
        bt["bad_weather"] = bt["date"].map(bad).fillna(False).astype(bool)
        res = {"n_bad_weather_days": int(bt[bt.model == "xgboost"].drop_duplicates(["series", "date"]).bad_weather.sum()), "wape": {}, "paired": {}, "by_series": {}}
        for m in ("xgboost", "xgboost_wx3", "xgboost_wx"):
            g = bt[bt.model == m]
            res["wape"][m] = {"all": metrics.wape(g), "bad_weather": metrics.wape(g[g.bad_weather]), "calm": metrics.wape(g[~g.bad_weather])}
            res["by_series"][m] = {s: metrics.wape(g[g.series == s]) for s in cfg["series"]}
        for s in cfg["series"]:
            for a in ("xgboost_wx3", "xgboost_wx"):
                res["paired"][f"{s}|{a}"] = metrics.origin_bootstrap(bt, s, a, "xgboost")
        out["oracle"] = res
        print("oracle WAPE:", {m: {k2: round(v * 100, 2) for k2, v in d.items()} for m, d in res["wape"].items()})
    (ASSETS / "weather.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote docs/assets/weather.json")


if __name__ == "__main__":
    main()
