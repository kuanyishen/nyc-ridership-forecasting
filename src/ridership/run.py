"""Command line entry points.

    python -m ridership.run fetch       # pull the latest published days from data.ny.gov into data/raw
    python -m ridership.run backtest    # rolling-origin backtest for every series and model (~5 min)
    python -m ridership.run forecast    # fit on all data, write the next 14 days with intervals
    python -m ridership.run monitor     # drift detectors + live forecast quality
    python -m ridership.run track       # log backtest results to MLflow
    python -m ridership.run weekly      # fetch + monitor + forecast + track (what the scheduled workflow runs)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import backtest as bt_mod
from . import data, drift, intervals, metrics, models, tracking

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FORECASTS = ROOT / "forecasts"
ASSETS = ROOT / "docs" / "assets"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"      # MLflow's file store is deprecated; SQLite needs no server


def load_config(path: Path = ROOT / "config" / "pipeline.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config()
    wide = data.to_wide(data.load_raw())
    series = args.series or cfg["series"]
    frames, t0 = [], time.time()
    for name in series:
        y = data.get_series(wide, name)
        origins = bt_mod.make_origins(y, cfg["backtest"]["first_origin"], cfg["backtest"]["step_days"], cfg["horizon"])
        print(f"{name}: {len(origins)} origins, {y.index[0].date()} -> {y.index[-1].date()}", flush=True)
        frames.append(bt_mod.run_backtest(y, origins, cfg["horizon"], cfg["models"], train_start=cfg["train_start"], workers=args.workers))
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
    bt = pd.concat(frames, ignore_index=True)
    bt = intervals.add_intervals(bt, cfg["interval_level"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / ("backtest.csv" if not args.series else "backtest_partial.csv")
    bt.to_csv(out, index=False)
    print(f"wrote {out} ({len(bt):,} forecasts)")


def cmd_fetch(args: argparse.Namespace) -> None:
    existing = data.load_raw()
    try:
        new = data.fetch_latest()
    except Exception as e:                                     # network or schema trouble must not break the weekly run
        print(f"fetch failed ({type(e).__name__}: {e}); keeping the existing file")
        return
    merged = data.merge_update(existing, new)
    changed = not existing.sort_values(["mode", "date"]).reset_index(drop=True).equals(merged)      # equals() treats NaN == NaN
    added = len(merged) - len(existing)
    print(f"API returned {len(new):,} rows; file had {len(existing):,}; now {len(merged):,} (+{added}); last date {merged['date'].max().date()}")
    if changed:
        data.save_raw(merged)
        print("raw file updated")
    else:
        print("no change")


def cmd_forecast(args: argparse.Namespace) -> None:
    cfg = load_config()
    wide = data.to_wide(data.load_raw())
    bt = pd.read_csv(PROCESSED / "backtest.csv", parse_dates=["origin", "date"])
    quantiles = intervals.final_interval_quantiles(bt, cfg["interval_level"])
    rows = []
    for name in cfg["series"]:
        y = data.get_series(wide, name)
        for model, f in models.run_models(y, cfg["horizon"], cfg["models"], train_start=cfg["train_start"]).items():
            for h, (d, v) in enumerate(f.items(), start=1):
                rows.append({"series": name, "model": model, "origin": y.index[-1], "date": d, "h": h, "forecast": float(v)})
    fc = intervals.apply_final(pd.DataFrame(rows), quantiles)
    FORECASTS.mkdir(exist_ok=True)
    stamp = fc["origin"].max().strftime("%Y-%m-%d")
    out = FORECASTS / f"{stamp}.csv"
    fc.to_csv(out, index=False)
    ens = fc[fc["model"] == "ensemble"]
    latest = {"data_end": stamp, "level": cfg["interval_level"], "series": {}}
    for name in cfg["series"]:
        y = data.get_series(wide, name)
        e = ens[ens["series"] == name]
        latest["series"][name] = {"history": {str(d.date()): float(v) for d, v in y.iloc[-90:].items()},
                                  "forecast": [{"date": str(r["date"].date()), "h": int(r["h"]), "forecast": r["forecast"], "lo": None if pd.isna(r["lo"]) else r["lo"], "hi": None if pd.isna(r["hi"]) else r["hi"]} for _, r in e.iterrows()]}
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "latest_forecast.json").write_text(json.dumps(latest, indent=1))
    print(f"wrote {out} ({len(fc)} rows) and docs/assets/latest_forecast.json")
    table = pd.DataFrame([{"series": n, "model": "ensemble", "horizon_days": cfg["horizon"], "n_training_days": len(data.get_series(wide, n).loc[cfg["train_start"]:])} for n in cfg["series"]])
    tracking.log_runs(table, {"train_start": cfg["train_start"], "models": ",".join(cfg["models"])}, kind="live_forecast", data_end=stamp, tracking_uri=MLFLOW_URI)


def live_evaluation(wide: pd.DataFrame) -> pd.DataFrame:
    """Score every saved forecast against whatever actuals have since been published."""
    rows = []
    for f in sorted(FORECASTS.glob("*.csv")):
        fc = pd.read_csv(f, parse_dates=["origin", "date"])
        for name, g in fc.groupby("series"):
            actual = data.get_series(wide, name)
            g = g.assign(y_true=g["date"].map(actual), y_pred=g["forecast"]).dropna(subset=["y_true"])
            if len(g):
                rows.append(g.assign(forecast_file=f.name))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cmd_monitor(args: argparse.Namespace) -> None:
    cfg = load_config()
    wide = data.to_wide(data.load_raw())
    k, tune_start, tune_end, oos_start = 0.5, "2023-01-01", "2024-06-30", "2024-07-01"
    out: dict = {"k": k, "tuning_window": [tune_start, tune_end], "evaluation_start": oos_start, "series": {}}
    deltas = [-0.2, -0.1, -0.05, 0.1]
    for name in cfg["series"]:
        y = data.get_series(wide, name)
        h = drift.tune_h(y, k, tune_start, tune_end, target_per_year=1.0)
        al = drift.detect(y, k, h)
        oos = al.loc[oos_start:]
        years = (y.index[-1] - pd.Timestamp(oos_start)).days / 365.25
        starts = list(pd.date_range("2024-09-02", y.index[-1] - pd.Timedelta(days=35), periods=24).normalize())
        inj = [drift.injection_test(y, d, starts, k, h) for d in deltas]
        recent = al.loc[y.index[-60]:]
        out["series"][name] = {"h": h, "alarms_out_of_sample": [{"date": str(d.date()), "direction": r["alarm"]} for d, r in oos.iterrows()],
                               "false_alarm_rate_per_year": len(oos) / years, "tuning_target_per_year": 1.0, "injection": inj,
                               "recent_alarms": [{"date": str(d.date()), "direction": r["alarm"]} for d, r in recent.iterrows()],
                               "last_data_date": str(y.index[-1].date())}
    live = live_evaluation(wide)
    out["live"] = {"n_forecast_files": len(list(FORECASTS.glob("*.csv"))), "status": "no live forecasts have matured yet"}
    bt_path = PROCESSED / "backtest.csv"
    if len(live) and bt_path.exists():
        bt = pd.read_csv(bt_path, parse_dates=["origin", "date"])
        ens = live[live["model"] == "ensemble"]
        recent_wape = float((ens["y_pred"] - ens["y_true"]).abs().sum() / ens["y_true"].sum())
        ref = pd.concat([metrics.origin_wapes(bt, s, "ensemble") for s in cfg["series"]])
        out["live"] = {**out["live"], **drift.performance_status(recent_wape, ref), "n_scored_forecasts": int(len(ens)), "status": "scored"}
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "monitor.json").write_text(json.dumps(out, indent=1, default=float))
    for name, v in out["series"].items():
        print(f"{name:7s} h={v['h']:.1f}  out-of-sample false alarms {len(v['alarms_out_of_sample'])} ({v['false_alarm_rate_per_year']:.1f}/yr)  injection:", {f"{i['delta']:+.2f}": (round(i['detected_share'], 2), i['median_delay_days']) for i in v["injection"]})


def cmd_track(args: argparse.Namespace) -> None:
    cfg = load_config()
    bt = metrics.annotate(pd.read_csv(PROCESSED / "backtest.csv", parse_dates=["origin", "date"]))
    s = metrics.summary(bt, cfg["reference_model"]).merge(metrics.interval_metrics(bt, cfg["interval_level"])[["series", "model", "coverage", "rel_width"]], on=["series", "model"])
    n = tracking.log_runs(s, {"horizon": cfg["horizon"], "train_start": cfg["train_start"], "first_origin": cfg["backtest"]["first_origin"],
                              "step_days": cfg["backtest"]["step_days"], "interval_level": cfg["interval_level"]},
                          kind="backtest", data_end=str(bt["date"].max().date()), tracking_uri=MLFLOW_URI)
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"logged {n} backtest runs; exported {tracking.export_runs(MLFLOW_URI, ASSETS / 'mlflow_runs.json')} runs to docs/assets/mlflow_runs.json")


def cmd_ablation(args: argparse.Namespace) -> None:
    """Does skipping the 2020-21 collapse help? Same backtest, different training start dates."""
    cfg = load_config()
    wide = data.to_wide(data.load_raw())
    rows = []
    for name in ["Subway", "Bus"]:
        y = data.get_series(wide, name)
        origins = bt_mod.make_origins(y, cfg["backtest"]["first_origin"], cfg["backtest"]["step_days"], cfg["horizon"])
        for start in [None, "2021-06-01", "2022-06-01"]:
            bt = bt_mod.run_backtest(y, origins, cfg["horizon"], ["prophet", "xgboost"], train_start=start, workers=args.workers, log=lambda *_: None)
            for model, g in bt.groupby("model"):
                rows.append({"series": name, "model": model, "train_start": start or "all history (2020-03)", "wape": metrics.wape(g), "n": len(g)})
            print(f"{name} train_start={start}: done", flush=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "ablation.json").write_text(json.dumps(rows, indent=1))


def cmd_weekly(args: argparse.Namespace) -> None:
    cmd_fetch(args)
    cmd_forecast(args)
    cmd_monitor(args)
    tracking.export_runs(MLFLOW_URI, ASSETS / "mlflow_runs.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--series", nargs="*")
    b.add_argument("--workers", type=int, default=4)
    b.set_defaults(fn=cmd_backtest)
    ab = sub.add_parser("ablation")
    ab.add_argument("--workers", type=int, default=4)
    ab.set_defaults(fn=cmd_ablation)
    for name, fn in (("fetch", cmd_fetch), ("forecast", cmd_forecast), ("monitor", cmd_monitor), ("track", cmd_track), ("weekly", cmd_weekly)):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
