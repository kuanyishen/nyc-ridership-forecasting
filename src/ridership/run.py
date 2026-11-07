"""Command line entry points.

    python -m ridership.run backtest      # rolling-origin backtest for every series and model (slow, ~5-10 min)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from . import backtest as bt_mod
from . import data, intervals

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--series", nargs="*")
    b.add_argument("--workers", type=int, default=4)
    b.set_defaults(fn=cmd_backtest)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
