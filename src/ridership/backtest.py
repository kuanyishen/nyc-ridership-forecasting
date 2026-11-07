"""Rolling-origin backtest: at each origin, fit on the past only, forecast the next ``horizon`` days, score on what happened."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import models


def make_origins(y: pd.Series, first_origin: str, step_days: int, horizon: int) -> list[pd.Timestamp]:
    last_ok = y.index[-1] - pd.Timedelta(days=horizon)
    return list(pd.date_range(pd.Timestamp(first_origin), last_ok, freq=f"{step_days}D"))


def _one_origin(y: pd.Series, origin: pd.Timestamp, horizon: int, names: list[str], train_start: str | None) -> list[dict]:
    hist = y.loc[:origin]
    preds = models.run_models(hist, horizon, names, train_start=train_start)
    truth = y.reindex(preds[names[0]].index)
    rows = []
    for name, f in preds.items():
        for h, (d, yhat) in enumerate(f.items(), start=1):
            if pd.notna(truth.loc[d]) and pd.notna(yhat):
                rows.append({"series": y.name, "model": name, "origin": origin, "date": d, "h": h, "y_true": float(truth.loc[d]), "y_pred": float(yhat)})
    return rows


def run_backtest(y: pd.Series, origins: list[pd.Timestamp], horizon: int, names: list[str], *, train_start: str | None = None,
                 workers: int = 4, log=print) -> pd.DataFrame:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one_origin, y, o, horizon, names, train_start) for o in origins]
        for i, f in enumerate(futs, 1):
            rows += f.result()
            if i % 10 == 0 or i == len(futs):
                log(f"  {y.name}: {i}/{len(futs)} origins")
    return pd.DataFrame(rows)
