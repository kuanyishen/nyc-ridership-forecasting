"""NOAA daily weather for Central Park (station USW00094728): precipitation, snowfall, snow depth, temperatures.

Units follow NOAA 'standard': inches and degrees Fahrenheit. Weather is used in two different ways in this project and
the difference matters:
* DIAGNOSIS (observed weather, after the fact): explain why a day was unusually low, and discount weather-driven alarms.
* FORECASTING: the weather of a future day is NOT known 14 days ahead, so using observed weather as a model input is an
  oracle experiment (an upper bound), not something a production forecast can do.
"""

from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "weather_central_park.csv"
STATION = "USW00094728"
NOAA_URL = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations={station}&startDate={start}&endDate={end}"
            "&dataTypes=PRCP,SNOW,SNWD,TMAX,TMIN&format=csv&units=standard")
COLS = {"PRCP": "prcp", "SNOW": "snow", "SNWD": "snwd", "TMAX": "tmax", "TMIN": "tmin"}


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """NOAA csv -> daily frame indexed by date with columns prcp, snow, snwd, tmax, tmin (floats)."""
    out = df.rename(columns={c: c.strip().upper() for c in df.columns})
    out = out.assign(date=pd.to_datetime(out["DATE"]).dt.normalize()).set_index("date").sort_index()
    out = out[[c for c in COLS if c in out.columns]].apply(pd.to_numeric, errors="coerce").rename(columns=COLS)
    out = out[~out.index.duplicated(keep="last")]
    full = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="D"))
    full["snwd"] = full["snwd"].ffill(limit=3).fillna(0.0)          # snow depth persists; a few missing days
    for c in ("prcp", "snow"):
        full[c] = full[c].fillna(0.0)                              # no report means no precipitation
    full.index.name = "date"
    return full


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    return parse(pd.read_csv(path))


def fetch_latest(start: str = "2020-03-01", end: str | None = None, timeout: float = 60.0) -> pd.DataFrame:
    url = NOAA_URL.format(station=STATION, start=start, end=end or date.today().isoformat())
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return parse(pd.read_csv(r))


def save(w: pd.DataFrame, path: Path = RAW_PATH) -> None:
    out = w.reset_index().rename(columns={v: k for k, v in COLS.items()}).rename(columns={"date": "DATE"})
    out.insert(0, "STATION", STATION)
    out["DATE"] = out["DATE"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def features(w: pd.DataFrame) -> pd.DataFrame:
    """Weather variables used for diagnosis, with a linear spline on snow (a first inch matters less than a foot)."""
    f = pd.DataFrame(index=w.index)
    f["snow_light"] = w["snow"].clip(upper=3.0)
    f["snow_heavy"] = (w["snow"] - 3.0).clip(lower=0.0)
    f["snwd"] = w["snwd"].clip(upper=12.0)
    f["prcp"] = w["prcp"]
    f["cold"] = (25.0 - w["tmax"]).clip(lower=0.0)                 # degrees F the high stayed below 25
    f["heat"] = (w["tmax"] - 90.0).clip(lower=0.0)
    return f


def baseline_difference(f: pd.DataFrame, lags: tuple[int, ...] = (7, 14, 21, 28)) -> pd.DataFrame:
    """Weather of day t minus the average weather on the four same-weekdays used as the ridership baseline."""
    base = sum(f.shift(l) for l in lags) / len(lags)
    return f - base


def ols_robust(y: np.ndarray, X: np.ndarray, return_cov: bool = False):
    """OLS with an intercept and heteroskedasticity-robust (HC1) standard errors. Returns (coefficients, std errors),
    or (coefficients, covariance matrix) when ``return_cov``."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    bread = np.linalg.pinv(A.T @ A)
    meat = (A * resid[:, None]).T @ (A * resid[:, None])
    cov = bread @ meat @ bread * len(y) / max(len(y) - A.shape[1], 1)
    return (beta, cov) if return_cov else (beta, np.sqrt(np.diag(cov)))


def is_weather_event(w: pd.DataFrame) -> pd.Series:
    """True on days a storm plausibly explains unusual ridership: snow today or yesterday (>= 1 in), at least 3 in of snow on the
    ground, or heavy rain (>= 1.5 in) today or yesterday. A transparent rule, not a fitted model: a regression adjustment did
    not predict the day AFTER a storm, so alarms are labelled rather than silently removed."""
    return ((w["snow"] >= 1) | (w["snow"].shift(1) >= 1) | (w["snwd"] >= 3) | (w["prcp"] >= 1.5) | (w["prcp"].shift(1) >= 1.5)).fillna(False)
