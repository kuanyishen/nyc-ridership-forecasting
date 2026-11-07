"""Load, clean and update the MTA Daily Ridership and Traffic data.

The source is long format (Date, Mode, Count). Counts arrive as text with thousands separators, the newest days are
often missing for some modes (publication lag), and a few modes have gaps, so nothing here assumes a complete grid.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "mta_daily_ridership.csv"
API_URL = "https://data.ny.gov/resource/sayj-mze2.json"   # same dataset as the CSV export (MTA Daily Ridership and Traffic: Beginning 2020)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"].astype(str).str.slice(0, 19), format="mixed").dt.normalize(),
        "mode": df["mode"].astype(str).str.strip(),
        "count": pd.to_numeric(df["count"].astype(str).str.replace(",", "", regex=False), errors="coerce"),
    })
    return out.drop_duplicates(["date", "mode"], keep="last").sort_values(["mode", "date"]).reset_index(drop=True)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Long table with columns date (datetime), mode (str), count (float, NaN where missing)."""
    return _tidy(pd.read_csv(path))


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One column per mode on a complete daily index."""
    wide = long.pivot(index="date", columns="mode", values="count").sort_index()
    return wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))


def get_series(wide: pd.DataFrame, mode: str) -> pd.Series:
    """One mode, trimmed to its last published value (the tail is NaN while the MTA catches up)."""
    s = wide[mode].astype(float)
    last = s.last_valid_index()
    s = s.loc[:last]
    s.index.name = "date"
    s.name = mode
    return s


def parse_api_records(records: list[dict]) -> pd.DataFrame:
    """Turn Socrata JSON rows (lowercase keys, ISO timestamps, counts as text) into the same long table."""
    return _tidy(pd.DataFrame(records))


def fetch_latest(limit: int = 100_000, timeout: float = 60.0) -> pd.DataFrame:
    url = f"{API_URL}?$limit={limit}&$order=date"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
        return parse_api_records(json.load(r))


def merge_update(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union of two long tables; where both have a date+mode the NEW value wins (late revisions and filled gaps)."""
    both = pd.concat([existing, new], ignore_index=True)
    return both.drop_duplicates(["date", "mode"], keep="last").sort_values(["mode", "date"]).reset_index(drop=True)


def save_raw(long: pd.DataFrame, path: Path = RAW_PATH) -> None:
    out = long.assign(date=long["date"].dt.strftime("%Y-%m-%d"), count=long["count"].round().astype("Int64"))
    out.rename(columns={"date": "Date", "mode": "Mode", "count": "Count"}).to_csv(path, index=False)
