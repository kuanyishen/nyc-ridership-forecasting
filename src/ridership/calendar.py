"""Calendar features shared by every model: weekday, holidays and distance to holidays (New York State calendar)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays as pyholidays

_YEARS = range(2019, 2032)


def holiday_dates() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in pyholidays.US(state="NY", years=_YEARS)))


def holiday_frame() -> pd.DataFrame:
    """Prophet-style holiday table with a one-day window on each side."""
    cal = pyholidays.US(state="NY", years=_YEARS)
    return pd.DataFrame({"holiday": [str(n) for n in cal.values()], "ds": pd.to_datetime(list(cal.keys())),
                         "lower_window": -1, "upper_window": 1})


def features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features for arbitrary dates."""
    dates = pd.DatetimeIndex(dates)
    hol = holiday_dates()
    day = dates.values.astype("datetime64[D]")
    hd = hol.values.astype("datetime64[D]")
    nxt = np.searchsorted(hd, day, side="left")                 # next holiday on or after the day
    prv = np.searchsorted(hd, day, side="right") - 1            # last holiday on or before the day
    to_next = np.where(nxt < len(hd), (hd[np.minimum(nxt, len(hd) - 1)] - day).astype(int), 99)
    since_prev = np.where(prv >= 0, (day - hd[np.maximum(prv, 0)]).astype(int), 99)
    doy = dates.dayofyear.values
    return pd.DataFrame({
        "dow": dates.day_of_week.values, "month": dates.month.values, "dom": dates.day.values,
        "is_holiday": (to_next == 0).astype(int), "to_next_hol": np.minimum(to_next, 14), "since_prev_hol": np.minimum(since_prev, 14),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=dates)
