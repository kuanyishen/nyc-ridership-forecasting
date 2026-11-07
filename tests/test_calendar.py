import pandas as pd

from ridership import calendar


def test_known_holidays_and_distances():
    f = calendar.features(pd.DatetimeIndex(["2025-11-27", "2025-11-26", "2025-11-28", "2025-12-25", "2025-03-04"]))
    assert f.loc["2025-11-27", "is_holiday"] == 1                      # Thanksgiving
    assert f.loc["2025-11-26", "to_next_hol"] == 1 and f.loc["2025-11-28", "since_prev_hol"] == 1
    assert f.loc["2025-12-25", "is_holiday"] == 1 and f.loc["2025-03-04", "is_holiday"] == 0
    assert f.loc["2025-11-27", "dow"] == 3


def test_features_cover_future_dates_without_error():
    f = calendar.features(pd.date_range("2030-12-20", periods=15))
    assert f["is_holiday"].sum() >= 1 and f["to_next_hol"].max() <= 14
