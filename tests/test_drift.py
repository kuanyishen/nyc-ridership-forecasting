import numpy as np
import pandas as pd

from ridership import drift


def test_surprise_is_small_for_a_stable_series_and_masks_holidays(weekly_series):
    s = drift.surprise(weekly_series)
    assert abs(np.nanmean(s)) < 0.01 and np.nanstd(s) < 0.06
    assert np.isnan(s.loc["2023-11-23"]) and np.isnan(s.loc["2023-11-24"])         # Thanksgiving and the day after


def test_few_false_alarms_on_a_stable_series(weekly_series):
    h = drift.tune_h(weekly_series, 0.5, "2022-06-01", "2023-06-30", target_per_year=1.0)
    al = drift.detect(weekly_series.loc["2022-01-01":], 0.5, h)
    after = al.loc["2023-07-01":]
    years = (weekly_series.index[-1] - pd.Timestamp("2023-07-01")).days / 365.25
    # tuned for ~1 false alarm a year on the tuning window; the 4-week baseline makes the noise autocorrelated, so the
    # out-of-sample rate runs higher. What matters is that it stays a few per year, not dozens.
    assert len(after) / years <= 3.5


def test_level_shifts_are_detected_quickly_and_small_ones_more_slowly(weekly_series):
    starts = [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-06-10"), pd.Timestamp("2024-09-02")]
    big = drift.injection_test(weekly_series, -0.15, starts, 0.5, 6.0)
    small = drift.injection_test(weekly_series, -0.04, starts, 0.5, 6.0)
    assert big["detected_share"] == 1.0 and big["median_delay_days"] <= 3
    assert small["detected_share"] >= 0.3 and (small["median_delay_days"] or 99) >= big["median_delay_days"]
    up = drift.injection_test(weekly_series, +0.15, starts, 0.5, 6.0)
    assert up["detected_share"] == 1.0                                              # two-sided


def test_cusum_resets_after_alarm_and_skips_nan():
    z = pd.Series([3.0, 3.0, 3.0, np.nan, 3.0, 0.0, 0.0], index=pd.date_range("2024-01-01", periods=7))
    out = drift.cusum(z, k=0.5, h=5.0)
    assert out["alarm"].tolist().count("up") == 1 and out.loc["2024-01-03", "alarm"] == "up"
    assert out.loc["2024-01-03", "s_pos"] == 0.0                                    # reset
    assert out.loc["2024-01-04", "s_pos"] == out.loc["2024-01-03", "s_pos"]          # NaN day changes nothing


def test_performance_status_flags_only_unusual_error():
    ref = pd.Series(np.linspace(0.02, 0.06, 50))
    assert drift.performance_status(0.04, ref)["alert"] is False
    assert drift.performance_status(0.12, ref)["alert"] is True


def test_year_over_year_adjustment_cancels_recurring_calendar_effects():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-04", periods=1400, freq="D")
    pattern = np.array([1.0, 1.05, 1.05, 1.05, 1.0, 0.65, 0.5])
    xmas = ((idx.month == 12) & (idx.day >= 20)) | ((idx.month == 1) & (idx.day <= 8))          # a recurring 20-day dip
    y = pd.Series(1e6 * pattern[idx.day_of_week] * np.where(xmas, 0.6, 1.0) * np.exp(rng.normal(0, 0.02, len(idx))), index=idx)
    raw, adj = drift.surprise(y).loc["2023-01-15":], drift.adjusted_surprise(y).loc["2023-01-15":]
    window = lambda s: s[(s.index.month == 1) & (s.index.day >= 9) & (s.index.day <= 31)].abs().mean()   # the weeks AFTER the dip
    assert window(adj) < 0.6 * window(raw)
