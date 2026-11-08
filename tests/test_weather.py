import numpy as np
import pandas as pd
import pytest

from ridership import drift, weather


def noaa_frame():
    return pd.DataFrame({"STATION": ["USW1"] * 4, "DATE": ["2026-01-24", "2026-01-25", "2026-01-26", "2026-01-28"],
                         "PRCP": ["0.00", "1.80", "0.02", "0.10"], "SNOW": ["0.0", "11.4", "0.0", "0.0"],
                         "SNWD": ["0.0", "1.0", None, "5.0"], "TMAX": ["17", "22", "27", "30"], "TMIN": ["9", "10", "17", "20"]})


def test_parse_fills_gaps_and_keeps_units():
    w = weather.parse(noaa_frame())
    assert len(w) == 5 and list(w.columns) == ["prcp", "snow", "snwd", "tmax", "tmin"]    # Jan 27 added as a missing day
    assert w.loc["2026-01-25", "snow"] == 11.4
    assert w.loc["2026-01-26", "snwd"] == 1.0                    # missing depth carried forward
    assert w.loc["2026-01-27", "snow"] == 0.0 and w.loc["2026-01-27", "prcp"] == 0.0


def test_save_and_reload_roundtrip(tmp_path):
    w = weather.parse(noaa_frame())
    weather.save(w, tmp_path / "w.csv")
    back = weather.load(tmp_path / "w.csv")
    pd.testing.assert_frame_equal(w, back, check_freq=False)


def test_snow_spline_and_baseline_difference():
    w = weather.parse(noaa_frame())
    f = weather.features(w)
    assert f.loc["2026-01-25", "snow_light"] == 3.0 and f.loc["2026-01-25", "snow_heavy"] == pytest.approx(8.4)
    assert f.loc["2026-01-24", "cold"] == 8.0 and f["heat"].sum() == 0
    idx = pd.date_range("2026-01-01", periods=60)
    flat = pd.DataFrame({"x": np.zeros(60)}, index=idx)
    flat.loc["2026-02-05", "x"] = 10.0
    d = weather.baseline_difference(flat)
    assert d.loc["2026-02-05", "x"] == 10.0 and d.loc["2026-02-12", "x"] == pytest.approx(-2.5)    # a storm one week earlier shifts the baseline


def test_robust_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = 0.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(0, 0.3, 2000)
    beta, se = weather.ols_robust(y, X)
    assert np.allclose(beta, [0.5, 2.0, -1.0], atol=0.05) and (se < 0.05).all()


def test_weather_adjustment_removes_a_storm_effect_from_the_drift_signal():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2021-01-04", periods=1300, freq="D")
    pattern = np.array([1.0, 1.05, 1.05, 1.05, 1.0, 0.65, 0.5])
    snow = pd.Series(0.0, index=idx)
    storm_days = pd.to_datetime(["2021-12-14", "2022-03-08", "2023-03-14", "2024-03-12"])      # ordinary Tuesdays, not next to a holiday
    snow.loc[storm_days] = 10.0
    y = pd.Series(1e6 * pattern[idx.day_of_week] * np.where(snow > 0, 0.6, 1.0) * np.exp(rng.normal(0, 0.02, len(idx))), index=idx)
    f = weather.baseline_difference(weather.features(pd.DataFrame({"prcp": 0.0, "snow": snow, "snwd": 0.0, "tmax": 50.0, "tmin": 40.0}, index=idx)))
    s = drift.surprise(y)
    ok = s.notna() & f.notna().all(axis=1)
    beta, _ = weather.ols_robust(s[ok].values, f.loc[ok, ["snow_light", "snow_heavy"]].values)
    effect = pd.Series(f[["snow_light", "snow_heavy"]].values @ beta[1:], index=idx)
    assert abs(s.loc["2024-03-12"]) > 0.3                                       # the storm is a big raw surprise
    assert abs((s - effect).loc["2024-03-12"]) < 0.15                           # and mostly explained afterwards


def test_weather_event_rule():
    idx = pd.date_range("2026-01-20", periods=8)
    w = pd.DataFrame({"prcp": 0.0, "snow": 0.0, "snwd": 0.0, "tmax": 40.0, "tmin": 30.0}, index=idx)
    w.loc["2026-01-22", "snow"] = 5.0                         # storm on the 22nd
    w.loc["2026-01-23", "snwd"] = 5.0
    w.loc["2026-01-26", "prcp"] = 2.0                         # a heavy-rain day
    ev = weather.is_weather_event(w)
    assert ev["2026-01-22"] and ev["2026-01-23"]               # the day itself and the day after
    assert not ev["2026-01-21"] and not ev["2026-01-25"]
    assert ev["2026-01-26"] and ev["2026-01-27"] and not ev["2026-01-24"]
