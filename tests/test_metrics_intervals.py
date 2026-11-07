import numpy as np
import pandas as pd
import pytest

from ridership import intervals, metrics


def make_bt(n_origins=30, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_origins):
        o = pd.Timestamp("2024-01-01") + pd.Timedelta(days=28 * i)
        for h in range(1, 15):
            truth = 1_000_000 * (1 + rng.normal(0, noise))
            rows.append({"series": "S", "model": "good", "origin": o, "date": o + pd.Timedelta(days=h), "h": h, "y_true": truth, "y_pred": 1_000_000})
            rows.append({"series": "S", "model": "bad", "origin": o, "date": o + pd.Timedelta(days=h), "h": h, "y_true": truth, "y_pred": 1_150_000})
    return pd.DataFrame(rows)


def test_wape_and_summary_skill():
    df = pd.DataFrame({"y_true": [100.0, 200.0], "y_pred": [110.0, 180.0]})
    assert metrics.wape(df) == pytest.approx(30 / 300)
    s = metrics.summary(make_bt(), reference="bad").set_index("model")
    assert s.loc["bad", "skill_vs_ref"] == pytest.approx(0.0) and s.loc["good", "skill_vs_ref"] > 0.5


def test_origin_bootstrap_detects_a_clearly_better_model_and_a_tie():
    bt = make_bt()
    r = metrics.origin_bootstrap(bt, "S", "good", "bad")
    assert r["wape_diff"] < 0 and r["hi"] < 0 and r["p_value"] < 0.01 and r["n_origins"] == 30
    tie = metrics.origin_bootstrap(bt, "S", "good", "good")
    assert tie["wape_diff"] == 0 and tie["p_value"] == 1.0


def test_intervals_reach_nominal_coverage_without_peeking():
    bt = intervals.add_intervals(make_bt(n_origins=60, noise=0.05), level=0.8)
    cov = metrics.interval_metrics(bt[bt["model"] == "good"], 0.8)
    assert cov["coverage"].iloc[0] == pytest.approx(0.8, abs=0.06)
    assert bt[bt["origin"] < bt["origin"].min() + pd.Timedelta(days=28 * 4)]["lo"].isna().all()   # warm-up origins get no interval


def test_intervals_ignore_future_origins():
    bt = make_bt(n_origins=20)
    a = intervals.add_intervals(bt)
    changed = bt.copy()
    changed.loc[changed["origin"] == changed["origin"].max(), "y_true"] *= 3                     # distort only the LAST origin
    b = intervals.add_intervals(changed)
    early = a["origin"] < a["origin"].max()
    pd.testing.assert_series_equal(a.loc[early, "lo"], b.loc[early, "lo"])
    pd.testing.assert_series_equal(a.loc[early, "hi"], b.loc[early, "hi"])


def test_final_quantiles_and_apply():
    bt = make_bt()
    q = intervals.final_interval_quantiles(bt, 0.8)
    fc = pd.DataFrame({"series": ["S"] * 2, "model": ["good"] * 2, "h": [1, 10], "forecast": [1_000_000.0, 1_000_000.0]})
    out = intervals.apply_final(fc, q)
    assert (out["lo"] < out["forecast"]).all() and (out["hi"] > out["forecast"]).all()


def test_backtest_origins_cover_every_weekday(weekly_series):
    from ridership.backtest import make_origins

    origins = make_origins(weekly_series, "2023-01-01", 29, 14)
    assert len({o.day_of_week for o in origins}) == 7
    assert origins[-1] + pd.Timedelta(days=14) <= weekly_series.index[-1]
    assert len({o.day_of_week for o in make_origins(weekly_series, "2023-01-01", 28, 14)}) == 1    # the flaw this guards against
