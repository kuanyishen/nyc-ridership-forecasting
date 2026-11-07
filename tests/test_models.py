import numpy as np
import pandas as pd

from ridership import models


def test_baselines_follow_the_weekly_cycle(weekly_series):
    hist = weekly_series.iloc[:500]
    sn = models.seasonal_naive(hist, 14)
    assert len(sn) == 14 and sn.index[0] == hist.index[-1] + pd.Timedelta(days=1)
    assert sn.iloc[0] == hist.loc[sn.index[0] - pd.Timedelta(days=7)]
    assert sn.iloc[8] == hist.loc[sn.index[8] - pd.Timedelta(days=14)]        # beyond 7 days: most recent same weekday
    med = models.seasonal_median4(hist, 7)
    assert med.iloc[0] == np.median([hist.loc[med.index[0] - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4)])
    assert (models.naive(hist, 5) == hist.iloc[-1]).all()


def test_features_never_use_data_after_the_origin(weekly_series):
    y = weekly_series.values.copy()
    dates = weekly_series.index
    o = np.array([600])
    for h in (1, 7, 14):
        f1, b1 = models._features_for(y, dates, o, h)
        y2 = y.copy()
        y2[601:] = 1e12                                                       # corrupt everything after the origin
        f2, b2 = models._features_for(y2, dates, o, h)
        pd.testing.assert_frame_equal(f1, f2)
        assert b1 == b2


def test_xgboost_beats_naive_on_a_weekly_series_and_is_deterministic(weekly_series):
    hist, actual = weekly_series.iloc[:900], weekly_series.iloc[900:914]
    f1 = models.xgboost_forecast(hist, 14, n_estimators=60)
    f2 = models.xgboost_forecast(hist, 14, n_estimators=60)
    assert np.allclose(f1.values, f2.values)
    err = lambda f: (f - actual).abs().sum() / actual.sum()
    assert err(f1) < 0.5 * err(models.naive(hist, 14)) and err(f1) < 0.08


def test_run_models_ensemble_is_the_mean_of_prophet_and_xgboost(weekly_series):
    hist = weekly_series.iloc[:700]
    out = models.run_models(hist, 7, ["seasonal_naive", "prophet", "xgboost", "ensemble"], train_start="2022-06-01")
    assert list(out) == ["seasonal_naive", "prophet", "xgboost", "ensemble"]
    assert np.allclose(out["ensemble"].values, (out["prophet"].values + out["xgboost"].values) / 2)
