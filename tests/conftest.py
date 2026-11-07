import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def weekly_series():
    """Three years of synthetic daily data: weekday pattern, slow trend, 3% noise."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=1100, freq="D")
    pattern = np.array([1.0, 1.05, 1.05, 1.05, 1.0, 0.65, 0.5])
    y = 1_000_000 * pattern[idx.day_of_week] * (1 + 0.0002 * np.arange(len(idx))) * np.exp(rng.normal(0, 0.03, len(idx)))
    return pd.Series(y, index=idx, name="Synthetic")
