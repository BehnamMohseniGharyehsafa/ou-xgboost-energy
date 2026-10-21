import numpy as np
import pandas as pd

from energy_forecast.features import build_daily_features


def test_target_and_future_prices_cannot_change_features(prices):
    origin = prices.index[60 * 24]
    original = build_daily_features(prices)
    changed = prices.copy()
    changed.loc[origin:] = changed.loc[origin:] * 100 - 9000
    altered = build_daily_features(changed)
    pd.testing.assert_series_equal(original.loc[origin].drop("y"), altered.loc[origin].drop("y"))
    assert original.loc[origin, "y"] != altered.loc[origin, "y"]


def test_adding_future_data_preserves_every_existing_feature(prices):
    cutoff = prices.index[80 * 24]
    before = build_daily_features(prices.loc[prices.index < cutoff]).drop(columns="y")
    after = build_daily_features(prices).loc[before.index].drop(columns="y")
    pd.testing.assert_frame_equal(before, after)


def test_negative_and_zero_prices_remain_finite(prices):
    prices.iloc[10:15] = [-20, -10, 0, 5, -1]
    daily = build_daily_features(prices)
    assert np.isfinite(daily.dropna().to_numpy()).all()
    assert len(daily.dropna()) == 105 - 28


def test_missing_days_preserve_calendar_lags(prices):
    gap = prices.index[40 * 24].normalize()
    prices = prices.loc[prices.index.normalize() != gap]
    daily = build_daily_features(prices)
    assert pd.isna(daily.loc[gap, "y"])
    assert pd.isna(daily.loc[gap + pd.Timedelta(days=1), "y_lag1"])
    assert pd.isna(daily.loc[gap + pd.Timedelta(days=7), "y_lag7"])


def test_partial_final_day_has_no_target(prices):
    daily = build_daily_features(prices.iloc[:-1])
    assert pd.isna(daily.iloc[-2]["y"])
    assert pd.isna(daily.iloc[-1]["last_price_lag1"])
