from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from energy_forecast import ForecastConfig, backtest, forecast_next_day
from energy_forecast.evaluation import score_predictions

CONFIG = ForecastConfig(backend="hist", min_train_days=40, n_estimators=12, n_jobs=1)


def test_backtest_chronology_and_refit_schedule(prices):
    frame = backtest(prices, CONFIG).predictions
    assert len(frame) == 105 - 28 - 40
    assert (frame["trained_through"] < frame.index).all()
    assert (frame["last_observed_at"] < frame.index).all()
    assert frame.iloc[0]["train_rows"] == 40
    assert frame.iloc[6]["train_rows"] == 40
    assert frame.iloc[7]["train_rows"] == 47
    assert np.isfinite(frame[["yhat_ml", "yhat_blend"]].to_numpy()).all()
    assert set(frame["backend"]) == {"hist"}


def test_future_perturbation_does_not_change_past_or_current_forecasts(prices):
    original = backtest(prices, CONFIG).predictions
    day = original.index[9]
    altered = prices.copy()
    altered.loc[day:] += 1e5
    changed = backtest(altered, CONFIG).predictions
    columns = ["yhat_ml", "yhat_ou", "yhat_blend", "kappa_per_hour", "P_end"]
    pd.testing.assert_frame_equal(original.loc[:day, columns], changed.loc[:day, columns])


def test_next_day_predicts_unobserved_target(prices):
    result = forecast_next_day(prices, CONFIG)
    assert len(result.predictions) == 1
    assert result.predictions.index[0] == prices.index[-1].normalize() + pd.Timedelta(days=1)
    assert "y_true" not in result.predictions
    with pytest.raises(ValueError, match="complete final"):
        forecast_next_day(prices.iloc[:-1], CONFIG)


def test_constant_prices_fall_back_to_ml(prices):
    prices[:] = -5
    frame = backtest(prices, CONFIG).predictions
    assert frame["blend_fallback"].all()
    np.testing.assert_allclose(frame["yhat_blend"], frame["yhat_ml"])
    assert frame["yhat_ou"].isna().all()
    metrics = score_predictions(frame)
    assert metrics.loc["ou", "n"] == 0
    assert metrics.loc["blend", "mae"] == 0


def test_weight_one_recovers_ml(prices):
    frame = backtest(prices, replace(CONFIG, blend_weight=1)).predictions
    np.testing.assert_allclose(frame["yhat_blend"], frame["yhat_ml"])


def test_insufficient_history_is_clear(prices):
    with pytest.raises(ValueError, match="28 warm-up"):
        backtest(prices.iloc[: 40 * 24], CONFIG)


def test_metrics_mask_missing_ou_values():
    frame = pd.DataFrame({"y_true": [0, -2, 4], "yhat_ou": [1, np.nan, 2], "yhat_ml": [0, -1, 4]})
    metrics = score_predictions(frame)
    assert metrics.loc["ou", "n"] == 2
    assert metrics.loc["ou", "mae"] == 1.5
    assert metrics.loc["ml", "n"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blend_weight": 1.01},
        {"blend_weight": np.nan},
        {"refit_every": 0},
        {"ou_window_days": -1},
        {"cadence_minutes": 17},
        {"n_jobs": 0},
        {"backend": "invalid"},
        {"device": "invalid"},
        {"backend": "hist", "device": "cuda"},
        {"seed": -1},
        {"min_train_days": True},
    ],
)
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        ForecastConfig(**kwargs)
