import numpy as np
import pandas as pd
import pytest

from energy_forecast.ou import estimate_ou, expected_average


def test_analytic_expectation_matches_numerical_integral():
    kappa, theta, end, offset, horizon = 0.12, 55, 100, 1, 24
    times = np.linspace(offset, offset + horizon, 100001)
    path = theta + (end - theta) * np.exp(-kappa * times)
    # Midpoint integration works across the supported NumPy versions.
    numerical = np.mean((path[1:] + path[:-1]) / 2)
    assert expected_average(kappa, theta, end, horizon, offset) == pytest.approx(
        numerical, rel=1e-9
    )


def test_zero_and_small_kappa_limit():
    assert expected_average(0, 55, -3) == -3
    assert expected_average(1e-14, 55, -3) == pytest.approx(-3, abs=1e-9)
    assert np.isnan(expected_average(-0.1, 55, 100))


def test_euler_drift_recovery_and_origin_cutoff():
    # A deterministic Euler path has exactly known regression coefficients.
    values = [110.0]
    for _ in range(80):
        values.append(values[-1] + 0.08 * (65 - values[-1]))
    prices = pd.Series(values, index=pd.date_range("2024-01-01", periods=81, freq="h", tz="UTC"))
    origin = prices.index[-1] + pd.Timedelta(hours=1)
    fit = estimate_ou(prices, origin)
    assert fit.status == "ok"
    assert fit.kappa_per_hour == pytest.approx(0.08)
    assert fit.theta == pytest.approx(65)
    extra = pd.Series([1e6], index=[origin])
    assert estimate_ou(pd.concat([prices, extra]), origin) == fit


def test_gaps_are_excluded_from_regression():
    index = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    values = [110.0]
    for _ in range(79):
        values.append(values[-1] + 0.08 * (65 - values[-1]))
    prices = pd.Series(values, index=index).drop(index[30:33])
    fit = estimate_ou(prices, index[-1] + pd.Timedelta(hours=1))
    assert fit.kappa_per_hour == pytest.approx(0.08)
    assert fit.n_pairs == 75


def test_constant_and_explosive_series_are_invalid():
    index = pd.date_range("2024-01-01", periods=70, freq="h", tz="UTC")
    origin = index[-1] + pd.Timedelta(hours=1)
    assert estimate_ou(pd.Series(20.0, index=index), origin).status == "constant_prices"
    explosive = pd.Series(1.02 ** np.arange(70), index=index)
    assert estimate_ou(explosive, origin).status == "non_mean_reverting"
