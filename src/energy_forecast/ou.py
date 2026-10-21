"""Windowed Euler drift estimation and an analytic OU conditional expectation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OUFit:
    kappa_per_hour: float
    theta: float
    end_price: float
    last_timestamp: pd.Timestamp
    n_pairs: int
    status: str


def estimate_ou(
    prices: pd.Series, origin: pd.Timestamp, window_days: int = 60, cadence_minutes: int = 60
) -> OUFit:
    """Fit dP/dt = a - kappa*P by dt-weighted least squares on past data.

    Exclude gap-spanning pairs and, with >100 pairs, the largest 1% absolute
    changes. This is an approximate drift estimator, not an exact OU MLE.
    """
    segment = prices.loc[
        (prices.index < origin) & (prices.index >= origin - pd.Timedelta(days=window_days))
    ]
    end_price = float(segment.iloc[-1]) if len(segment) else np.nan
    last_timestamp = segment.index[-1] if len(segment) else pd.NaT

    def invalid(status, n=0):
        return OUFit(np.nan, np.nan, end_price, last_timestamp, n, status)

    if len(segment) < 21:
        return invalid("insufficient_pairs")
    dt = np.asarray(np.diff(segment.index) / pd.Timedelta(hours=1), dtype=float)
    delta = np.diff(segment.to_numpy())
    previous = segment.to_numpy()[:-1]
    keep = (dt > 0) & (dt <= 1.5 * cadence_minutes / 60)
    if keep.sum() > 100:
        threshold = np.quantile(np.abs(delta[keep]), 0.99)
        keep &= np.abs(delta) <= threshold
    dt, delta, previous = dt[keep], delta[keep], previous[keep]
    n = len(previous)
    if n < 20:
        return invalid("insufficient_pairs", n)
    if np.ptp(previous) < 1e-10:
        return invalid("constant_prices", n)
    design = np.column_stack([np.ones(n), previous])
    weights = np.sqrt(dt)
    intercept, slope = np.linalg.lstsq(design * weights[:, None], delta / dt * weights, rcond=None)[
        0
    ]
    kappa = -float(slope)
    if not np.isfinite(kappa) or kappa <= 1e-10:
        return invalid("non_mean_reverting", n)
    theta = float(intercept / kappa)
    if not np.isfinite(theta):
        return invalid("non_finite_level", n)
    return OUFit(kappa, theta, end_price, last_timestamp, n, "ok")


def expected_average(
    kappa_per_hour: float,
    theta: float,
    end_price: float,
    horizon_hours: float = 24,
    offset_hours: float = 0,
) -> float:
    """Expected continuous average over [offset, offset + horizon] from last sample.

    Use expm1 for stability; kappa=0 has the well-defined constant expectation P0.
    A negative speed is invalid rather than silently clamped to a positive value.
    """
    if not np.isfinite([kappa_per_hour, theta, end_price, horizon_hours, offset_hours]).all():
        return np.nan
    if kappa_per_hour < 0 or horizon_hours <= 0 or offset_hours < 0:
        return np.nan
    x = kappa_per_hour * horizon_hours
    factor = 1.0 if x == 0 else -np.expm1(-x) / x
    return float(theta + (end_price - theta) * np.exp(-kappa_per_hour * offset_hours) * factor)
