"""Seeded artificial prices for demonstrations; no real market data."""

import numpy as np
import pandas as pd


def make_synthetic_prices(days: int = 365, seed: int = 42, cadence_minutes: int = 60) -> pd.Series:
    """Seasonal level, mean-reverting noise and occasional signed spikes."""
    if days < 1 or cadence_minutes < 1 or 1440 % cadence_minutes:
        raise ValueError("Use positive days and a cadence that divides 1440")
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        "2024-01-01",
        periods=days * 1440 // cadence_minutes,
        freq=f"{cadence_minutes}min",
        tz="UTC",
        name="timestamp",
    )
    t = np.arange(len(index)) * cadence_minutes / 60
    level = 65 + 15 * np.cos(2 * np.pi * t / (24 * 365))
    level += 9 * np.sin(2 * np.pi * (t - 6) / 24) - 8 * (index.dayofweek >= 5)
    phi = np.exp(-0.18 * cadence_minutes / 60)
    noise = np.zeros(len(index))
    innovations = rng.normal(0, 8 * np.sqrt(1 - phi**2), size=len(index))
    for i in range(1, len(index)):
        noise[i] = phi * noise[i - 1] + innovations[i]
    spikes = rng.binomial(1, 0.008, len(index)) * rng.normal(0, 90, len(index))
    return pd.Series(np.asarray(level) + noise + spikes, index=index, name="price")
