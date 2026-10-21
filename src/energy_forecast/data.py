"""Explicit CSV schema and regular UTC observation-grid validation."""

from pathlib import Path

import numpy as np
import pandas as pd


def _column(columns, requested, aliases, kind):
    if requested is not None:
        if requested not in columns:
            raise ValueError(f"{kind} column {requested!r} not found")
        return requested
    matches = [c for c in columns if str(c).strip().lower() in aliases]
    if len(matches) != 1:
        raise ValueError(f"Specify the {kind} column explicitly; found candidates: {matches}")
    return matches[0]


def validate_prices(prices: pd.Series, cadence_minutes: int = 60) -> pd.Series:
    """Return a sorted copy; gaps are allowed, duplicates/off-grid rows are errors."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError("Prices require a timezone-aware DatetimeIndex")
    if prices.empty or prices.index.hasnans:
        raise ValueError("Prices must contain valid timestamps")
    if cadence_minutes <= 0 or cadence_minutes > 720 or 1440 % cadence_minutes:
        raise ValueError("cadence_minutes must divide 1440 and be at most 720")
    result = pd.to_numeric(prices, errors="raise").astype(float).sort_index()
    result.index = result.index.tz_convert("UTC")
    result.name = "price"
    if result.index.has_duplicates:
        raise ValueError("Duplicate timestamps: resolve duplicates before forecasting")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Prices must be finite numbers; remove or repair missing rows explicitly")
    grid = pd.Timedelta(minutes=cadence_minutes)
    if ((result.index - result.index.normalize()) % grid != pd.Timedelta(0)).any():
        raise ValueError("Timestamps must align to the declared cadence from UTC midnight")
    return result


def load_prices(
    path: str | Path,
    timestamp_col: str | None = None,
    price_col: str | None = None,
    cadence_minutes: int = 60,
) -> pd.Series:
    """Load ISO 8601 timestamps; naive strings are explicitly interpreted as UTC."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("CSV is empty")
    timestamp = _column(
        frame.columns, timestamp_col, {"timestamp", "datetime", "date", "time"}, "timestamp"
    )
    price = _column(frame.columns, price_col, {"price", "smp", "market_price"}, "price")
    if timestamp == price or pd.api.types.is_numeric_dtype(frame[timestamp]):
        raise ValueError("Timestamp column must contain date/time strings, not numeric identifiers")
    times = pd.to_datetime(frame[timestamp], utc=True, errors="raise", format="mixed")
    values = pd.to_numeric(frame[price], errors="raise").to_numpy()
    return validate_prices(pd.Series(values, index=pd.DatetimeIndex(times)), cadence_minutes)
