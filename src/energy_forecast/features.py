"""Feature construction that is invariant to changes at/after the forecast origin."""

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14, 21, 28)
WINDOWS = (3, 7, 14, 28)


def build_daily_features(prices: pd.Series, cadence_minutes: int = 60) -> pd.DataFrame:
    """Include one future row, preserve calendar gaps, mask incomplete UTC days.

    The input must already pass ``validate_prices``. The target is the arithmetic
    mean on an equally spaced grid. All observed-price features shift by one day.
    """
    daily = prices.resample("D").agg(["mean", "last", "max", "min", "count"])
    daily = daily.reindex(pd.date_range(daily.index[0], daily.index[-1] + pd.Timedelta(days=1)))
    complete = daily["count"].eq(1440 // cadence_minutes)
    state = pd.DataFrame(index=daily.index)
    state["last_price"] = daily["last"]
    state["day_high"] = daily["max"]
    state["day_low"] = daily["min"]
    state["day_range"] = daily["max"] - daily["min"]
    # Price differences support zero and negative electricity prices. No overnight jump.
    same_day = prices.index.normalize().to_series(index=prices.index).diff().eq(pd.Timedelta(0))
    changes = prices.diff().where(same_day)
    state["price_change_std"] = changes.resample("D").std(ddof=0).reindex(daily.index)
    state = state.where(complete)
    result = pd.DataFrame({"y": daily["mean"].where(complete)}, index=daily.index)
    for column in state:
        result[f"{column}_lag1"] = state[column].shift(1)
    for lag in LAGS:
        result[f"y_lag{lag}"] = result["y"].shift(lag)
    history = result["y"].shift(1)
    for window in WINDOWS:
        result[f"y_rollmean{window}"] = history.rolling(window).mean()
        result[f"y_rollstd{window}"] = history.rolling(window).std()
    dow = result.index.dayofweek
    month = result.index.month
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    result["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    result.index.name = "date"
    return result
