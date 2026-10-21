"""Expanding-window backtesting and unseen next-day prediction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .data import validate_prices
from .features import build_daily_features
from .models import TreeRegressor
from .ou import estimate_ou, expected_average


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    feature_names: list[str]
    eligible_days: int
    skipped_days: int


def _prediction(prices, day, row, model, config):
    ml = float(model.predict(row.to_frame().T.to_numpy())[0])
    fit = estimate_ou(prices, day, config.ou_window_days, config.cadence_minutes)
    offset = (day - fit.last_timestamp) / pd.Timedelta(hours=1)
    conditioned = expected_average(fit.kappa_per_hour, ml, fit.end_price, offset_hours=offset)
    historical = expected_average(fit.kappa_per_hour, fit.theta, fit.end_price, offset_hours=offset)
    valid = np.isfinite(conditioned)
    blend = config.blend_weight * ml + (1 - config.blend_weight) * conditioned if valid else ml
    return {
        "date": day,
        "yhat_ml": ml,
        "yhat_ou": conditioned,
        "yhat_ou_historical": historical,
        "yhat_blend": blend,
        "yhat_persistence": float(row["y_lag1"]),
        "yhat_seasonal_naive": float(row["y_lag7"]),
        "kappa_per_hour": fit.kappa_per_hour,
        "theta_historical": fit.theta,
        "P_end": fit.end_price,
        "last_observed_at": fit.last_timestamp,
        "ou_pairs": fit.n_pairs,
        "ou_status": fit.status,
        "blend_fallback": not valid,
        "backend": model.backend,
        "device": model.device,
    }


def _prepare(prices, config):
    prices = validate_prices(prices, config.cadence_minutes)
    daily = build_daily_features(prices, config.cadence_minutes)
    features = [column for column in daily if column != "y"]
    eligible = daily.dropna(subset=["y", *features])
    return prices, daily, features, eligible


def backtest(prices: pd.Series, config: ForecastConfig | None = None) -> ForecastResult:
    """Predict each complete eligible day using strictly earlier training labels.

    Refit on the first test day and every ``refit_every`` calendar days thereafter.
    Daily features and OU state update at each origin even when the tree is reused.
    """
    config = config or ForecastConfig()
    prices, daily, features, eligible = _prepare(prices, config)
    if len(eligible) <= config.min_train_days:
        raise ValueError(
            f"Need more than {config.min_train_days} eligible training days; got {len(eligible)}. "
            "Allow 28 warm-up days plus the training period and at least one test day. "
            "Missing/incomplete days also remove dependent feature rows."
        )
    model = TreeRegressor(config)
    last_refit = None
    records = []
    for position in range(config.min_train_days, len(eligible)):
        day = eligible.index[position]
        if last_refit is None or (day - last_refit).days >= config.refit_every:
            train = eligible.iloc[:position]
            model.fit(train[features].to_numpy(), train["y"].to_numpy())
            last_refit = day
            trained_through = train.index[-1]
            train_rows = len(train)
        record = _prediction(prices, day, daily.loc[day, features], model, config)
        record.update(
            y_true=float(daily.loc[day, "y"]),
            trained_through=trained_through,
            train_rows=train_rows,
        )
        records.append(record)
    predictions = pd.DataFrame.from_records(records).set_index("date")
    observed_days = len(daily) - 1
    return ForecastResult(predictions, features, len(eligible), observed_days - len(eligible))


def forecast_next_day(prices: pd.Series, config: ForecastConfig | None = None) -> ForecastResult:
    """Forecast the UTC day after the last supplied day; require a complete final day."""
    config = config or ForecastConfig()
    prices, daily, features, eligible = _prepare(prices, config)
    day = daily.index[-1]
    if daily.iloc[-1][features].isna().any() or pd.isna(daily.iloc[-2]["y"]):
        raise ValueError("Next-day forecasting requires a complete final UTC day and lag history")
    if len(eligible) < config.min_train_days:
        raise ValueError(
            f"Need at least {config.min_train_days} eligible training days; got {len(eligible)}"
        )
    model = TreeRegressor(config)
    model.fit(eligible[features].to_numpy(), eligible["y"].to_numpy())
    record = _prediction(prices, day, daily.loc[day, features], model, config)
    record.update(trained_through=eligible.index[-1], train_rows=len(eligible))
    predictions = pd.DataFrame.from_records([record]).set_index("date")
    return ForecastResult(predictions, features, len(eligible), len(daily) - 1 - len(eligible))
