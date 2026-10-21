"""Electricity price forecasting with causal features and OU dynamics."""

from .config import ForecastConfig
from .pipeline import backtest, forecast_next_day

__version__ = "0.1.0"
__all__ = ["ForecastConfig", "backtest", "forecast_next_day"]
