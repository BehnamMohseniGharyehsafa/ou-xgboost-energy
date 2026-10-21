# Data contract

## CSV schema

Supply one series from one market and one consistent price unit. Required fields are timestamp and price. Safe automatic aliases are `timestamp`, `datetime`, `date`, `time` for the timestamp and `price`, `smp`, `market_price` for the price (case insensitive). If several candidates exist, choose explicitly using `--timestamp-col` and `--price-col`. The loader never chooses the most volatile numeric column.

Timestamps must be parseable date/time strings; numeric epoch values and identifiers are rejected. Offset-bearing ISO 8601 strings are recommended. Naive strings are interpreted as UTC. Rows are sorted; duplicate timestamps, missing/non-finite prices and off-grid timestamps cause an error. Zero and negative prices are valid. Decimal values must use `.`; preprocess other locale formats explicitly.

## Regular grid and daily completeness

Set `--cadence-minutes` to the known sampling interval: 60 for hourly, 30 for half-hourly, or 15 for quarter-hourly prices. It must divide 1440 and be at most 720 minutes. The grid starts at UTC midnight. A day is complete only if it contains all expected unique grid timestamps. For hourly data that is exactly 00:00 through 23:00.

The arithmetic mean equals the equal-duration weighted mean on this grid. Irregular-duration or volume-weighted products require a different aggregation. Missing rows are allowed in the series but invalidate that day's target and state features. Dependent lags and rolling features become missing, which may exclude several later days. No forward filling or interpolation is performed. OU gap-spanning pairs are also excluded.

## Interval labels

The program treats timestamps as observation times or interval starts. If your feed uses interval-end labels, convert them to interval starts before loading; the final midnight endpoint belongs to the previous delivery interval. For the OU component, using interval-start-labelled settlement prices as point observations is a modelling approximation. Actual publication times must still be no later than the forecast origin.

## Timezones and DST

UTC days are always 24 hours. Local market delivery days may have 23 or 25 hours at daylight-saving changes and generally do not align with UTC midnight. Convert timezone-aware data to UTC explicitly and check your desired target definition. For a local-market daily product, adapt aggregation and the horizon instead of relabelling the UTC result.

Example preprocessing when the source already has explicit timezone offsets:

```python
import pandas as pd

raw = pd.read_csv("source.csv")
raw["timestamp"] = pd.to_datetime(raw["delivery_time"], utc=True, format="mixed")
# For interval-end labels only, after confirming the vendor's convention:
# raw["timestamp"] -= pd.Timedelta(minutes=30)
raw[["timestamp", "price"]].to_csv("normalized.csv", index=False)
```

Do not resolve ambiguous naive local times by assuming UTC. Localize them using the market timezone and handle ambiguous/nonexistent DST observations explicitly before conversion.

## Required history

On an uninterrupted grid, 28 days are needed to build the longest lag and rolling feature. The default backtest then needs 120 eligible training days plus at least one eligible evaluation day: 149 raw days minimum. The forecast command needs 148 complete raw days by default and requires the final supplied day to be complete. Extra missing days increase these requirements.

Store private or licensed inputs in `data/raw/`, which Git ignores. The public repository contains generated data only. No source market dataset accompanied the original code.
