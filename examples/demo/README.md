# Reproduced synthetic experiment

This directory contains an executed demonstration, not market evidence. The input is the bundled `data/synthetic_prices.csv`, generated from 365 UTC days of artificial hourly prices beginning 2024-01-01, with seed 42. The price unit is arbitrary.

## Commands

After installing the project with XGBoost and plotting support:

```bash
energy-forecast demo --days 365 --seed 42 --out data/synthetic_prices.csv
energy-forecast backtest --csv data/synthetic_prices.csv --backend xgboost --device cpu --plots --out examples/demo
```

The settings are 28 feature warm-up days, 120 eligible training days, an expanding training window, seven-calendar-day tree refits, a 60-day OU window, 250 tree rounds, and blend weight 0.5. This yields **217 evaluation days**. The recorded backend is XGBoost on CPU.

## Observed results

Values are rounded from the committed `metrics.csv`; lower MAE/RMSE is better. Bias is forecast minus actual. All six methods have 217 scored dates, so the comparison uses identical coverage.

| Method | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: |
| XGBoost | 4.6234 | 6.0334 | -1.1337 |
| ML-conditioned OU | 6.0954 | 9.0885 | -3.9831 |
| Historical-level OU | 7.4230 | 10.3373 | -5.1812 |
| 50/50 hybrid | 5.0752 | 6.9189 | -2.5584 |
| Previous-day mean | 6.0709 | 7.5926 | -0.1026 |
| Same-weekday mean | 6.0108 | 7.4904 | -0.7377 |

**The hybrid improves on the naive baselines here, but worsens the XGBoost forecast.** The final within-day price is not necessarily a useful correction to a daily mean. The synthetic process also includes seasonality and spikes that do not match a constant-level OU model. The purpose is to expose the workflow and its tradeoffs, not to establish superiority. No parameters were selected by optimizing these reported errors.

![Synthetic diagnostics](diagnostics.png)

## Artifacts

- `predictions.csv`: dated forecasts, actuals, training cutoffs and OU diagnostics.
- `metrics.csv`: unrounded scores and per-model sample counts.
- `run.json`: complete configuration, input SHA-256 and dependency/device metadata.
- `requirements-tested.txt`: pinned numerical and plotting stack for this executed Python 3.13 run.

Cross-platform or cross-version floating point differences are possible. See [the reproduction protocol](../../docs/evaluation.md) and [scientific assumptions](../../docs/methodology.md).
