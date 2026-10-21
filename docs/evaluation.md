# Evaluation and reproducibility

## Walk-forward protocol

1. Aggregate complete UTC days and build strictly lagged features.
2. Reserve the first `min_train_days` eligible rows for initial fitting.
3. At the next eligible date, fit the tree on all earlier eligible dates.
4. Refit after at least `refit_every` calendar days (default seven); between refits reuse the tree.
5. At every origin update lagged features and refit the OU drift on its past-only window.
6. Predict and record the target value only for scoring. Continue forward without shuffling.

The training window expands. Missing days are skipped without compressing calendar lags. `trained_through` records the latest label used in the most recent tree fit; `last_observed_at` records the OU endpoint. Both precede the target date.

The next-day forecast command fits a fresh tree on all eligible historical rows and predicts exactly one future date. It does not create a future actual or calculate metrics.

## Models to compare

| Column | Interpretation |
| --- | --- |
| `yhat_ml` | Actual XGBoost or scikit-learn tree forecast |
| `yhat_ou` | OU expectation with tree forecast as the future level |
| `yhat_ou_historical` | OU expectation with fitted historical level |
| `yhat_blend` | Weighted tree + conditioned OU forecast, with logged fallback |
| `yhat_persistence` | Previous complete day's mean |
| `yhat_seasonal_naive` | Mean exactly seven calendar days earlier |

Using only the previous mean as a baseline can hide a strong weekly cycle; report both naive benchmarks. The hybrid is not assumed to win.

## Metrics

For $e_d=\widehat y_d-y_d$, report $\mathrm{MAE}=n^{-1}\sum|e_d|$, $\mathrm{RMSE}=\sqrt{n^{-1}\sum e_d^2}$ and $\mathrm{bias}=n^{-1}\sum e_d$. Scores are in original input units. Every model reports `n`, its finite actual/prediction pair count. No missing value is passed to the scoring computation. Zero and negative actuals are valid.

When OU validity differs from other forecasts, its score covers a different subset. For a controlled comparison, select the intersection first:

```python
columns = ["yhat_ml", "yhat_ou", "yhat_blend", "yhat_seasonal_naive"]
common = predictions.dropna(subset=["y_true", *columns])
from energy_forecast.evaluation import score_predictions

print(score_predictions(common[["y_true", *columns]]))
```

There is no MAPE, automatic blend selection, significance test, confidence interval or financial backtest. Overlapping dependence in errors means naive independent-sample uncertainty estimates would need additional justification.

## Reproduction

`run.json` records configuration, feature names, Python and package versions, input SHA-256, actual backends/devices, evaluation dates and fallback counts. It intentionally omits absolute input paths. The synthetic generator has a local seeded NumPy random generator; CUDA probing does not mutate global randomness.

The example pins a CPU XGBoost run for portability. Its exact Python 3.13 package versions are in `examples/demo/requirements-tested.txt`. Use them when reproducing that run, or install the broader ranges in `pyproject.toml` on another Python version. Bitwise equivalence across library versions, hardware, operating systems and GPU reductions is not promised. The manifest creation timestamp changes on each run.

```bash
python -m pip install -r examples/demo/requirements-tested.txt
python -m pip install -e . --no-deps
energy-forecast demo --days 365 --seed 42 --out data/synthetic_prices.csv
energy-forecast backtest --csv data/synthetic_prices.csv --backend xgboost --device cpu --plots --out outputs/reproduction
```

## Extending to a publication

Use market data whose timestamps and publication history match the information set. Define a separate chronological validation period for hyperparameters and blend weight; keep the final test period untouched. Report market, unit, sampling cadence, exclusions, data vintage, forecast cutoff, rolling schedule and all baselines. Assess seasonality, price spikes, regime shifts and OU residual behavior. If studying an auction, rebuild features at its true cutoff before calling the exercise day-ahead auction forecasting.
