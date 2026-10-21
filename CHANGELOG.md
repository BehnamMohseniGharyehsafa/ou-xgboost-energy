# Changelog

## 0.1.0 — 2026-10-21

Initial research package based on the author's supplied OU + XGBoost prototype.

- Correct target-day feature leakage and remove whole-sample clipping.
- Support negative prices through additive within-day changes.
- Validate CSV schema, UTC cadence, duplicate timestamps and complete days.
- Estimate past-window OU drift using elapsed times, trimmed increments and gap exclusion.
- Include last-observation offset in the analytic daily expectation.
- Verify actual CUDA use; provide logged automatic CPU and scikit-learn fallbacks.
- Add chronological backtesting, unseen next-day forecasts and two naive baselines.
- Export finite-pair metrics, provenance manifests and optional scientific diagnostics.
- Add reproducible synthetic results, mathematical documentation, tests and CI.

See [the migration notes](docs/migration.md) for behavioral changes.
