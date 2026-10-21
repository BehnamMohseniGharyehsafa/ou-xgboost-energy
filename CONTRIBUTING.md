# Contributing

Issues and pull requests are welcome. For scientific changes, describe the hypothesis, information set, mathematical assumptions and baseline comparison. For bugs, include a minimal synthetic input and relevant settings from `run.json`. Avoid uploading private or restricted market data.

## Local checks

```bash
python -m venv .venv
# Activate using the command for your platform in README.md.
python -m pip install -e ".[dev,xgb,plots]"
python -m pytest --cov=energy_forecast --cov-report=term-missing
ruff check .
ruff format --check .
python -m build
```

Keep changes focused and add tests for changed scientific behavior. Any feature based on observed prices must be available before the target origin. Preserve the future-perturbation tests; training transformations must fit only on historical training data. Calendar features known in advance may use the target date.

Do not optimize the blend or other settings on the reported test period. Compare against both persistence and weekly seasonal baselines. Report backend, device and date coverage when sharing results. If intentionally changing the data contract or forecast timing, update `docs/` and the CLI help together.
