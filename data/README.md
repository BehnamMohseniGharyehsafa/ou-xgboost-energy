# Example data

`synthetic_prices.csv` is artificial hourly data produced by:

```bash
energy-forecast demo --days 365 --seed 42 --out data/synthetic_prices.csv
```

It includes a smooth annual level, daily/weekly structure, mean-reverting noise and occasional signed spikes. Units are arbitrary. It is provided only to exercise the software; no market measurements or proprietary data are included.

Store your own inputs in the Git-ignored `data/raw/` directory. Read [the data contract](../docs/data.md) before converting market data.
