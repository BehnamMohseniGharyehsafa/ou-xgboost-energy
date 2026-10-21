"""Command-line entry points for synthetic data, backtests and real forecasts."""

import argparse
import logging
from pathlib import Path

from .config import ForecastConfig
from .data import load_prices
from .pipeline import backtest, forecast_next_day
from .reporting import save_result
from .synthetic import make_synthetic_prices


def build_parser():
    parser = argparse.ArgumentParser(
        description="OU + gradient boosting electricity price research"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo", help="Generate reproducible synthetic prices (no downloads)"
    )
    demo.add_argument("--out", type=Path, default=Path("data/synthetic_prices.csv"))
    demo.add_argument("--days", type=int, default=365)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--cadence-minutes", type=int, default=60)
    for command, help_text in (
        ("backtest", "Expanding-window evaluation on historical complete UTC days"),
        ("forecast", "Predict the day after the final complete UTC day"),
    ):
        item = subparsers.add_parser(command, help=help_text)
        item.add_argument("--csv", type=Path, required=True)
        item.add_argument("--timestamp-col")
        item.add_argument("--price-col")
        item.add_argument(
            "--out", type=Path, default=Path(f"outputs/{command}"), help="Output directory"
        )
        item.add_argument("--cadence-minutes", type=int, default=60)
        item.add_argument("--min-train-days", type=int, default=120)
        item.add_argument("--refit-every", type=int, default=7)
        item.add_argument("--ou-window-days", type=int, default=60)
        item.add_argument("--blend", type=float, default=0.5, help="Weight on ML in [0,1]")
        item.add_argument("--backend", choices=("auto", "xgboost", "hist"), default="auto")
        item.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        item.add_argument("--n-estimators", type=int, default=250)
        item.add_argument("--seed", type=int, default=42)
        item.add_argument("--n-jobs", type=int, default=2)
        if command == "backtest":
            item.add_argument("--plots", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "demo":
            config = ForecastConfig(cadence_minutes=args.cadence_minutes, seed=args.seed)
            prices = make_synthetic_prices(args.days, config.seed, config.cadence_minutes)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(args.out, float_format="%.8f", lineterminator="\n")
            print(f"Saved {len(prices)} artificial price observations: {args.out}")
            return 0
        config = ForecastConfig(
            cadence_minutes=args.cadence_minutes,
            min_train_days=args.min_train_days,
            refit_every=args.refit_every,
            ou_window_days=args.ou_window_days,
            blend_weight=args.blend,
            backend=args.backend,
            device=args.device,
            n_estimators=args.n_estimators,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        prices = load_prices(args.csv, args.timestamp_col, args.price_col, config.cadence_minutes)
        operation = backtest if args.command == "backtest" else forecast_next_day
        result = operation(prices, config)
        metrics = save_result(
            result,
            config,
            args.csv,
            args.out,
            plots=getattr(args, "plots", False),
            mode=args.command,
        )
        if metrics is not None:
            print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))
        else:
            print(result.predictions[["yhat_ml", "yhat_ou", "yhat_blend"]].to_string())
        print(f"Saved {len(result.predictions)} prediction(s) and provenance to {args.out}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
