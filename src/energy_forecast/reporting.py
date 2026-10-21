"""Portable experiment artifacts, provenance and optional scientific figures."""

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import score_predictions


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_result(result, config, input_path, output_dir, *, plots=False, mode="backtest"):
    """Write predictions, metrics where observed, and a JSON run manifest."""
    output = Path(output_dir)
    # Check optional dependency before creating partial output artifacts.
    if plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install plotting support: pip install '.[plots]'") from error
    output.mkdir(parents=True, exist_ok=True)
    frame = result.predictions
    frame.to_csv(output / "predictions.csv", lineterminator="\n")
    metrics = score_predictions(frame) if "y_true" in frame else None
    if metrics is not None:
        metrics.to_csv(output / "metrics.csv", lineterminator="\n")
    versions = {"python": platform.python_version()}
    for package in (
        "ou-xgboost-energy",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    manifest = {
        "mode": mode,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "config": asdict(config),
        "versions": versions,
        "feature_names": result.feature_names,
        "eligible_days": result.eligible_days,
        "warmup_or_incomplete_days": result.skipped_days,
        "prediction_count": len(frame),
        "prediction_start": frame.index.min().isoformat(),
        "prediction_end": frame.index.max().isoformat(),
        "actual_backends": sorted(frame["backend"].unique().tolist()),
        "actual_devices": sorted(frame["device"].unique().tolist()),
        "ou_status_counts": frame["ou_status"].value_counts().to_dict(),
        "blend_fallback_count": int(frame["blend_fallback"].sum()),
        "metrics": metrics.reset_index().to_dict(orient="records") if metrics is not None else None,
    }
    (output / "run.json").write_text(
        json.dumps(_clean(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if plots and metrics is not None:
        plot_results(frame, metrics, output / "diagnostics.png")
    return metrics


def plot_results(frame, metrics, path):
    """Four-panel comparison; figures never claim superiority of the blend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    actual, blend = frame["y_true"], frame["yhat_blend"]
    axes[0, 0].plot(frame.index, actual, color="#112b3c", lw=1.1, label="Observed")
    axes[0, 0].plot(frame.index, blend, color="#008d91", lw=1, alpha=0.9, label="OU + boosting")
    axes[0, 0].set(title="Next-day daily mean", ylabel="Price (input units)")
    axes[0, 0].legend(frameon=False)
    colors = ["#008d91" if name == "blend" else "#8699ac" for name in metrics.index]
    axes[0, 1].barh(metrics.index, metrics["mae"], color=colors)
    axes[0, 1].set(title="All reported models", xlabel="MAE (lower is better)")
    axes[1, 0].scatter(actual, blend, s=12, alpha=0.6, color="#008d91", edgecolors="none")
    bounds = [min(actual.min(), blend.min()), max(actual.max(), blend.max())]
    axes[1, 0].plot(bounds, bounds, "--", color="#a06932", lw=1)
    axes[1, 0].set(title="Calibration of point forecasts", xlabel="Observed", ylabel="Blend")
    error = blend - actual
    axes[1, 1].plot(frame.index, error, lw=0.7, alpha=0.5, color="#8699ac")
    axes[1, 1].plot(frame.index, error.rolling(14).mean(), color="#008d91", label="14-day mean")
    axes[1, 1].axhline(0, color="#a06932", lw=1, ls="--")
    axes[1, 1].set(title="Forecast error", ylabel="Predicted minus observed")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    for ax in (axes[0, 0], axes[1, 1]):
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("OU + gradient boosting | chronological backtest", fontsize=17, weight="bold")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
