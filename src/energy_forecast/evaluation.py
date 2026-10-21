"""Metrics with explicit missing-value masks and zero-safe reporting."""

import numpy as np
import pandas as pd


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE and bias in input units; each row reports its scored sample count.

    Missing standalone OU forecasts use a model-specific mask. Compare models
    on their common dates when masks differ. Percentage errors are omitted because
    electricity prices can be zero or negative.
    """
    records = []
    for column in predictions:
        if not column.startswith("yhat_"):
            continue
        valid = np.isfinite(predictions["y_true"]) & np.isfinite(predictions[column])
        error = (predictions.loc[valid, column] - predictions.loc[valid, "y_true"]).to_numpy()
        records.append(
            {
                "model": column.removeprefix("yhat_"),
                "n": len(error),
                "mae": float(np.mean(np.abs(error))) if len(error) else np.nan,
                "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                "bias": float(np.mean(error)) if len(error) else np.nan,
            }
        )
    return pd.DataFrame(records).set_index("model")
