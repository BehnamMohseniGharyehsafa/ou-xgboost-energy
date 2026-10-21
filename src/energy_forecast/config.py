"""Validated, serializable experiment settings."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ForecastConfig:
    """All durations use UTC calendar days or hours, as named."""

    cadence_minutes: int = 60
    min_train_days: int = 120
    refit_every: int = 7
    ou_window_days: int = 60
    blend_weight: float = 0.5
    backend: str = "auto"
    device: str = "auto"
    n_estimators: int = 250
    seed: int = 42
    n_jobs: int = 2

    def __post_init__(self):
        for name in (
            "cadence_minutes",
            "min_train_days",
            "refit_every",
            "ou_window_days",
            "n_estimators",
            "n_jobs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cadence_minutes > 720 or 1440 % self.cadence_minutes:
            raise ValueError("cadence_minutes must divide 1440 and be at most 720")
        if not isfinite(self.blend_weight) or not 0 <= self.blend_weight <= 1:
            raise ValueError("blend_weight must be in [0, 1]")
        if self.backend not in {"auto", "xgboost", "hist"}:
            raise ValueError("backend must be auto, xgboost, or hist")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.backend == "hist" and self.device == "cuda":
            raise ValueError("The hist backend runs on CPU; CUDA requires XGBoost")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
