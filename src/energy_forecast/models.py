"""Deterministic boosting with verified CUDA selection and explicit fallbacks."""

import importlib
import json
import logging
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .config import ForecastConfig

LOGGER = logging.getLogger(__name__)


def _uses_cuda(model) -> bool:
    config = json.loads(model.get_booster().save_config())
    return config["learner"]["generic_param"]["device"].startswith("cuda")


def _cuda_available(xgb, seed: int, n_jobs: int) -> bool:
    """A successful fit is insufficient: XGBoost may silently fall back to CPU."""
    rng = np.random.default_rng(seed)
    try:
        probe = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=n_jobs,
        )
        with warnings.catch_warnings(record=True):
            probe.fit(rng.normal(size=(32, 4)), rng.normal(size=32))
        return _uses_cuda(probe)
    except (xgb.core.XGBoostError, ValueError, RuntimeError):
        return False


class TreeRegressor:
    """Train XGBoost >=2, or scikit-learn histogram boosting when unavailable."""

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        self.xgb = None
        if config.backend != "hist":
            try:
                self.xgb = importlib.import_module("xgboost")
            except (ImportError, OSError) as error:
                if config.backend == "xgboost" or config.device == "cuda":
                    raise RuntimeError("Install a compatible XGBoost >=2 build") from error
                LOGGER.info("XGBoost unavailable; using scikit-learn histogram boosting")
            if self.xgb is not None and int(self.xgb.__version__.split(".")[0]) < 2:
                if config.backend == "xgboost" or config.device == "cuda":
                    raise RuntimeError("XGBoost >=2 is required")
                LOGGER.warning("XGBoost <2 is unsupported; using scikit-learn histogram boosting")
                self.xgb = None
        self.backend = "xgboost" if self.xgb is not None else "hist"
        self.device = "cpu"
        if self.xgb is not None and config.device != "cpu":
            available = _cuda_available(self.xgb, config.seed, config.n_jobs)
            if config.device == "cuda" and not available:
                raise RuntimeError(
                    "CUDA requested but an actual CUDA XGBoost fit could not be verified"
                )
            if available:
                self.device = "cuda"
            else:
                LOGGER.info("CUDA unavailable; XGBoost will use CPU")

    def _new_xgb(self):
        c = self.config
        return self.xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=self.device,
            n_estimators=c.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1,
            random_state=c.seed,
            n_jobs=c.n_jobs,
        )

    def fit(self, X, y):
        c = self.config
        if self.xgb is None:
            self.model = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.06,
                max_iter=c.n_estimators,
                min_samples_leaf=20,
                random_state=c.seed,
                early_stopping=False,
            )
            with threadpool_limits(limits=c.n_jobs):
                self.model.fit(X, y)
            return self
        self.model = self._new_xgb()
        try:
            self.model.fit(X, y)
            if self.device == "cuda" and not _uses_cuda(self.model):
                raise RuntimeError("XGBoost silently switched to CPU")
        except (self.xgb.core.XGBoostError, RuntimeError) as error:
            if self.device != "cuda" or c.device == "cuda":
                raise
            LOGGER.warning("CUDA training failed (%s); retrying with XGBoost CPU", error)
            self.device = "cpu"
            self.model = self._new_xgb()
            self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Fit the model before predicting")
        with threadpool_limits(limits=self.config.n_jobs):
            return self.model.predict(X)
