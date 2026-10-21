from types import SimpleNamespace

import numpy as np
import pytest

from energy_forecast.config import ForecastConfig
from energy_forecast.models import TreeRegressor, _cuda_available


def test_silent_cpu_fallback_is_not_misreported_as_gpu():
    class SilentCPU:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def get_booster(self):
            return SimpleNamespace(
                save_config=lambda: '{"learner":{"generic_param":{"device":"cpu"}}}'
            )

    fake = SimpleNamespace(XGBRegressor=SilentCPU, core=SimpleNamespace(XGBoostError=RuntimeError))
    assert not _cuda_available(fake, seed=42, n_jobs=1)


def test_auto_backend_without_xgboost(monkeypatch):
    def missing(name):
        raise ImportError("not installed")

    monkeypatch.setattr("energy_forecast.models.import_module", missing)
    assert TreeRegressor(ForecastConfig()).backend == "hist"
    with pytest.raises(RuntimeError, match="Install"):
        TreeRegressor(ForecastConfig(backend="xgboost"))


def test_xgboost_cpu_smoke():
    pytest.importorskip("xgboost", minversion="2.0")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4))
    model = TreeRegressor(ForecastConfig(backend="xgboost", device="cpu", n_estimators=3))
    model.fit(X, X[:, 0] * 2)
    assert model.backend == "xgboost"
    assert model.device == "cpu"
    assert np.isfinite(model.predict(X[:2])).all()


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_cuda_training_failure_respects_device_request(monkeypatch, device):
    class DeviceFailure:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]

        def fit(self, X, y):
            if self.device == "cuda":
                raise RuntimeError("GPU memory unavailable")
            return self

        def predict(self, X):
            return np.full(len(X), 1.0)

    fake = SimpleNamespace(
        __version__="3.0.0",
        XGBRegressor=DeviceFailure,
        core=SimpleNamespace(XGBoostError=RuntimeError),
    )
    monkeypatch.setattr("energy_forecast.models.import_module", lambda name: fake)
    monkeypatch.setattr("energy_forecast.models._cuda_available", lambda *args: True)
    model = TreeRegressor(ForecastConfig(device=device))
    X, y = np.zeros((30, 2)), np.ones(30)
    if device == "cuda":
        with pytest.raises(RuntimeError, match="GPU memory"):
            model.fit(X, y)
    else:
        model.fit(X, y)
        assert model.device == "cpu"
        assert model.predict(X[:1])[0] == 1


def test_explicit_cuda_cannot_silently_use_cpu(monkeypatch):
    fake = SimpleNamespace(__version__="3.0.0")
    monkeypatch.setattr("energy_forecast.models.import_module", lambda name: fake)
    monkeypatch.setattr("energy_forecast.models._cuda_available", lambda *args: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        TreeRegressor(ForecastConfig(device="cuda"))
