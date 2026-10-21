import pytest

from energy_forecast.synthetic import make_synthetic_prices


@pytest.fixture
def prices():
    return make_synthetic_prices(days=105, seed=7)
