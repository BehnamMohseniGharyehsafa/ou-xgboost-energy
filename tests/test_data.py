import numpy as np
import pandas as pd
import pytest

from energy_forecast.data import load_prices, validate_prices


def test_load_explicit_schema_and_negative_prices(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "delivery": ["2024-01-01T01:00:00+01:00", "2024-01-01T02:00:00+01:00"],
            "eur_mwh": [-12.5, 0],
            "load": [10000, 20000],
        }
    ).to_csv(path, index=False)
    result = load_prices(path, "delivery", "eur_mwh")
    assert result.tolist() == [-12.5, 0]
    assert result.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_numeric_identifier_is_not_a_timestamp(tmp_path):
    path = tmp_path / "numeric.csv"
    pd.DataFrame({"timestamp": [1, 2], "price": [40, 60]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not numeric"):
        load_prices(path)


def test_ambiguous_price_column_requires_override(tmp_path):
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"timestamp": ["2024-01-01"], "price": [1], "smp": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="explicitly"):
        load_prices(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_prices_rejected(bad):
    prices = pd.Series([1, bad], index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="finite"):
        validate_prices(prices)


def test_duplicate_and_off_grid_rejected(prices):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prices(pd.concat([prices, prices.iloc[:1]]))
    shifted = prices.copy()
    shifted.index += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="align"):
        validate_prices(shifted)


def test_naive_index_rejected(prices):
    prices.index = prices.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_prices(prices)
