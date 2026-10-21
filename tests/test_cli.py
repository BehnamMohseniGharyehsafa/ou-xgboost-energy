import hashlib
import json

import pandas as pd
import pytest

from energy_forecast.cli import main


def test_demo_backtest_forecast_and_manifest(tmp_path):
    path, output = tmp_path / "demo.csv", tmp_path / "run"
    assert main(["demo", "--out", str(path), "--days", "80"]) == 0
    common = [
        "--csv",
        str(path),
        "--out",
        str(output),
        "--min-train-days",
        "40",
        "--backend",
        "hist",
        "--n-estimators",
        "3",
        "--n-jobs",
        "1",
    ]
    assert main(["backtest", *common]) == 0
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["prediction_count"] == 12
    assert manifest["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert b"\r\n" not in path.read_bytes()
    assert manifest["actual_devices"] == ["cpu"]
    assert (output / "metrics.csv").is_file()
    forecast_output = tmp_path / "forecast"
    assert main(["forecast", *common, "--out", str(forecast_output)]) == 0
    result = pd.read_csv(forecast_output / "predictions.csv")
    assert len(result) == 1 and "y_true" not in result


def test_bad_cli_input_exits_with_message(capsys, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["backtest", "--csv", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
