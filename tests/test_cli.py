import json

import pandas as pd

import forexgrand_core.backtesting as backtesting
from forexgrand_core.cli import main


def test_preprocess_data_applies_function_and_writes_gzip_pickle(tmp_path, capsys):
    input_path = tmp_path / "prices.csv"
    input_path.write_text("close\n1.0\n2.5\n", encoding="utf-8")
    preprocess_path = tmp_path / "preprocess.py"
    preprocess_path.write_text(
        "def preprocess_fn(frame):\n"
        "    frame = frame.copy()\n"
        "    frame['delta'] = frame['close'].diff().fillna(0)\n"
        "    return frame\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "processed.pkl.gz"

    assert main([
        "preprocess_data",
        str(input_path),
        str(preprocess_path),
        "--output",
        str(output_path),
    ]) == 0

    result = pd.read_pickle(output_path, compression="gzip")
    assert result["delta"].tolist() == [0.0, 1.5]
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_path)


def test_run_backtest_cli_maps_current_signature(monkeypatch, tmp_path, capsys):
    strategy_path = tmp_path / "strategy.py"
    strategy_path.write_text("", encoding="utf-8")
    captured = {}

    class FakeResult:
        positions_total = 3
        buy_count = 2
        sell_count = 1
        unsupported_signal_count = 4
        profit_equity = pd.Series([10.0, 20.0])
        dd_equity = pd.Series([-1.0, -2.0])

    def fake_run_backtest(strategy, **kwargs):
        captured["strategy"] = strategy
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(backtesting, "run_backtest", fake_run_backtest)

    assert main([
        "run_backtest",
        str(strategy_path),
        "EURUSD",
        "--bucket",
        "forexgrand-test",
        "--source",
        "dukascopy",
        "--instrument-group",
        "forex_majors",
        "--sequence-length",
        "2800",
        "--stride",
        "60",
        "--batch-size",
        "32",
        "--sl-calculation",
        '{"mode":"fixed","sl_points":100,"tp_points":150}',
        "--end-index",
        "10000",
        "--return-in-points",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert captured["strategy"] == strategy_path
    assert captured["instrument_group"] == "forex_majors"
    assert captured["sequence_length"] == 2800
    assert captured["stride"] == 60
    assert captured["batch_size"] == 32
    assert captured["sl_calculation"] == {"mode": "fixed", "sl_points": 100, "tp_points": 150}
    assert captured["return_in_points"] is True
    assert payload["positions_total"] == 3
    assert payload["profit_equity_points"] == [10.0, 20.0]