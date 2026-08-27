from types import SimpleNamespace

import pandas as pd

import forexgrand_core.backtesting as backtesting


class FakeStrategy:
    def __init__(self):
        self.batch_times = []

    def generate_signals(self, batch):
        self.batch_times.extend(batch["time"][:, -1].tolist())
        return [0] * len(batch["time"])


def test_market_table_builder_converts_spread_points_to_price():
    data = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=1, freq="min"),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "spread": [20],
        }
    )

    market = backtesting.MarketTableBuilder.build(data, point_size=0.0001)

    assert market.spread.iloc[0] == 0.002
    assert market.high_ask.iloc[0] == 1.102
    assert market.low_ask.iloc[0] == 0.902


def test_run_backtest_limits_market_to_requested_indexes(monkeypatch):
    data = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="min"),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
        }
    )
    strategy = FakeStrategy()

    class FakeDataManager:
        def __init__(self, **kwargs):
            pass

        def load_data(self, **kwargs):
            return data, SimpleNamespace(point_size=1.0)

    monkeypatch.setattr(backtesting, "DataManager", FakeDataManager)
    monkeypatch.setattr(backtesting, "_load_strategy", lambda *args: strategy)

    result = backtesting.run_backtest(
        "strategy.py",
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sequence_length=2,
        start_index=1,
        end_index=4,
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
    )

    assert strategy.batch_times == [
        data.time.iloc[2].value // 10**9,
        data.time.iloc[3].value // 10**9,
    ]
    assert len(result.profit_equity) == 3

    strategy.batch_times.clear()
    result = backtesting.run_backtest(
        "strategy.py",
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sequence_length=2,
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
    )

    assert len(strategy.batch_times) == 4
    assert len(result.profit_equity) == 5


def test_run_backtest_can_return_profit_and_drawdown_in_points(monkeypatch):
    data = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="min"),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
        }
    )
    strategy = FakeStrategy()

    class FakeDataManager:
        def __init__(self, **kwargs):
            pass

        def load_data(self, **kwargs):
            return data, SimpleNamespace(point_size=0.1)

    monkeypatch.setattr(backtesting, "DataManager", FakeDataManager)
    monkeypatch.setattr(backtesting, "_load_strategy", lambda *args: strategy)

    result = backtesting.run_backtest(
        "strategy.py",
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sequence_length=2,
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
        return_in_points=True,
    )

    assert result.positions["profit"].tolist() == [30.0, 20.0, 10.0, 0.0]
    assert result.positions["max_profit"].tolist() == [30.0, 20.0, 10.0, 0.0]
    assert result.positions["min_dd"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result.profit_equity.tolist() == [0.0, 0.0, 10.0, 30.0, 60.0]
    assert result.dd_equity.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert result.positions["sl"].iloc[0] == -998.0