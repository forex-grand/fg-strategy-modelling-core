from types import SimpleNamespace

import pandas as pd

import fg_core.backtesting as backtesting


class FakeStrategy(backtesting.SignalsBase):
    def __init__(self, sequence_length=2):
        super().__init__(sequence_length)
        self.batch_times = []

    def signals(self, batch):
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


def test_fixed_sl_tp_scales_by_symbol_points():
    calculator = backtesting.SLTPCalculator(
        {"mode": "fixed", "sl_points": 100, "tp_points": 50},
        symbol_points=0.0001,
    )
    signal = {"row_index": 0, "open_price": 1.0, "direction": 0}
    market = pd.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]})

    stop_loss, take_profit = calculator.compute(signal, market)

    assert stop_loss == 0.99
    assert take_profit == 1.005


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
    result = backtesting.run_backtest(
        strategy,
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
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
        strategy,
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
    )

    assert len(strategy.batch_times) == 4
    assert len(result.profit_equity) == 5


def test_run_backtest_returns_current_trade_result_contract(monkeypatch):
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
    result = backtesting.run_backtest(
        strategy,
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
        return_in_points=True,
    )

    assert result.positions["profit"].tolist() == [30.0]
    assert result.positions["time_spent"].tolist() == [180]
    assert result.positions["close_reason"].tolist() == ["eod"]
    assert result.positions.index.tolist() == [
        1704067260,
    ]
    assert result.balance_equity.tolist() == [10000.0, 10000.0, 10000.0, 10000.0, 10030.0]
    assert result.final_balance == 10030.0
    assert result.profit_equity.equals(result.balance_equity)
    assert result.dd_equity.equals(result.balance_equity - result.equity_curve)
    assert result.positions["sl"].iloc[0] == -8.0
    assert result.positions["tp"].iloc[0] == 12.0


def test_run_backtest_limits_open_trades_and_statistics_to_traded_signals(monkeypatch):
    data = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="min"),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
        }
    )

    class FakeDataManager:
        def __init__(self, **kwargs):
            pass

        def load_data(self, **kwargs):
            return data, SimpleNamespace(point_size=0.1)

    monkeypatch.setattr(backtesting, "DataManager", FakeDataManager)
    result = backtesting.run_backtest(
        FakeStrategy(),
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
        result_type=backtesting.RESULT_TYPE(backtesting.trade_result_and_statistics),
    )

    assert isinstance(result, dict)
    assert result["positions_total"] == 1
    assert len(result["positions"]) == 1
    assert result["statistics"]["meta"]["n_trades"] == 1

    result = backtesting.run_backtest(
        FakeStrategy(),
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
        max_open_trades=2,
    )
    assert result["positions_total"] == 2


def test_range_sl_tp_highlow_default_and_close():
    market = pd.DataFrame({
        "high": [1.15, 1.20, 1.10],
        "low": [1.05, 1.08, 1.02],
        "close": [1.12, 1.14, 1.05],
    })

    # Default is highlow
    config_default = backtesting.RANGE(range=2)
    assert config_default["range_type"] == "highlow"
    calc_hl = backtesting.SLTPCalculator(config_default, symbol_points=0.0001)
    # At row_index 2, context is rows 0 and 1:
    # high max = 1.20, low min = 1.05 -> range = 0.15
    sl_hl, tp_hl = calc_hl.compute({"row_index": 2, "open_price": 1.10, "direction": 0}, market)
    assert round(sl_hl, 6) == round(1.10 - 0.15, 6)
    assert round(tp_hl, 6) == round(1.10 + 0.15, 6)

    # Use close
    config_close = backtesting.RANGE(range=2, range_type="close")
    calc_close = backtesting.SLTPCalculator(config_close, symbol_points=0.0001)
    # At row_index 2, context is rows 0 and 1:
    # close max = 1.14, close min = 1.12 -> range = 0.02
    sl_c, tp_c = calc_close.compute({"row_index": 2, "open_price": 1.10, "direction": 0}, market)
    assert round(sl_c, 6) == round(1.10 - 0.02, 6)
    assert round(tp_c, 6) == round(1.10 + 0.02, 6)


class DummyAuxManager:
    def __init__(self, model_dict):
        self.model = model_dict
        self.prepared_batches = []

    def prepare_data(self, batch, sequence_length):
        self.prepared_batches.append((batch, sequence_length))
        return {k: batch[k] for k in batch}


def test_backtest_with_model_id_for_signals(monkeypatch):
    import numpy as np

    data = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=6, freq="min"),
        "open": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "high": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        "low": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        "close": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    })

    class FakeDataManager:
        def __init__(self, **kwargs):
            pass

        def load_data(self, **kwargs):
            return data, SimpleNamespace(point_size=0.1)

    monkeypatch.setattr(backtesting, "DataManager", FakeDataManager)

    for output_key in ("output", "signal", "direction"):
        def fake_model_fn(**kwargs):
            # return shape (batch_size, 3) probabilities where class 0 (buy) is highest
            batch_len = len(list(kwargs.values())[0])
            probs = np.zeros((batch_len, 3))
            probs[:, 0] = 1.0  # buy
            return {output_key: probs}

        dummy_manager = DummyAuxManager({
            "kind": "single",
            "metadata": {"sequence_length": 2, "model_type": "nn"},
            "model": fake_model_fn,
        })

        monkeypatch.setattr(
            "fg_core.aux_model_manager.AuxilaryModelManager",
            lambda model_id, output_path=None: dummy_manager,
        )

        result = backtesting.run_backtest(
            model_id="test_signal_model",
            bucket_name="bucket",
            source="source",
            symbol_pair="EURUSD",
            sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 100},
        )

        assert result.buy_count > 0
        assert len(dummy_manager.prepared_batches) > 0
        # sequence_length was inferred as 2
        assert dummy_manager.prepared_batches[0][1] == 2


def test_backtest_with_model_id_for_sltp(monkeypatch):
    import numpy as np

    data = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=5, freq="min"),
        "open": [1.0, 1.1, 1.2, 1.3, 1.4],
        "high": [1.1, 1.2, 1.3, 1.4, 1.5],
        "low": [0.9, 1.0, 1.1, 1.2, 1.3],
        "close": [1.0, 1.1, 1.2, 1.3, 1.4],
    })

    class FakeDataManager:
        def __init__(self, **kwargs):
            pass

        def load_data(self, **kwargs):
            return data, SimpleNamespace(point_size=0.0001)

    monkeypatch.setattr(backtesting, "DataManager", FakeDataManager)

    def fake_sltp_fn(**kwargs):
        # returns raw difference: sl=0.05, tp=0.10
        batch_len = len(list(kwargs.values())[0])
        return {"output": np.tile([0.05, 0.10], (batch_len, 1))}

    dummy_sltp_manager = DummyAuxManager({
        "kind": "single",
        "metadata": {"sequence_length": 2, "model_type": "nn"},
        "model": fake_sltp_fn,
    })

    monkeypatch.setattr(
        "fg_core.aux_model_manager.AuxilaryModelManager",
        lambda model_id, output_path=None: dummy_sltp_manager,
    )

    result = backtesting.run_backtest(
        FakeStrategy(sequence_length=2),
        bucket_name="bucket",
        source="source",
        symbol_pair="EURUSD",
        sltp_model_id="test_sltp_model",
    )

    # Check positions sl and tp are actual raw differences, not multiplied by 0.0001
    assert len(result.positions) > 0
    first_pos = result.positions.iloc[0]
    open_p = first_pos["open_price"]
    # Direction is 0 (buy): sl = open_price - 0.05, tp = open_price + 0.10
    assert round(first_pos["sl"], 4) == round(open_p - 0.05, 4)
    assert round(first_pos["tp"], 4) == round(open_p + 0.10, 4)
    # Check that sequence_length was inferred as 2
    assert dummy_sltp_manager.prepared_batches[0][1] == 2