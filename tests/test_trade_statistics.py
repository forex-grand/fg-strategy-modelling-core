"""Synthetic data generator and smoke tests for trade_statistics."""

import numpy as np
import pandas as pd

from forexgrand_core.trade_statistics import NUMBA_AVAILABLE, TradeStatisticsEngine


def make_synthetic_data(
    n_trades=4000,
    start="2023-01-01",
    end="2025-01-01",
    equity_step_seconds=300,
    initial_balance=10000.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    start_s = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_s = int(pd.Timestamp(end, tz="UTC").timestamp())

    open_time = np.sort(rng.integers(start_s, end_s - 3600, size=n_trades)).astype(np.int64)
    duration = rng.exponential(scale=3600 * 6, size=n_trades).astype(np.int64) + 60
    close_time = np.minimum(open_time + duration, end_s)
    duration = close_time - open_time

    win_mask = rng.random(n_trades) < 0.52
    profit = np.where(
        win_mask,
        rng.gamma(shape=2.0, scale=15.0, size=n_trades),
        -rng.gamma(shape=2.0, scale=17.0, size=n_trades),
    )
    size = np.round(rng.uniform(0.01, 2.0, size=n_trades), 2)
    risk = np.abs(profit) * rng.uniform(0.8, 1.4, size=n_trades)
    mfe = np.abs(profit) * rng.uniform(1.0, 2.5, size=n_trades)
    mae = np.abs(np.where(win_mask, profit * rng.uniform(0.0, 0.6, size=n_trades), profit))

    positions = pd.DataFrame(
        {
            "profit": profit,
            "time_spent": duration.astype(np.float64),
            "open_time": open_time,
            "close_time": close_time,
            "size": size,
            "risk": risk,
            "mfe": mfe,
            "mae": mae,
        },
        index=close_time,
    )

    order = np.argsort(close_time)
    cum_profit = np.concatenate(([0.0], np.cumsum(profit[order])))
    close_sorted = close_time[order]
    equity_time = np.arange(start_s, end_s, equity_step_seconds, dtype=np.int64)
    counts = np.searchsorted(close_sorted, equity_time, side="right")
    balance = initial_balance + cum_profit[counts]

    noise = np.convolve(
        rng.normal(0, 0.003 * initial_balance, size=len(equity_time)),
        0.85 ** np.arange(20),
        mode="same",
    )
    return {
        "positions": positions,
        "equity": balance + noise,
        "balance": balance,
        "equity_time": equity_time,
    }


def test_synthetic_statistics_smoke():
    data = make_synthetic_data()
    stats = TradeStatisticsEngine(n_mc_sims=2000, mc_seed=7).compute(data)

    assert stats["meta"]["n_trades"] == 4000
    assert "indexes" not in stats

    def contains_array(value):
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, dict):
            return any(contains_array(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_array(item) for item in value)
        return False

    assert not contains_array(stats)

    drawdowns = stats["drawdown_stats"]
    assert drawdowns["balance_dd"]["max_dd_abs"] >= 0
    assert drawdowns["relative_dd"]["max_dd_abs"] >= 0
    assert all(not isinstance(value, np.ndarray) for value in drawdowns.values())
    assert all(
        not isinstance(value, np.ndarray)
        for section in (drawdowns["balance_dd"], drawdowns["relative_dd"])
        for value in section.values()
    )

    trade_profit = stats["trade_profit_stats"]
    assert abs(
        trade_profit["gross_profit"]
        + trade_profit["gross_loss"]
        - trade_profit["net_profit_abs"]
    ) < 1e-6

    streaks = stats["streak_stats"]
    assert streaks["num_win_streaks"] + streaks["num_loss_streaks"] > 0
    assert not {"start_idx", "end_idx", "streak_type"} & streaks.keys()
    assert stats["position_sizing_stats"]["column_used"] == "size"
    assert stats["trade_profit_stats"]["trade_risk_column_used"] == "risk"
    assert stats["trade_efficiency_stats"] is not None
    assert isinstance(NUMBA_AVAILABLE, bool)


def test_trade_time_field_selects_period_category():
    positions = pd.DataFrame(
        {
            "profit": [10.0, -5.0],
            "time_spent": [86400.0, 86400.0],
            "open_time": [
                int(pd.Timestamp("2024-01-31", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-02-01", tz="UTC").timestamp()),
            ],
            "close_time": [
                int(pd.Timestamp("2024-02-01", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-02-02", tz="UTC").timestamp()),
            ],
        }
    )
    data = {
        "positions": positions,
        "equity": [1000.0, 1010.0, 1005.0],
        "balance": [1000.0, 1010.0, 1005.0],
        "equity_time": [
            int(pd.Timestamp("2024-01-31", tz="UTC").timestamp()),
            int(pd.Timestamp("2024-02-01", tz="UTC").timestamp()),
            int(pd.Timestamp("2024-02-02", tz="UTC").timestamp()),
        ],
    }

    open_stats = TradeStatisticsEngine(n_mc_sims=1).compute(data)
    close_stats = TradeStatisticsEngine(n_mc_sims=1, trade_time_field="close_time").compute(data)

    assert open_stats["meta"]["trade_time_field"] == "open_time"
    assert close_stats["meta"]["trade_time_field"] == "close_time"
    open_monthly = open_stats["trade_count_stats"]["monthly"]
    close_monthly = close_stats["trade_count_stats"]["monthly"]

    assert set(open_monthly) == {"num_trades", "num_wins", "num_losses", "win_ratio", "net_profit"}
    assert open_monthly["num_trades"]["average_value"] == 1.0
    assert open_monthly["num_trades"]["min_value"] == 1.0
    assert open_monthly["num_trades"]["maximum_value"] == 1.0
    assert close_monthly["num_trades"]["average_value"] == 2.0
    assert close_monthly["num_trades"]["min_value"] == 2.0
    assert close_monthly["num_trades"]["maximum_value"] == 2.0


def test_net_profit_percentage_sums_trade_returns_not_absolute_profit():
    start = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
    positions = pd.DataFrame(
        {
            "profit": [200.0, -100.0],
            "time_spent": [60.0, 60.0],
            "open_time": [start, start + 60],
            "close_time": [start + 60, start + 120],
        }
    )
    data = {
        "positions": positions,
        "equity": [1000.0, 1200.0, 1100.0],
        "balance": [1000.0, 1200.0, 1100.0],
        "equity_time": [start, start + 60, start + 120],
    }

    trade_profit = TradeStatisticsEngine(n_mc_sims=1).compute(data)["trade_profit_stats"]

    assert trade_profit["net_profit_abs"] == 100.0
    assert trade_profit["net_profit_pct"] == (20.0 - (100.0 / 1200.0 * 100.0))


def test_position_sizing_stats_accept_backtest_lot_column():
    data = make_synthetic_data(n_trades=2)
    data["positions"] = data["positions"].drop(columns="size").assign(lot=[0.5, 1.25])

    stats = TradeStatisticsEngine(n_mc_sims=1).compute(data)

    assert stats["position_sizing_stats"]["column_used"] == "lot"
    assert stats["position_sizing_stats"]["all"]["p50"] == 0.875
    assert stats["position_sizing_stats"]["all"]["min_value"] == 0.5
    assert stats["position_sizing_stats"]["all"]["average_value"] == 0.875
    assert stats["position_sizing_stats"]["all"]["maximum_value"] == 1.25