"""Signal-level bar replay backtesting."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from forexgrand_core.data_manager import DataManager
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase


class StrategyLoadError(ValueError):
    """Raised when a strategy module cannot be loaded or validated."""


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    profit_equity: pd.Series
    dd_equity: pd.Series
    positions_total: int
    buy_count: int
    sell_count: int
    unsupported_signal_count: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SLTPCalculator:
    _KEYS = {
        "fixed": {"mode", "sl_points", "tp_points"},
        "range": {"mode", "range", "sl_ratio", "tp_ratio"},
        "atr": {"mode", "sl_multiplier", "tp_multiplier", "atr_period"},
    }

    def __init__(self, configuration: Optional[Mapping[str, Any]], symbol_points: float) -> None:
        config = dict(configuration or {"mode": "fixed", "sl_points": 100, "tp_points": 100})
        mode = config.get("mode", "fixed")
        defaults = {
            "fixed": {"sl_points": 100, "tp_points": 100},
            "range": {"range": 60, "sl_ratio": 1.0, "tp_ratio": 1.0},
            "atr": {"sl_multiplier": 3.0, "tp_multiplier": 3.0, "atr_period": 14},
        }
        if mode not in self._KEYS:
            raise ValueError(f"Unsupported SL/TP mode: {mode!r}.")
        unknown = set(config) - self._KEYS[mode]
        if unknown:
            raise ValueError(f"Unknown keys for {mode} SL/TP mode: {sorted(unknown)}")
        config = {**defaults[mode], **config, "mode": mode}
        if symbol_points <= 0:
            raise ValueError("symbol_points must be greater than zero.")
        self.config = config
        self.symbol_points = float(symbol_points)

    def compute(self, signal: Mapping[str, Any], market: pd.DataFrame) -> tuple[float, float]:
        mode = self.config["mode"]
        row_index = int(signal["row_index"])
        open_price = float(signal["open_price"])
        direction = int(signal["direction"])
        if mode == "fixed":
            sl_distance = float(self.config["sl_points"]) / self.symbol_points
            tp_distance = float(self.config["tp_points"]) / self.symbol_points
        elif mode == "range":
            length = int(self.config["range"])
            if length <= 0 or row_index < length:
                raise ValueError("range must be positive and have enough preceding bars.")
            context = market.iloc[row_index - length:row_index]
            bar_range = context["high"].max() - context["low"].min()
            sl_distance = bar_range * float(self.config["sl_ratio"])
            tp_distance = bar_range * float(self.config["tp_ratio"])
        else:
            period = int(self.config["atr_period"])
            if period <= 0 or row_index < period:
                raise ValueError("atr_period must be positive and have enough preceding bars.")
            high, low = market.high.to_numpy(float), market.low.to_numpy(float)
            close = market.close.to_numpy(float)
            previous_close = np.concatenate(([close[0]], close[:-1]))
            true_range = np.maximum.reduce((high - low, abs(high - previous_close), abs(low - previous_close)))
            atr = pd.Series(true_range).rolling(period).mean().iloc[row_index - 1]
            sl_distance = float(atr) * float(self.config["sl_multiplier"])
            tp_distance = float(atr) * float(self.config["tp_multiplier"])
        if direction == 0:
            return open_price - sl_distance, open_price + tp_distance
        return open_price + sl_distance, open_price - tp_distance


class SignalExtractor:
    def __init__(self, strategy: Any, sequence_length: int, batch_size: int = 1024, entry_price_type: str = "bid", stride: int = 1) -> None:
        if sequence_length <= 0 or batch_size <= 0 or stride <= 0:
            raise ValueError("sequence_length, batch_size, and stride must be positive.")
        if entry_price_type not in {"bid", "ask", "mid"}:
            raise ValueError("entry_price_type must be 'bid', 'ask', or 'mid'.")
        self.strategy, self.sequence_length = strategy, sequence_length
        self.batch_size, self.entry_price_type, self.stride = batch_size, entry_price_type, stride

    def extract(self, market: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        count = len(market) - self.sequence_length + 1
        if count <= 0:
            return pd.DataFrame(), 0
        window_starts = np.arange(0, count, self.stride)
        window_count = len(window_starts)
        method = next((getattr(self.strategy, name, None) for name in ("generate_signals", "signals", "predict") if callable(getattr(self.strategy, name, None))), None)
        if method is None:
            raise StrategyLoadError("Strategy must expose generate_signals(batch).")
        chunks = []
        columns = ("_timestamp", "open", "high", "low", "close", "spread", "real_volume", "tick_volume")
        print(f"[BACKTEST] extracting {window_count} windows in batches of {self.batch_size}", flush=True)
        for start in tqdm(
            range(0, window_count, self.batch_size),
            total=(window_count + self.batch_size - 1) // self.batch_size,
            desc="[BACKTEST] strategy batches",
            unit="batch",
        ):
            end = min(window_count, start + self.batch_size)
            batch_starts = window_starts[start:end]
            batch = {
                "time" if column == "_timestamp" else column: np.stack(
                    [market[column].to_numpy()[window_start:window_start + self.sequence_length] for window_start in batch_starts]
                )
                for column in columns
            }
            result = method(batch)
            values = np.asarray(result.get("direction") if isinstance(result, Mapping) else result)
            if values.ndim != 1 or len(values) != end - start:
                raise ValueError("Strategy directions must contain one value per input window.")
            if np.issubdtype(values.dtype, np.floating):
                if not np.all(np.isfinite(values)) or not np.all(np.mod(values, 1) == 0):
                    raise ValueError("Strategy directions must be integral.")
                values = values.astype(np.int64)
            chunks.append(values.astype(np.int64, copy=False))
        print("[BACKTEST] strategy batches complete", flush=True)
        directions = np.concatenate(chunks)
        unsupported = int((~np.isin(directions, (0, 1, 2))).sum())
        rows = []
        for window_number, direction in enumerate(directions):
            if direction not in (0, 1):
                continue
            row_index = self.sequence_length - 1 + window_number * self.stride
            row = market.iloc[row_index]
            spread = float(row.spread) if pd.notna(row.spread) else 0.0
            close = float(row.close)
            if self.entry_price_type == "bid":
                open_price = close + spread if direction == 0 else close
            elif self.entry_price_type == "ask":
                open_price = close if direction == 0 else close - spread
            else:
                open_price = close + spread / 2 if direction == 0 else close - spread / 2
            rows.append({"time": int(row._timestamp), "direction": int(direction), "open_price": open_price, "spread": spread, "row_index": row_index})
        return pd.DataFrame(rows), unsupported


@njit(cache=True)
def _run_core(
    order, open_time_sorted, direction, open_price, sl, tp,
    m_timestamps, high_bid, low_bid, high_ask, low_ask, close_col,
    n, n_bars,
):
    max_profit = np.zeros(n, dtype=np.float64)
    min_dd = np.zeros(n, dtype=np.float64)
    close_time = np.full(n, -1, dtype=np.int64)
    close_price = np.full(n, np.nan, dtype=np.float64)
    profit = np.full(n, np.nan, dtype=np.float64)
    close_reason = np.full(n, -1, dtype=np.int8)
    status = np.zeros(n, dtype=np.int8)

    profit_equity = np.zeros(n_bars, dtype=np.float64)
    dd_equity = np.zeros(n_bars, dtype=np.float64)
    active = np.empty(n, dtype=np.int64)
    active_len = 0
    activate_ptr = 0
    bars_used = n_bars

    for bar_i in range(n_bars):
        ts = m_timestamps[bar_i]
        hb = high_bid[bar_i]
        lb = low_bid[bar_i]
        ha = high_ask[bar_i]
        la = low_ask[bar_i]
        bar_close = close_col[bar_i]
        pe = 0.0
        de = 0.0
        write = 0

        for read in range(active_len):
            pid = active[read]
            position_direction = direction[pid]
            if position_direction == 0:
                mp = (hb if hb < tp[pid] else tp[pid]) - open_price[pid]
                dd = (lb if lb > sl[pid] else sl[pid]) - open_price[pid]
                sl_hit = lb <= sl[pid]
                tp_hit = hb >= tp[pid]
            else:
                mp = open_price[pid] - (la if la > tp[pid] else tp[pid])
                dd = open_price[pid] - (ha if ha < sl[pid] else sl[pid])
                sl_hit = ha >= sl[pid]
                tp_hit = la <= tp[pid]

            if mp > max_profit[pid]:
                max_profit[pid] = mp
            if dd < min_dd[pid]:
                min_dd[pid] = dd
            pe += max_profit[pid]
            de += min_dd[pid]

            if sl_hit or tp_hit:
                both = sl_hit and tp_hit
                if both:
                    hit_tp = (bar_close >= tp[pid]) if position_direction == 0 else (bar_close <= tp[pid])
                else:
                    hit_tp = tp_hit
                price = tp[pid] if hit_tp else sl[pid]
                status[pid] = 2
                close_time[pid] = ts
                close_price[pid] = price
                profit[pid] = price - open_price[pid] if position_direction == 0 else open_price[pid] - price
                close_reason[pid] = 3 if both else (0 if hit_tp else 1)
            else:
                active[write] = pid
                write += 1

        active_len = write
        profit_equity[bar_i] = pe
        dd_equity[bar_i] = de

        while activate_ptr < n and open_time_sorted[activate_ptr] == ts:
            pid = order[activate_ptr]
            status[pid] = 1
            active[active_len] = pid
            active_len += 1
            activate_ptr += 1

        if activate_ptr >= n and active_len == 0:
            bars_used = bar_i + 1
            break

    return (
        max_profit, min_dd, close_time, close_price, profit, close_reason,
        status, profit_equity, dd_equity, bars_used,
    )


class MarketTableBuilder:
    @staticmethod
    def build(dataframe: pd.DataFrame, point_size: float = 1.0) -> pd.DataFrame:
        point_size = float(point_size)
        if point_size <= 0:
            raise ValueError("point_size must be greater than zero.")
        frame = dataframe.reset_index() if "time" not in dataframe.columns and dataframe.index.name == "time" else dataframe.copy()
        required = {"time", "open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Backtest data is missing required columns: {sorted(missing)}")
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        if frame.time.isna().any():
            raise ValueError("The 'time' column contains invalid datetime values.")
        frame = frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        spread_values = frame["spread"] if "spread" in frame.columns else pd.Series(0.0, index=frame.index)
        frame["spread"] = pd.to_numeric(spread_values, errors="coerce").fillna(0.0) * float(point_size)
        for column in ("real_volume", "tick_volume"):
            if column not in frame.columns:
                frame[column] = 0.0
        frame["_timestamp"] = frame.time.astype("datetime64[s]").astype("int64")
        for column in ("open", "high", "low", "close", "spread"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        result = frame.set_index("_timestamp", drop=False)
        result.index.name = "time"
        result["high_bid"], result["low_bid"] = result.high, result.low
        result["high_ask"], result["low_ask"] = result.high + result.spread, result.low + result.spread
        return result


class BacktestEngine:
    def __init__(self, calculator: SLTPCalculator) -> None:
        self.calculator = calculator

    def run(self, signals: pd.DataFrame, market: pd.DataFrame, unsupported: int) -> BacktestResult:
        n = len(signals)
        open_price = np.empty(n, dtype=np.float64)
        direction = np.empty(n, dtype=np.int8)
        sl = np.empty(n, dtype=np.float64)
        tp = np.empty(n, dtype=np.float64)
        open_time = np.empty(n, dtype=np.int64)
        keep = np.zeros(n, dtype=bool)

        for i, signal in enumerate(signals.to_dict("records")):
            try:
                stop_loss, take_profit = self.calculator.compute(signal, market)
            except ValueError:
                continue
            open_price[i] = signal["open_price"]
            direction[i] = signal["direction"]
            sl[i] = stop_loss
            tp[i] = take_profit
            open_time[i] = signal["time"]
            keep[i] = True

        idx = np.nonzero(keep)[0]
        open_price, direction, sl, tp, open_time = (
            open_price[idx], direction[idx], sl[idx], tp[idx], open_time[idx]
        )
        n = len(idx)
        order = np.argsort(open_time, kind="stable")
        open_time_sorted = open_time[order]
        m_timestamps = market.index.to_numpy(dtype=np.int64)
        high_bid = market["high_bid"].to_numpy(dtype=np.float64)
        low_bid = market["low_bid"].to_numpy(dtype=np.float64)
        high_ask = market["high_ask"].to_numpy(dtype=np.float64)
        low_ask = market["low_ask"].to_numpy(dtype=np.float64)
        close_col = market["close"].to_numpy(dtype=np.float64)
        spread_col = market["spread"].to_numpy(dtype=np.float64)
        n_bars = len(market)

        replay_progress = tqdm(total=n_bars, desc="[BACKTEST] replaying bars", unit="bar")
        try:
            (
                max_profit, min_dd, close_time, close_price, profit, close_reason,
                status, profit_equity, dd_equity, bars_used,
            ) = _run_core(
                order, open_time_sorted, direction, open_price, sl, tp,
                m_timestamps, high_bid, low_bid, high_ask, low_ask, close_col,
                n, n_bars,
            )
            replay_progress.update(n_bars)
        finally:
            replay_progress.close()
        profit_equity = profit_equity[:bars_used]
        dd_equity = dd_equity[:bars_used]

        remaining = np.nonzero(status != 2)[0]
        if remaining.size:
            last_ts = int(m_timestamps[-1])
            last_close, last_spread = close_col[-1], spread_col[-1]
            remaining_directions = direction[remaining]
            remaining_close = np.where(remaining_directions == 0, last_close, last_close + last_spread)
            close_time[remaining] = last_ts
            close_price[remaining] = remaining_close
            profit[remaining] = np.where(remaining_directions == 0, remaining_close - open_price[remaining], open_price[remaining] - remaining_close)
            close_reason[remaining] = 2
            status[remaining] = 2

        reason_map = {0: "tp", 1: "sl", 2: "eod", 3: "tiebreak", -1: None}
        positions = pd.DataFrame({
            "open_time": open_time,
            "direction": direction,
            "status": pd.Categorical(["closed"] * n),
            "open_price": open_price,
            "sl": sl,
            "tp": tp,
            "max_profit": max_profit,
            "min_dd": min_dd,
            "close_time": pd.array(close_time, dtype="Int64"),
            "close_price": close_price,
            "profit": profit,
            "close_reason": pd.Categorical([reason_map[code] for code in close_reason]),
        }).set_index("open_time")
        return BacktestResult(
            positions,
            pd.Series(profit_equity, index=market.index[:len(profit_equity)], name="profit_equity"),
            pd.Series(dd_equity, index=market.index[:len(dd_equity)], name="dd_equity"),
            len(positions),
            int((positions.direction == 0).sum()) if len(positions) else 0,
            int((positions.direction == 1).sum()) if len(positions) else 0,
            unsupported,
        )

    @staticmethod
    def _close_hits(positions: pd.DataFrame, indexes: pd.Index, bar: Any, timestamp: int, is_buy: bool) -> None:
        rows = positions.loc[indexes]
        sl_hit = bar.low_bid <= rows.sl if is_buy else bar.high_ask >= rows.sl
        tp_hit = bar.high_bid >= rows.tp if is_buy else bar.low_ask <= rows.tp
        for index in rows.index[sl_hit | tp_hit]:
            both = bool(sl_hit.get(index, False) and tp_hit.get(index, False))
            hit_tp = bool(tp_hit.get(index, False) and (not both or (bar.close >= rows.at[index, "tp"] if is_buy else bar.close <= rows.at[index, "tp"])))
            price = rows.at[index, "tp"] if hit_tp else rows.at[index, "sl"]
            positions.at[index, "status"] = "closed"
            positions.at[index, "close_time"] = timestamp
            positions.at[index, "close_price"] = price
            positions.at[index, "profit"] = price - rows.at[index, "open_price"] if is_buy else rows.at[index, "open_price"] - price
            positions.at[index, "close_reason"] = "tiebreak" if both else ("tp" if hit_tp else "sl")


def _load_strategy(path: str | Path, sequence_length: int) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("fg_backtest_strategy", module_path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"Cannot load strategy file: {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise StrategyLoadError(f"Could not import strategy {module_path}: {error}") from error
    classes = [value for value in vars(module).values() if isinstance(value, type) and issubclass(value, PreprocessBase) and value is not PreprocessBase]
    if len(classes) != 1:
        raise StrategyLoadError("Strategy file must define exactly one PreprocessBase subclass.")
    try:
        return classes[0](sequence_length=sequence_length)
    except Exception as error:
        raise StrategyLoadError(f"Could not initialize strategy: {error}") from error


def run_backtest(strategy_path: str | Path, *, bucket_name: str, source: str, symbol_pair: str, instrument_group: Optional[str] = None, sequence_length: int = 60, stride: int = 1, batch_size: int = 1024, sl_calculation: Optional[Mapping[str, Any]] = None, entry_price_type: str = "bid", start_index: int = 0, end_index: int = -1, return_in_points: bool = False) -> BacktestResult:
    """Run a strategy against DataManager-compatible OHLC data.

    Args:
        strategy_path: Python file containing one ``PreprocessBase`` strategy.
        bucket_name: Storage bucket used by ``DataManager``.
        source: Market-data source name passed to ``DataManager``.
        symbol_pair: Symbol to backtest, for example ``"EURUSD"``.
        instrument_group: Optional data group, for example ``"forex_majors"``.
        sequence_length: Number of bars supplied to the strategy per window.
        stride: Number of bars between consecutive strategy windows.
        batch_size: Number of windows sent to the strategy at a time.
        sl_calculation: SL/TP mode configuration. ``None`` uses fixed mode with
            ``sl_points=100`` and ``tp_points=100``. For ``mode="fixed"``,
            accepted keys are ``mode``, ``sl_points`` and ``tp_points``; both
            point values default to ``100``. For ``mode="range"``, accepted
            keys are ``mode``, ``range``, ``sl_ratio`` and ``tp_ratio``; the
            defaults are ``range=60``, ``sl_ratio=1.0`` and ``tp_ratio=1.0``.
            For ``mode="atr"``, accepted keys are ``mode``,
            ``sl_multiplier``, ``tp_multiplier`` and ``atr_period``; the
            defaults are ``sl_multiplier=3.0``, ``tp_multiplier=3.0`` and
            ``atr_period=14``. Unknown keys or unsupported modes raise
            ``ValueError``. Point distances are converted using the symbol's
            ``point_size`` from its instrument properties.
        entry_price_type: Entry convention: ``"bid"`` (default), ``"ask"``,
            or ``"mid"``.
        start_index: Inclusive starting row in the normalized market table.
        end_index: Exclusive ending row; ``-1`` (default) means the final row.
        return_in_points: If true, divide position profit/drawdown fields and
            both equity curves by the symbol point size. Price fields remain
            in price units.

    Returns:
        A ``BacktestResult`` containing positions, equity curves, and counts.
    """
    print("[BACKTEST] loading market data", flush=True)
    properties = None
    dm = DataManager(base_bucket_name=bucket_name, source=source)
    data, properties = dm.load_data(symbol_pair=symbol_pair, instrument_group=instrument_group)
    print(f"[BACKTEST] loaded {len(data)} market rows", flush=True)
    point_size = float(getattr(properties, "point_size", 1.0))
    print("[BACKTEST] building market table", flush=True)
    market = MarketTableBuilder.build(data, point_size)
    if start_index < 0:
        raise ValueError("start_index must be greater than or equal to zero.")
    if end_index < -1:
        raise ValueError("end_index must be -1 or greater.")
    market = market.iloc[start_index:] if end_index == -1 else market.iloc[start_index:end_index]
    print(f"[BACKTEST] selected market rows {start_index}:{end_index}", flush=True)
    print("[BACKTEST] loading strategy", flush=True)
    strategy = _load_strategy(strategy_path, sequence_length)
    print("[BACKTEST] extracting signals", flush=True)
    signals, unsupported = SignalExtractor(strategy, sequence_length, batch_size, entry_price_type, stride).extract(market)
    print(f"[BACKTEST] extracted {len(signals)} supported signals", flush=True)
    print("[BACKTEST] running engine", flush=True)
    result = BacktestEngine(SLTPCalculator(sl_calculation, point_size)).run(signals, market, unsupported)
    if not return_in_points:
        return result

    positions = result.positions.copy()
    for column in ("max_profit", "min_dd", "profit"):
        positions[column] = positions[column] / point_size
    return BacktestResult(
        positions,
        result.profit_equity / point_size,
        result.dd_equity / point_size,
        result.positions_total,
        result.buy_count,
        result.sell_count,
        result.unsupported_signal_count,
    )
