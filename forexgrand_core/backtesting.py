"""Signal-level bar replay backtesting."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

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
        window_starts = range(0, count, self.stride)
        windows = {
            column: np.stack([market[column].to_numpy()[i:i + self.sequence_length] for i in window_starts])
            for column in ("_timestamp", "open", "high", "low", "close", "spread", "real_volume", "tick_volume")
        }
        windows["time"] = windows.pop("_timestamp")
        window_count = len(windows["time"])
        method = next((getattr(self.strategy, name, None) for name in ("generate_signals", "signals", "predict") if callable(getattr(self.strategy, name, None))), None)
        if method is None:
            raise StrategyLoadError("Strategy must expose generate_signals(batch).")
        chunks = []
        for start in range(0, window_count, self.batch_size):
            end = min(window_count, start + self.batch_size)
            result = method({key: value[start:end] for key, value in windows.items()})
            values = np.asarray(result.get("direction") if isinstance(result, Mapping) else result)
            if values.ndim != 1 or len(values) != end - start:
                raise ValueError("Strategy directions must contain one value per input window.")
            if np.issubdtype(values.dtype, np.floating):
                if not np.all(np.isfinite(values)) or not np.all(np.mod(values, 1) == 0):
                    raise ValueError("Strategy directions must be integral.")
                values = values.astype(np.int64)
            chunks.append(values.astype(np.int64, copy=False))
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


class MarketTableBuilder:
    @staticmethod
    def build(dataframe: pd.DataFrame) -> pd.DataFrame:
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
        frame["spread"] = pd.to_numeric(spread_values, errors="coerce").fillna(0.0)
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
        records = []
        for signal in signals.to_dict("records"):
            try:
                sl, tp = self.calculator.compute(signal, market)
            except ValueError:
                continue
            records.append({"open_time": signal.time, "direction": signal.direction, "status": "pending", "open_price": signal.open_price, "sl": sl, "tp": tp, "max_profit": 0.0, "min_dd": 0.0, "close_time": pd.NA, "close_price": np.nan, "profit": np.nan, "close_reason": None})
        columns = ["open_time", "direction", "status", "open_price", "sl", "tp", "max_profit", "min_dd", "close_time", "close_price", "profit", "close_reason"]
        positions = pd.DataFrame(records, columns=columns)
        profit_equity, dd_equity = [], []
        for timestamp, bar in market.iterrows():
            open_mask = positions.status.eq("open") if len(positions) else pd.Series(dtype=bool)
            if open_mask.any():
                current = positions.loc[open_mask]
                buy_index, sell_index = current.index[current.direction.eq(0)], current.index[current.direction.eq(1)]
                positions.loc[buy_index, "max_profit"] = np.maximum(current.loc[buy_index, "max_profit"], np.minimum(bar.high_bid, current.loc[buy_index, "tp"]) - current.loc[buy_index, "open_price"])
                positions.loc[buy_index, "min_dd"] = np.minimum(current.loc[buy_index, "min_dd"], np.maximum(bar.low_bid, current.loc[buy_index, "sl"]) - current.loc[buy_index, "open_price"])
                positions.loc[sell_index, "max_profit"] = np.maximum(current.loc[sell_index, "max_profit"], current.loc[sell_index, "open_price"] - np.maximum(bar.low_ask, current.loc[sell_index, "tp"]))
                positions.loc[sell_index, "min_dd"] = np.minimum(current.loc[sell_index, "min_dd"], current.loc[sell_index, "open_price"] - np.minimum(bar.high_ask, current.loc[sell_index, "sl"]))
                profit_equity.append(float(positions.loc[open_mask, "max_profit"].sum()))
                dd_equity.append(float(positions.loc[open_mask, "min_dd"].sum()))
                self._close_hits(positions, buy_index, bar, timestamp, True)
                self._close_hits(positions, sell_index, bar, timestamp, False)
            else:
                profit_equity.append(0.0)
                dd_equity.append(0.0)
            positions.loc[(positions.status == "pending") & (positions.open_time == timestamp), "status"] = "open"
            if not len(positions) or not positions.status.isin(["pending", "open"]).any():
                break
        if len(positions) and positions.status.isin(["pending", "open"]).any():
            timestamp, bar = int(market.iloc[-1]._timestamp), market.iloc[-1]
            pending = positions.status.isin(["pending", "open"])
            positions.loc[pending, "close_time"] = timestamp
            positions.loc[pending, "status"] = "closed"
            positions.loc[pending, "close_price"] = np.where(positions.loc[pending, "direction"].eq(0), bar.close, bar.close + bar.spread)
            positions.loc[pending, "profit"] = np.where(positions.loc[pending, "direction"].eq(0), positions.loc[pending, "close_price"] - positions.loc[pending, "open_price"], positions.loc[pending, "open_price"] - positions.loc[pending, "close_price"])
            positions.loc[pending, "close_reason"] = "eod"
        if len(positions):
            positions.direction = positions.direction.astype("int8")
            positions.close_time = positions.close_time.astype("Int64")
            positions.status = positions.status.astype("category")
            positions.close_reason = positions.close_reason.astype("category")
            positions = positions.set_index("open_time")
        return BacktestResult(positions, pd.Series(profit_equity, index=market.index[:len(profit_equity)], name="profit_equity"), pd.Series(dd_equity, index=market.index[:len(dd_equity)], name="dd_equity"), len(positions), int((positions.direction == 0).sum()) if len(positions) else 0, int((positions.direction == 1).sum()) if len(positions) else 0, unsupported)

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


def run_backtest(strategy_path: str | Path, *, data: Optional[pd.DataFrame] = None, data_manager: Optional[DataManager] = None, symbol_pair: Optional[str] = None, instrument_group: Optional[str] = None, sequence_length: int = 1, stride: int = 1, batch_size: int = 1024, sl_calculation: Optional[Mapping[str, Any]] = None, entry_price_type: str = "bid") -> BacktestResult:
    """Run a strategy against a DataManager-compatible OHLC dataframe."""
    properties = None
    if data is None:
        if data_manager is None or not symbol_pair or not instrument_group:
            raise ValueError("Provide data or data_manager with symbol_pair and instrument_group.")
        data, properties = data_manager.load_data(symbol_pair, instrument_group)
    point_size = float(getattr(properties, "point_size", 1.0))
    market = MarketTableBuilder.build(data)
    strategy = _load_strategy(strategy_path, sequence_length)
    signals, unsupported = SignalExtractor(strategy, sequence_length, batch_size, entry_price_type, stride).extract(market)
    return BacktestEngine(SLTPCalculator(sl_calculation, point_size)).run(signals, market, unsupported)
