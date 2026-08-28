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


# =====================================================================
# Lot sizing
# =====================================================================

LOTSIZE_FIXED = 0
LOTSIZE_PERCENTAGE = 1
LOTSIZE_MARTINGALE = 2
LOTSIZE_ANTIMARTINGALE = 3

_LOTSIZE_PARAM_LEN = 8


def FIXED_LOT(lots: float = 0.01) -> tuple[int, np.ndarray]:
    """Constant lot size regardless of account state or trade history."""
    if lots <= 0:
        raise ValueError("lots must be positive")
    params = np.zeros(_LOTSIZE_PARAM_LEN, dtype=np.float64)
    params[0] = float(lots)
    return LOTSIZE_FIXED, params


def PERCENTAGE(
    risk_ratio: float,
    point_value: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
) -> tuple[int, np.ndarray]:
    """Risk a fixed ratio of current balance per trade."""
    if not (0 < risk_ratio <= 1):
        raise ValueError("risk_ratio must be between 0 (exclusive) and 1 (inclusive)")
    if point_value <= 0:
        raise ValueError("point_value must be positive")
    if min_lot <= 0:
        raise ValueError("min_lot must be positive")
    if max_lot < min_lot:
        raise ValueError("max_lot must be >= min_lot")
    params = np.zeros(_LOTSIZE_PARAM_LEN, dtype=np.float64)
    params[0] = float(risk_ratio)
    params[4] = float(point_value)
    params[5] = float(min_lot)
    params[6] = float(max_lot)
    return LOTSIZE_PERCENTAGE, params


def _validate_ratio_progression(
    base: float, multiplier: float, max_steps: int,
    base_lot_mode: str, point_value: float, min_lot: float, max_lot: float,
) -> None:
    if base_lot_mode not in {"ratio", "lot"}:
        raise ValueError("base_lot_mode must be 'ratio' or 'lot'")
    if base_lot_mode == "ratio" and not (0 < base <= 1):
        raise ValueError("base must be between 0 (exclusive) and 1 (inclusive) in ratio mode")
    if base_lot_mode == "lot" and base <= 0:
        raise ValueError("base must be positive in lot mode")
    if multiplier < 1:
        raise ValueError("multiplier must be >= 1")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if point_value <= 0:
        raise ValueError("point_value must be positive")
    if min_lot <= 0:
        raise ValueError("min_lot must be positive")
    if max_lot < min_lot:
        raise ValueError("max_lot must be >= min_lot")


def _ratio_progression(
    type_code: int, base: float, multiplier: float, max_steps: int,
    hold_at_max_steps: bool, base_lot_mode: str, point_value: float,
    min_lot: float, max_lot: float,
) -> tuple[int, np.ndarray]:
    _validate_ratio_progression(base, multiplier, max_steps, base_lot_mode,
                                point_value, min_lot, max_lot)
    params = np.zeros(_LOTSIZE_PARAM_LEN, dtype=np.float64)
    params[0] = float(base)
    params[1] = float(multiplier)
    params[2] = float(max_steps)
    params[3] = 1.0 if hold_at_max_steps else 0.0
    params[4] = float(point_value)
    params[5] = float(min_lot)
    params[6] = float(max_lot)
    params[7] = 1.0 if base_lot_mode == "lot" else 0.0
    return type_code, params


def MARTINGALE(base: float = 0.01, multiplier: float = 2.0, max_steps: int = 4,
               hold_at_max_steps: bool = False, point_value: float = 1.0,
               min_lot: float = 0.01, max_lot: float = 100.0,
               base_lot_mode: str = "ratio") -> tuple[int, np.ndarray]:
    """Scale a ratio or lot value up on consecutive losing closes."""
    return _ratio_progression(LOTSIZE_MARTINGALE, base, multiplier, max_steps,
                              hold_at_max_steps, base_lot_mode, point_value,
                              min_lot, max_lot)


def ANTIMARTINGALE(base: float = 0.01, multiplier: float = 2.0, max_steps: int = 4,
                   hold_at_max_steps: bool = False, point_value: float = 1.0,
                   min_lot: float = 0.01, max_lot: float = 100.0,
                   base_lot_mode: str = "ratio") -> tuple[int, np.ndarray]:
    """Scale a ratio or lot value up on consecutive winning closes."""
    return _ratio_progression(LOTSIZE_ANTIMARTINGALE, base, multiplier, max_steps,
                              hold_at_max_steps, base_lot_mode, point_value,
                              min_lot, max_lot)


LOTSIZERS = [
    ("FIXED_LOT", LOTSIZE_FIXED, FIXED_LOT),
    ("PERCENTAGE", LOTSIZE_PERCENTAGE, PERCENTAGE),
    ("MARTINGALE", LOTSIZE_MARTINGALE, MARTINGALE),
    ("ANTIMARTINGALE", LOTSIZE_ANTIMARTINGALE, ANTIMARTINGALE),
]


trade_result_only = "trade_result"
statistics_only = "statistics"
trade_result_and_statistics = "trade_result_and_statistics"
# Short aliases make the public factory pleasant to use from this module.
trade_result = trade_result_only
statistics = statistics_only


@dataclass(frozen=True)
class ResultType:
    """Select the backtest payload and whether monetary values are percentages."""

    type: str = trade_result_only
    return_percent: bool = False

    def __post_init__(self) -> None:
        if self.type not in {trade_result_only, statistics_only, trade_result_and_statistics}:
            raise ValueError(
                "type must be 'trade_result', 'statistics', or "
                "'trade_result_and_statistics'"
            )


def RESULT_TYPE(type: str = trade_result_only, return_percent: bool = False) -> ResultType:
    """Build a result selection, for example ``RESULT_TYPE(statistics)``."""
    return ResultType(type=type, return_percent=return_percent)


@njit(cache=True, inline="always")
def _round2(x: float) -> float:
    return np.floor(x * 100.0 + 0.5) / 100.0


@njit(cache=True, inline="always")
def _ratio_to_lot(ratio: float, balance: float, sl_points: float,
                  point_value: float, min_lot: float, max_lot: float) -> float:
    if sl_points <= 0.0:
        return min_lot
    lot = ratio * (balance / point_value) / sl_points
    return _round2(min_lot if lot < min_lot else max_lot if lot > max_lot else lot)


@njit(cache=True, inline="always")
def _compute_lot(lotsizer_type: int, params: np.ndarray, balance: float,
                 sl_points: float, win_streak: int, loss_streak: int) -> float:
    if lotsizer_type == LOTSIZE_FIXED:
        return params[0]
    if lotsizer_type == LOTSIZE_PERCENTAGE:
        return _ratio_to_lot(params[0], balance, sl_points, params[4], params[5], params[6])
    if lotsizer_type == LOTSIZE_MARTINGALE or lotsizer_type == LOTSIZE_ANTIMARTINGALE:
        streak = loss_streak if lotsizer_type == LOTSIZE_MARTINGALE else win_streak
        max_steps = int(params[2])
        steps = min(streak, max_steps)
        if max_steps > 0 and steps == max_steps and params[3] <= 0.5:
            steps = streak % max_steps
        calculated_value = params[0] * (params[1] ** steps)
        if params[7] > 0.5:
            return calculated_value
        return _ratio_to_lot(calculated_value, balance, sl_points,
                             params[4], params[5], params[6])
    return params[0]


# =====================================================================
# SL / TP calculation
# =====================================================================

def FIXED_SLTP(sl_points: float = 100, tp_points: float = 100) -> dict:
    return {"mode": "fixed", "sl_points": sl_points, "tp_points": tp_points}


def RANGE(range: int = 60, sl_ratio: float = 1.0, tp_ratio: float = 1.0) -> dict:
    return {"mode": "range", "range": range, "sl_ratio": sl_ratio, "tp_ratio": tp_ratio}


def ATR(sl_multiplier: float = 3.0, tp_multiplier: float = 3.0, atr_period: int = 14) -> dict:
    return {"mode": "atr", "sl_multiplier": sl_multiplier, "tp_multiplier": tp_multiplier,
            "atr_period": atr_period}


SLTP_MODES = [("FIXED_SLTP", FIXED_SLTP), ("RANGE", RANGE), ("ATR", ATR)]


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    balance_equity: pd.Series
    equity_curve: pd.Series
    positions_total: int
    buy_count: int
    sell_count: int
    unsupported_signal_count: int
    final_balance: float
    statistics: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @property
    def profit_equity(self) -> pd.Series:
        """Compatibility alias for the realized balance curve."""
        return self.balance_equity

    @property
    def dd_equity(self) -> pd.Series:
        """Compatibility alias for floating drawdown relative to balance."""
        return self.balance_equity - self.equity_curve


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
    n, n_bars, lotsizer_type, lotsizer_params, point_size, initial_balance,
):
    close_time = np.full(n, -1, dtype=np.int64)
    close_price = np.full(n, np.nan, dtype=np.float64)
    profit = np.full(n, np.nan, dtype=np.float64)
    close_reason = np.full(n, -1, dtype=np.int8)
    status = np.zeros(n, dtype=np.int8)
    lots = np.zeros(n, dtype=np.float64)

    balance_equity = np.zeros(n_bars, dtype=np.float64)
    equity_curve = np.zeros(n_bars, dtype=np.float64)
    active = np.empty(n, dtype=np.int64)
    active_len = 0
    activate_ptr = 0
    bars_used = n_bars
    balance = initial_balance
    win_streak = 0
    loss_streak = 0

    for bar_i in range(n_bars):
        ts = m_timestamps[bar_i]
        hb = high_bid[bar_i]
        lb = low_bid[bar_i]
        ha = high_ask[bar_i]
        la = low_ask[bar_i]
        bar_close = close_col[bar_i]
        floating = 0.0
        write = 0

        for read in range(active_len):
            pid = active[read]
            position_direction = direction[pid]
            if position_direction == 0:
                sl_hit = lb <= sl[pid]
                tp_hit = hb >= tp[pid]
            else:
                sl_hit = ha >= sl[pid]
                tp_hit = la <= tp[pid]

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
                pos_profit = price - open_price[pid] if position_direction == 0 else open_price[pid] - price
                q_profit = lots[pid] * (pos_profit / point_size)
                profit[pid] = q_profit
                close_reason[pid] = 3 if both else (0 if hit_tp else 1)
                balance += q_profit
                if q_profit > 0.0:
                    win_streak += 1
                    loss_streak = 0
                elif q_profit < 0.0:
                    loss_streak += 1
                    win_streak = 0
                else:
                    win_streak = 0
                    loss_streak = 0
            else:
                active[write] = pid
                write += 1
                floating_price_diff = (bar_close - open_price[pid]) if position_direction == 0 else (open_price[pid] - bar_close)
                floating += lots[pid] * (floating_price_diff / point_size)

        active_len = write
        balance_equity[bar_i] = balance
        equity_curve[bar_i] = balance + floating

        while activate_ptr < n and open_time_sorted[activate_ptr] == ts:
            pid = order[activate_ptr]
            status[pid] = 1
            sl_points = abs(open_price[pid] - sl[pid]) / point_size
            lots[pid] = _compute_lot(lotsizer_type, lotsizer_params, balance, sl_points, win_streak, loss_streak)
            active[active_len] = pid
            active_len += 1
            activate_ptr += 1

        if activate_ptr >= n and active_len == 0:
            bars_used = bar_i + 1
            break

    return close_time, close_price, profit, close_reason, status, bars_used, lots, balance_equity, equity_curve, balance


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
    def __init__(self, calculator: SLTPCalculator,
                 lot_sizing: Optional[tuple[int, np.ndarray]] = None,
                 initial_balance: float = 10_000.0) -> None:
        self.calculator = calculator
        self.lot_sizing = lot_sizing if lot_sizing is not None else FIXED_LOT(lots=1.0)
        self.initial_balance = float(initial_balance)

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
        n_bars = len(market)
        lotsizer_type, lotsizer_params = self.lot_sizing
        point_size = self.calculator.symbol_points

        replay_progress = tqdm(total=n_bars, desc="[BACKTEST] replaying bars", unit="bar")
        try:
            (
                close_time, close_price, profit, close_reason, status, bars_used,
                lots, balance_equity, equity_curve, final_balance,
            ) = _run_core(
                order, open_time_sorted, direction, open_price, sl, tp,
                m_timestamps, high_bid, low_bid, high_ask, low_ask, close_col,
                n, n_bars, lotsizer_type, lotsizer_params, point_size, self.initial_balance,
            )
            replay_progress.update(n_bars)
        finally:
            replay_progress.close()
        balance_equity = balance_equity[:bars_used]
        equity_curve = equity_curve[:bars_used]

        remaining = np.nonzero(status != 2)[0]
        if remaining.size:
            last_ts = int(m_timestamps[-1])
            last_close, last_spread = close_col[-1], market["spread"].to_numpy(dtype=np.float64)[-1]
            remaining_directions = direction[remaining]
            remaining_close = np.where(remaining_directions == 0, last_close, last_close + last_spread)
            close_time[remaining] = last_ts
            close_price[remaining] = remaining_close
            remaining_price_diff = np.where(remaining_directions == 0, remaining_close - open_price[remaining], open_price[remaining] - remaining_close)
            profit[remaining] = lots[remaining] * (remaining_price_diff / point_size)
            close_reason[remaining] = 2
            status[remaining] = 2
            final_balance += float(profit[remaining].sum())
            if len(balance_equity):
                balance_equity[-1] = final_balance
                equity_curve[-1] = final_balance

        reason_map = {0: "tp", 1: "sl", 2: "eod", 3: "tiebreak", -1: None}
        positions = pd.DataFrame({
            "open_time": open_time,
            "direction": direction,
            "status": pd.Categorical(["closed"] * n),
            "open_price": open_price,
            "sl": sl,
            "tp": tp,
            "lot": lots,
            "close_time": pd.array(close_time, dtype="Int64"),
            "time_spent": close_time - open_time,
            "close_price": close_price,
            "profit": profit,
            "close_reason": pd.Categorical([reason_map[code] for code in close_reason]),
        }).set_index("open_time")
        return BacktestResult(
            positions,
            pd.Series(balance_equity, index=market.index[:len(balance_equity)], name="balance_equity"),
            pd.Series(equity_curve, index=market.index[:len(equity_curve)], name="equity_curve"),
            len(positions),
            int((positions.direction == 0).sum()) if len(positions) else 0,
            int((positions.direction == 1).sum()) if len(positions) else 0,
            unsupported,
            float(final_balance),
        )


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


def _percentage_result(result: BacktestResult) -> BacktestResult:
    """Convert realized P&L and curves to percentage returns from their base."""
    from forexgrand_core.trade_statistics import lookup_value_at_time

    balance = result.balance_equity.to_numpy(dtype=float)
    initial = float(balance[0]) if len(balance) else 0.0
    positions = result.positions.copy()
    if len(positions) and len(balance):
        open_times = positions.index.to_numpy(dtype=np.int64)
        base = lookup_value_at_time(
            open_times,
            result.balance_equity.index.to_numpy(dtype=np.int64),
            balance,
        )
        positions["profit"] = np.where(base != 0, positions["profit"] / base * 100.0, np.nan)
    if initial:
        balance_curve = (result.balance_equity - initial) / initial * 100.0
        equity_curve = (result.equity_curve - initial) / initial * 100.0
    else:
        balance_curve = result.balance_equity * np.nan
        equity_curve = result.equity_curve * np.nan
    return BacktestResult(
        positions, balance_curve, equity_curve, result.positions_total,
        result.buy_count, result.sell_count, result.unsupported_signal_count,
        result.final_balance, result.statistics,
    )


def _statistics_input(result: BacktestResult, percentage: bool = False) -> dict[str, Any]:
    positions = result.positions.reset_index()
    positions = positions.rename(columns={positions.columns[0]: "open_time"})
    equity = result.equity_curve.to_numpy(dtype=float)
    balance = result.balance_equity.to_numpy(dtype=float)
    if percentage:
        equity, balance = equity + 100.0, balance + 100.0
    return {
        "positions": positions,
        "equity": equity,
        "balance": balance,
        "equity_time": result.equity_curve.index.to_numpy(dtype=np.int64),
    }


def run_backtest(
    strategy_path: str | Path,
    *,
    bucket_name: str,
    source: str,
    symbol_pair: str,
    instrument_group: Optional[str] = None,
    sequence_length: int = 60,
    stride: int = 1,
    batch_size: int = 1024,
    sl_calculation: Optional[Mapping[str, Any]] = None,
    lot_sizing: Optional[tuple[int, np.ndarray]] = None,
    initial_balance: float = 10_000.0,
    entry_price_type: str = "bid",
    start_index: int = 0,
    end_index: int = -1,
    result_type: Optional[ResultType] = None,
    return_in_points: bool = False,
) -> Any:
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
        lot_sizing: Lot sizer configuration built with ``FIXED_LOT``,
            ``PERCENTAGE``, ``MARTINGALE``, or ``ANTIMARTINGALE``. ``None``
            defaults to ``FIXED_LOT(lots=1.0)``. Martingale and
            antimartingale accept ``base_lot_mode="ratio"`` (the default,
            internal mode 0) or ``base_lot_mode="lot"`` (direct calculated
            lot value, internal mode 1).
        initial_balance: Starting account balance used by balance-dependent
            lot sizing and both equity curves.
        entry_price_type: Entry convention: ``"bid"`` (default), ``"ask"``,
            or ``"mid"``.
        start_index: Inclusive starting row in the normalized market table.
        end_index: Exclusive ending row; ``-1`` (default) means the final row.
        result_type: ``RESULT_TYPE(...)`` selection. Defaults to trade results.
            Use ``RESULT_TYPE(statistics)`` for statistics only or
            ``RESULT_TYPE(trade_result_and_statistics)`` for both.
        return_in_points: Deprecated compatibility option. Point conversion is
            already represented by the backtest's monetary P&L contract.
    Returns:
        A ``BacktestResult`` containing positions, equity curves, counts, and
        final realized balance.
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
    result = BacktestEngine(
        SLTPCalculator(sl_calculation, point_size),
        lot_sizing=lot_sizing,
        initial_balance=initial_balance,
    ).run(signals, market, unsupported)
    selection = result_type or ResultType()
    if not isinstance(selection, ResultType):
        raise TypeError("result_type must be created with RESULT_TYPE(...)")
    computed = None
    if selection.type in {statistics_only, trade_result_and_statistics}:
        from forexgrand_core.trade_statistics import compute_statistics
        computed = compute_statistics(_statistics_input(result))
        if selection.type == statistics_only:
            return computed
    if selection.return_percent:
        result = _percentage_result(result)
    if computed is not None:
        result.statistics = computed
    return result
