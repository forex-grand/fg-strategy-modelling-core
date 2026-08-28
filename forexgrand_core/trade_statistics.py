"""Vectorized trade and equity statistics for backtest results."""
from __future__ import annotations

import math
import time as _time
from typing import Optional

import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return args[0] if args and callable(args[0]) and not kwargs else decorate


DEFAULT_QUANTILES = (25, 50, 75, 95, 99)
DD_QUANTILES = (25, 50, 75, 100)
RECOVERY_QUANTILES = (50, 95)
SECONDS_PER_YEAR = 365.25 * 86400
TRADING_DAYS_PER_YEAR = 252
REQUIRED_POSITION_COLUMNS = ("profit", "time_spent", "open_time", "close_time")
TRADE_TIME_FIELDS = ("open_time", "close_time")


def quantiles(values, qs=DEFAULT_QUANTILES, abs_value=False):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if abs_value:
        values = np.abs(values)
    return {f"p{q}": float(np.percentile(values, q)) if values.size else float("nan") for q in qs}


def safe_divide(a, b, default=0.0):
    return default if float(b) == 0 else float(a) / float(b)


def moments_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"n": int(values.size), "variance": float("nan"), "skewness": float("nan"), "kurtosis": float("nan")}
    centered = values - values.mean()
    m2, m3, m4 = np.mean(centered ** 2), np.mean(centered ** 3), np.mean(centered ** 4)
    return {"n": int(values.size), "variance": float(np.var(values, ddof=1)),
            "skewness": float(m3 / m2 ** 1.5) if m2 else float("nan"),
            "kurtosis": float(m4 / m2 ** 2 - 3) if m2 else float("nan")}


def distribution_stats(values, qs=DEFAULT_QUANTILES):
    values = np.asarray(values, dtype=np.float64)
    result = moments_stats(values)
    result.update(mean=float(np.nanmean(values)) if values.size else float("nan"), quantiles=quantiles(values, qs))
    return result


def lookup_value_at_time(query, reference_times, reference_values):
    query, reference_times = np.asarray(query), np.asarray(reference_times)
    indices = np.searchsorted(reference_times, query, side="right") - 1
    return np.asarray(reference_values)[np.clip(indices, 0, len(reference_values) - 1)]


def _walk_drawdown(values):
    starts, troughs, ends, peaks, trough_values = [], [], [], [], []
    peak, peak_i, trough, trough_i, in_dd = values[0], 0, values[0], 0, False
    for i, value in enumerate(values[1:], 1):
        if value >= peak:
            if in_dd:
                starts.append(peak_i); troughs.append(trough_i); ends.append(i)
                peaks.append(peak); trough_values.append(trough); in_dd = False
            peak, peak_i = value, i
        elif not in_dd:
            in_dd, trough, trough_i = True, value, i
        elif value < trough:
            trough, trough_i = value, i
    if in_dd:
        starts.append(peak_i); troughs.append(trough_i); ends.append(-1)
        peaks.append(peak); trough_values.append(trough)
    return (np.asarray(starts, dtype=np.int64), np.asarray(troughs, dtype=np.int64),
            np.asarray(ends, dtype=np.int64), np.asarray(peaks), np.asarray(trough_values))


def _period_groups(times):
    dt = pd.to_datetime(times, unit="s", utc=True)
    def groups(keys):
        return {key: values.to_numpy() for key, values in pd.Series(np.arange(len(times)), index=keys).groupby(level=0, sort=True)}
    return {
        "hour_of_day_idx": groups(dt.hour.to_numpy()), "day_of_week_idx": groups(dt.dayofweek.to_numpy()),
        "month_of_year_idx": groups(dt.month.to_numpy()), "daily_idx": groups(dt.floor("D")),
        "weekly_idx": groups(dt.to_period("W")), "monthly_idx": groups(dt.to_period("M")),
    }


def build_index_collections(positions, trade_time_field="open_time"):
    if trade_time_field not in TRADE_TIME_FIELDS:
        raise ValueError("trade_time_field must be 'open_time' or 'close_time'")
    profit = positions.profit.to_numpy(float)
    times = positions[trade_time_field].to_numpy(np.int64)
    result = {"times": times, "win_idx": np.flatnonzero(profit > 0), "loss_idx": np.flatnonzero(profit <= 0)}
    result.update(_period_groups(times))
    return result


def compute_duration_stats(positions, win_idx, loss_idx, qs=DEFAULT_QUANTILES):
    duration = positions.time_spent.to_numpy(float); profit = positions.profit.to_numpy(float)
    def split(index):
        values, profits = duration[index], profit[index]
        if not len(values): return {"median_duration": float("nan"), "below_median": None, "above_median": None}
        median = float(np.median(values)); output = {"median_duration": median}
        for name, mask in (("below_median", values <= median), ("above_median", values > median)):
            output[name] = {"count": int(mask.sum()), "mean_duration": float(values[mask].mean()) if mask.any() else float("nan"),
                            "mean_profit": float(profits[mask].mean()) if mask.any() else float("nan"),
                            "sum_profit": float(profits[mask].sum()) if mask.any() else float("nan")}
        return output
    return {"all_trades": quantiles(duration, qs), "win_trades": quantiles(duration[win_idx], qs),
            "loss_trades": quantiles(duration[loss_idx], qs), "win_duration_split": split(win_idx), "loss_duration_split": split(loss_idx)}


def compute_distribution_stats(positions, equity_time, balance, qs=DEFAULT_QUANTILES, trade_time_field="open_time"):
    profit = positions.profit.to_numpy(float); trade_times = positions[trade_time_field].to_numpy(np.int64)
    base = lookup_value_at_time(trade_times, equity_time, balance)
    returns = np.where(base != 0, profit / base * 100, np.nan)
    return {"profit": distribution_stats(profit, qs), "returns_pct": distribution_stats(returns, qs)}


def compute_drawdown_stats(equity, balance, equity_time):
    def one(values, label):
        starts, troughs, ends, peaks, lows = _walk_drawdown(values)
        absolute = peaks - lows; pct = np.where(peaks != 0, absolute / peaks * 100, np.nan)
        recovered = ends >= 0; end_times = np.where(recovered, equity_time[np.maximum(ends, 0)], -1)
        return {"count": len(starts), "recovered_count": int(recovered.sum()), "start_idx": starts, "trough_idx": troughs,
                "end_idx": ends, "dd_abs": absolute, "dd_pct": pct, "dd_pct_quantiles": quantiles(pct, DD_QUANTILES),
                "time_to_trough_quantiles": quantiles(equity_time[troughs] - equity_time[starts]),
                "time_to_recover_quantiles": quantiles((end_times - equity_time[troughs])[recovered], RECOVERY_QUANTILES),
                "max_dd_abs": float(absolute.max()) if len(absolute) else float("nan"),
                "max_dd_pct": float(np.nanmax(pct)) if len(pct) else float("nan")}
    return {"balance_dd": one(balance, "balance"), "relative_dd": _relative_dd(equity, balance, equity_time), "running_peak_balance": np.maximum.accumulate(balance)}


def _relative_dd(equity, balance, times):
    dd = balance - equity; positive = dd > 0; starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    ends = np.array([next((i for i in range(start + 1, len(dd)) if not positive[i]), -1) for start in starts], dtype=np.int64)
    extremes = np.array([start + int(np.argmax(dd[start:(end if end >= 0 else len(dd))])) for start, end in zip(starts, ends)], dtype=np.int64)
    maximum = dd[extremes] if len(extremes) else np.array([], dtype=float)
    peak_ref = np.maximum.accumulate(balance)[starts] if len(starts) else np.array([], dtype=float)
    recovered = ends >= 0; end_times = np.where(recovered, times[np.maximum(ends, 0)], -1)
    pct = np.where(peak_ref != 0, maximum / peak_ref * 100, np.nan)
    return {"count": len(starts), "recovered_count": int(recovered.sum()), "start_idx": starts, "extreme_idx": extremes, "end_idx": ends,
            "dd_abs": maximum, "dd_pct": pct, "dd_pct_quantiles": quantiles(pct, DD_QUANTILES),
            "time_to_extreme_quantiles": quantiles(times[extremes] - times[starts]),
            "time_to_recover_quantiles": quantiles((end_times - times[extremes])[recovered], RECOVERY_QUANTILES),
            "max_dd_abs": float(maximum.max()) if len(maximum) else float("nan"), "max_dd_pct": float(np.nanmax(pct)) if len(pct) else float("nan")}


def _table(profit, groups):
    return [{"period": str(key), "num_trades": len(index), "num_wins": int((profit[index] > 0).sum()),
             "num_losses": int((profit[index] <= 0).sum()), "win_ratio": safe_divide((profit[index] > 0).sum(), len(index)),
             "net_profit": float(profit[index].sum())} for key, index in sorted(groups.items(), key=lambda item: str(item[0]))]


def compute_trade_count_stats(positions, idx):
    profit = positions.profit.to_numpy(float); return {"win_rate": safe_divide(len(idx["win_idx"]), len(profit)),
        "daily": _table(profit, idx["daily_idx"]), "weekly": _table(profit, idx["weekly_idx"]), "monthly": _table(profit, idx["monthly_idx"])}


def compute_trade_profit_stats(positions, idx, equity, balance, equity_time, max_balance_dd_abs, qs=DEFAULT_QUANTILES, trading_days_per_year=TRADING_DAYS_PER_YEAR):
    profit = positions.profit.to_numpy(float); wins, losses = profit[idx["win_idx"]], profit[idx["loss_idx"]]
    initial, final = float(balance[0]), float(balance[-1]); years = (equity_time[-1] - equity_time[0]) / SECONDS_PER_YEAR
    daily = np.asarray([profit[index].sum() for index in idx["daily_idx"].values()])
    daily_equity = pd.Series(equity, index=pd.to_datetime(equity_time, unit="s", utc=True)).resample("D").last().dropna().to_numpy()
    returns = np.diff(daily_equity) / daily_equity[:-1] if len(daily_equity) > 1 else np.array([])
    trade_bases = lookup_value_at_time(positions.open_time.to_numpy(np.int64), equity_time, balance)
    trade_returns_pct = np.where(trade_bases != 0, profit / trade_bases * 100, np.nan)
    risk_column = next((column for column in ("risk", "risk_amount", "risk_pct") if column in positions), None)
    return {"net_profit_abs": float(profit.sum()), "net_profit_pct": float(np.nansum(trade_returns_pct)),
        "gross_profit": float(wins.sum()), "gross_loss": float(losses.sum()),
        "profit_factor": safe_divide(wins.sum(), abs(losses.sum()), float("inf") if wins.sum() > 0 else float("nan")),
        "recovery_factor": safe_divide(profit.sum(), max_balance_dd_abs, float("nan")),
        "cagr": (final / initial) ** (1 / years) - 1 if years > 0 and initial > 0 and final > 0 else float("nan"),
        "sharpe_ratio": float(returns.mean() / returns.std(ddof=1) * math.sqrt(trading_days_per_year)) if len(returns) > 1 and returns.std(ddof=1) else float("nan"),
        "profit_quantiles": quantiles(profit, qs), "win_profit_quantiles": quantiles(wins, qs), "loss_profit_quantiles": quantiles(losses, qs, True),
        "daily_net_profit_quantiles": quantiles(daily, qs), "daily_win_amount_quantiles": quantiles(daily[daily > 0], qs),
        "daily_loss_amount_quantiles": quantiles(daily[daily < 0], qs, True), "daily_net_profit_std": float(np.std(daily, ddof=1)) if len(daily) > 1 else float("nan"),
        "trade_risk_quantiles": quantiles(positions[risk_column].to_numpy(float), qs) if risk_column else None,
        "trade_risk_column_used": risk_column}


def compute_position_sizing_stats(positions, idx, qs=DEFAULT_QUANTILES):
    column = next((c for c in ("size", "volume", "lots", "position_size") if c in positions), None)
    if column is None: return None
    values = positions[column].to_numpy(float); return {"column_used": column, "all": quantiles(values, qs), "win": quantiles(values[idx["win_idx"]], qs), "loss": quantiles(values[idx["loss_idx"]], qs)}


def compute_streak_stats(positions, equity_time, balance, qs=DEFAULT_QUANTILES, trade_time_field="open_time"):
    profit = positions.profit.to_numpy(float); wins = profit > 0; starts = np.r_[0, np.flatnonzero(wins[1:] != wins[:-1]) + 1]; ends = np.r_[starts[1:] - 1, len(profit) - 1]
    if not len(profit):
        empty = np.array([], dtype=np.int64)
        return {"num_win_streaks": 0, "num_loss_streaks": 0, "max_win_streak": 0, "max_loss_streak": 0,
                "win_streak_length_quantiles": quantiles(empty, qs), "loss_streak_length_quantiles": quantiles(empty, qs),
                "win_streak_profit_abs_quantiles": quantiles(empty, qs), "loss_streak_profit_abs_quantiles": quantiles(empty, qs, True),
                "win_streak_duration_quantiles": quantiles(empty, qs), "loss_streak_duration_quantiles": quantiles(empty, qs),
                "start_idx": empty, "end_idx": empty, "streak_type": empty}
    lengths = ends - starts + 1; sums = np.asarray([profit[start:end + 1].sum() for start, end in zip(starts, ends)]); types = wins[starts]
    trade_times = positions[trade_time_field].to_numpy(np.int64)
    durations = trade_times[ends] - trade_times[starts]
    def q(values, mask, absolute=False): return quantiles(values[mask], qs, absolute)
    base = lookup_value_at_time(trade_times[starts], equity_time, balance)
    profit_pct = np.where(base != 0, sums / base * 100, np.nan)
    return {"num_win_streaks": int(types.sum()), "num_loss_streaks": int((~types).sum()), "max_win_streak": int(lengths[types].max()) if types.any() else 0,
        "max_loss_streak": int(lengths[~types].max()) if (~types).any() else 0, "win_streak_length_quantiles": q(lengths, types), "loss_streak_length_quantiles": q(lengths, ~types),
        "win_streak_profit_abs_quantiles": q(sums, types), "loss_streak_profit_abs_quantiles": q(sums, ~types, True),
        "win_streak_profit_pct_quantiles": q(profit_pct, types), "loss_streak_profit_pct_quantiles": q(profit_pct, ~types, True),
        "win_streak_duration_quantiles": q(durations, types), "loss_streak_duration_quantiles": q(durations, ~types),
        "start_idx": starts, "end_idx": ends, "streak_type": types.astype(np.int64)}


def compute_monte_carlo_stats(positions, balance, n_sims=2000, worst_pct=5, best_pct=95, seed=None, max_chunk_elems=20_000_000):
    profit = positions.profit.to_numpy(float); n = len(profit)
    if not n: return None
    rng = np.random.default_rng(seed); output = np.empty(n_sims); chunk = max(1, max_chunk_elems // n)
    for start in range(0, n_sims, chunk): output[start:start + chunk] = profit[rng.integers(0, n, (min(chunk, n_sims - start), n))].sum(1)
    output = output / balance[0] * 100 if balance[0] else np.full(n_sims, np.nan)
    return {"n_simulations": n_sims, "median_return_pct": float(np.median(output)), "worst_case_pct": float(np.percentile(output, worst_pct)), "best_case_pct": float(np.percentile(output, best_pct)), "worst_percentile_used": worst_pct, "best_percentile_used": best_pct}


def compute_walk_forward_stats(positions, split_fraction=0.7):
    if len(positions) < 10: return None
    split = int(len(positions) * split_fraction)
    def metrics(frame):
        p = frame.profit.to_numpy(float); wins, losses = p[p > 0], p[p <= 0]
        return {"net_profit": float(p.sum()), "profit_factor": safe_divide(wins.sum(), abs(losses.sum()), float("inf") if wins.sum() > 0 else float("nan")), "win_rate": safe_divide(len(wins), len(p)), "n_trades": len(p)}
    inside, outside = metrics(positions.iloc[:split]), metrics(positions.iloc[split:]); efficiency = safe_divide(outside["net_profit"], inside["net_profit"], float("nan"))
    return {"split_fraction": split_fraction, "in_sample": inside, "out_of_sample": outside, "walk_forward_efficiency": efficiency, "performance_degradation": 1 - efficiency if np.isfinite(efficiency) else float("nan")}


def compute_seasonal_stats(positions, idx):
    profit = positions.profit.to_numpy(float)
    def buckets(groups): return {str(key): {"num_trades": len(index), "num_wins": int((profit[index] > 0).sum()), "num_losses": int((profit[index] <= 0).sum()), "win_rate": safe_divide((profit[index] > 0).sum(), len(index)), "net_profit": float(profit[index].sum())} for key, index in groups.items()}
    monthly = {str(k): float(profit[v].sum()) for k, v in idx["monthly_idx"].items()}
    best = max(monthly.items(), key=lambda item: item[1]) if monthly else (None, float("nan")); worst = min(monthly.items(), key=lambda item: item[1]) if monthly else (None, float("nan"))
    return {"hourly": buckets(idx["hour_of_day_idx"]), "day_of_week": buckets(idx["day_of_week_idx"]), "month_of_year": buckets(idx["month_of_year_idx"]), "best_month": {"period": best[0], "net_profit": best[1]}, "worst_month": {"period": worst[0], "net_profit": worst[1]}}


def compute_trade_efficiency_stats(positions, qs=DEFAULT_QUANTILES):
    output = {}; 
    if "mfe" in positions: output["favorable_excursion_quantiles"] = quantiles(positions.mfe, qs)
    if "mae" in positions: output["adverse_excursion_quantiles"] = quantiles(positions.mae, qs, True)
    return output or None


def compute_statistics(data: dict, quantile_levels=DEFAULT_QUANTILES, n_mc_sims=2000, mc_seed: Optional[int] = None, wf_split=0.7, trading_days_per_year=TRADING_DAYS_PER_YEAR, trade_time_field="open_time"):
    missing = [key for key in ("positions", "equity", "balance", "equity_time") if key not in data]
    if missing: raise ValueError(f"compute_statistics: missing required keys: {missing}")
    positions = data["positions"]
    if trade_time_field not in TRADE_TIME_FIELDS:
        raise ValueError("trade_time_field must be 'open_time' or 'close_time'")
    if not isinstance(positions, pd.DataFrame): raise TypeError("data['positions'] must be a pandas DataFrame")
    missing = [column for column in REQUIRED_POSITION_COLUMNS if column not in positions]
    if missing: raise ValueError(f"positions DataFrame missing required columns: {missing}")
    equity, balance, equity_time = map(lambda value: np.asarray(value, dtype=np.float64), (data["equity"], data["balance"], data["equity_time"]))
    if not len(equity) or not (len(equity) == len(balance) == len(equity_time)): raise ValueError("equity, balance, and equity_time must all be non-empty and the same length")
    order = np.argsort(equity_time, kind="mergesort"); equity, balance, equity_time = equity[order], balance[order], equity_time[order]
    positions = positions.sort_values(trade_time_field, kind="mergesort").reset_index(drop=True); idx = build_index_collections(positions, trade_time_field); drawdowns = compute_drawdown_stats(equity, balance, equity_time); started = _time.perf_counter()
    return {"meta": {"n_trades": len(positions), "n_equity_points": len(equity), "numba_available": NUMBA_AVAILABLE, "trade_time_field": trade_time_field, "compute_seconds": _time.perf_counter() - started}, "indexes": {"win_idx": idx["win_idx"], "loss_idx": idx["loss_idx"]}, "duration_stats": compute_duration_stats(positions, idx["win_idx"], idx["loss_idx"], quantile_levels), "distribution_stats": compute_distribution_stats(positions, equity_time.astype(np.int64), balance, quantile_levels, trade_time_field), "drawdown_stats": drawdowns, "trade_count_stats": compute_trade_count_stats(positions, idx), "trade_profit_stats": compute_trade_profit_stats(positions, idx, equity, balance, equity_time.astype(np.int64), drawdowns["balance_dd"]["max_dd_abs"], quantile_levels, trading_days_per_year), "position_sizing_stats": compute_position_sizing_stats(positions, idx, quantile_levels), "streak_stats": compute_streak_stats(positions, equity_time.astype(np.int64), balance, quantile_levels, trade_time_field), "monte_carlo_stats": {"bootstrap": compute_monte_carlo_stats(positions, balance, n_mc_sims, seed=mc_seed), "walk_forward": compute_walk_forward_stats(positions, wf_split)}, "seasonal_stats": compute_seasonal_stats(positions, idx), "trade_efficiency_stats": compute_trade_efficiency_stats(positions, quantile_levels)}


class TradeStatisticsEngine:
    def __init__(self, quantile_levels=DEFAULT_QUANTILES, n_mc_sims=2000, mc_seed=None, wf_split=0.7, trading_days_per_year=TRADING_DAYS_PER_YEAR, trade_time_field="open_time"):
        if trade_time_field not in TRADE_TIME_FIELDS:
            raise ValueError("trade_time_field must be 'open_time' or 'close_time'")
        self.quantile_levels, self.n_mc_sims, self.mc_seed, self.wf_split, self.trading_days_per_year, self.trade_time_field = quantile_levels, n_mc_sims, mc_seed, wf_split, trading_days_per_year, trade_time_field
    def compute(self, data):
        return compute_statistics(data, self.quantile_levels, self.n_mc_sims, self.mc_seed, self.wf_split, self.trading_days_per_year, self.trade_time_field)