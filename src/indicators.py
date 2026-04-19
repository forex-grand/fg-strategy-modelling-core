from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def tf_ma(df: pd.DataFrame, period: int = 12, column: str = "close") -> pd.Series:
    values = df[column].astype(float)
    return values.rolling(window=period, min_periods=period).mean()

def tf_slope(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    values = df[column].astype(float)
    x = np.arange(period, dtype=float)
    x_mean = float(x.mean())
    denominator = float(np.sum((x - x_mean) ** 2))

    def slope_fn(window: np.ndarray) -> float:
        y_mean = float(window.mean())
        numerator = float(np.sum((x - x_mean) * (window - y_mean)))
        return numerator / denominator if denominator else np.nan

    return values.rolling(window=period, min_periods=period).apply(slope_fn, raw=True)

def tf_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def tf_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    values = df[column].astype(float)
    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gains = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_losses = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gains / avg_losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(np.where(avg_gains > 0.0, 100.0, 50.0))
    rsi.iloc[:period] = np.nan
    return rsi


def tf_stdev(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    values = df[column].astype(float)
    return values.rolling(window=period, min_periods=period).std(ddof=0)


def tf_bollinger_bands(
    df: pd.DataFrame,
    period: int = 14,
    deviation: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    middle = tf_ma(df, period=period, column=column)
    stdev = tf_stdev(df, period=period, column=column)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + (stdev * deviation),
            "lower": middle - (stdev * deviation),
        },
        index=df.index,
    )


def tf_german_klass_volatility(df: pd.DataFrame, period: int = 14) -> pd.Series:
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    open_ = df["open"].astype(float).replace(0.0, np.nan)
    high = df["high"].astype(float).replace(0.0, np.nan)
    low = df["low"].astype(float).replace(0.0, np.nan)
    close = df["close"].astype(float).replace(0.0, np.nan)

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * (log_hl**2) - ((2 * np.log(2.0)) - 1) * (log_co**2)
    return np.sqrt(gk.rolling(window=period, min_periods=period).mean()) * np.sqrt(period)


def tf_wick_bar_range_ratio(df: pd.DataFrame) -> pd.Series:
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    body_range = (df["open"].astype(float) - df["close"].astype(float)).abs()
    wick_range = (df["high"].astype(float) - df["low"].astype(float)).replace(0.0, np.nan)
    return body_range / wick_range


def tf_normalize_feature(
    df: pd.DataFrame,
    column: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    values = df[column].astype(float)
    lower = float(values.quantile(lower_quantile))
    upper = float(values.quantile(upper_quantile))
    clipped = values.clip(lower=lower, upper=upper)
    span = upper - lower
    if span == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=df.index)
    return (clipped - lower) / span


def ma_factory(period: int = 12, column: str = "close") -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_ma(df, period=period, column=column)

    return indicator_func


def slope_factory(period: int = 14, column: str = "close") -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_slope(df, period=period, column=column)

    return indicator_func


def atr_factory(period: int = 14) -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_atr(df, period=period)

    return indicator_func


def rsi_factory(period: int = 14, column: str = "close") -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_rsi(df, period=period, column=column)

    return indicator_func


def stdev_factory(period: int = 14, column: str = "close") -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_stdev(df, period=period, column=column)

    return indicator_func


def bollinger_bands_factory(period: int = 14, deviation: float = 2.0, column: str = "close") -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return tf_bollinger_bands(df, period=period, deviation=deviation, column=column)

    return indicator_func


def gk_volatility_factory(period: int = 14) -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_german_klass_volatility(df, period=period)

    return indicator_func


def wick_ratio_factory() -> Callable:
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_wick_bar_range_ratio(df)

    return indicator_func


class TensorFlowIndicators:
    @staticmethod
    def MA(period: int = 12, column: str = "close") -> Callable:
        return ma_factory(period, column)

    @staticmethod
    def Slope(period: int = 14, column: str = "close") -> Callable:
        return slope_factory(period, column)

    @staticmethod
    def ATR(period: int = 14) -> Callable:
        return atr_factory(period)

    @staticmethod
    def RSI(period: int = 14, column: str = "close") -> Callable:
        return rsi_factory(period, column)

    @staticmethod
    def StdDev(period: int = 14, column: str = "close") -> Callable:
        return stdev_factory(period, column)

    @staticmethod
    def BollingerBands(period: int = 14, deviation: float = 2.0, column: str = "close") -> Callable:
        return bollinger_bands_factory(period, deviation, column)

    @staticmethod
    def GKVolatility(period: int = 14) -> Callable:
        return gk_volatility_factory(period)

    @staticmethod
    def WickRatio() -> Callable:
        return wick_ratio_factory()

    @staticmethod
    def get_all_indicators() -> Dict[str, Callable]:
        return {
            "MA": TensorFlowIndicators.MA,
            "Slope": TensorFlowIndicators.Slope,
            "ATR": TensorFlowIndicators.ATR,
            "RSI": TensorFlowIndicators.RSI,
            "StdDev": TensorFlowIndicators.StdDev,
            "BollingerBands": TensorFlowIndicators.BollingerBands,
            "GKVolatility": TensorFlowIndicators.GKVolatility,
            "WickRatio": TensorFlowIndicators.WickRatio,
        }
