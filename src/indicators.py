"""Technical indicators for forex data analysis.

This module provides a collection of technical indicators commonly used in forex trading,
implemented using pandas and numpy for efficient computation on timeseries data.

Indicators include:
    - Moving Average (MA)
    - Slope (linear regression slope)
    - Average True Range (ATR)
    - Relative Strength Index (RSI)
    - Standard Deviation (Stdev)
    - Bollinger Bands
    - German-Klass Volatility
    - Wick-to-Bar Range Ratio
    - Feature Normalization
    
All indicator functions operate on DataFrames with OHLCV data and return Series or DataFrames.
Factory functions are provided to create indicator functions with fixed parameters.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def tf_ma(df: pd.DataFrame, period: int = 12, column: str = "close") -> pd.Series:
    """Simple Moving Average (SMA).
    
    Calculates the average price over a specified period.
    
    Args:
        df: DataFrame containing OHLCV data.
        period: Window size in periods (default: 12). Must be positive.
        column: Column name to calculate MA on (default: 'close'). 
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        pd.Series: Moving average values. First (period-1) values are NaN.
    
    Raises:
        KeyError: If specified column doesn't exist in DataFrame.
    
    Example:
        >>> ma = tf_ma(df, period=20, column='close')
    """
    
    values = df[column].astype(float)
    return values.rolling(window=period, min_periods=period).mean()

def tf_slope(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Linear regression slope indicator.
    
    Calculates the slope of a linear regression line fitted to price data.
    Positive slope indicates uptrend, negative indicates downtrend.
    
    Args:
        df: DataFrame containing OHLCV data.
        period: Lookback period in bars (default: 14). Must be positive.
        column: Column to analyze (default: 'close').
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        pd.Series: Slope values. First (period-1) values are NaN.
    
    Example:
        >>> slope = tf_slope(df, period=20, column='close')
    """
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
    """Average True Range (ATR) volatility indicator.
    
    Measures market volatility by calculating the average of true range values.
    True range = max(high-low, abs(high-prev_close), abs(low-prev_close)).
    
    Args:
        df: DataFrame containing OHLCV data. Must have 'high', 'low', 'close' columns.
        period: EMA period for smoothing (default: 14). Must be positive.
    
    Returns:
        pd.Series: ATR values. First (period-1) values are NaN.
    
    Raises:
        ValueError: If required columns ('high', 'low', 'close') are missing.
    
    Example:
        >>> atr = tf_atr(df, period=14)
    """
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
    """Relative Strength Index (RSI) momentum indicator.
    
    Measures the magnitude of recent price changes to evaluate overbought/oversold levels.
    RSI values range from 0-100, with >70 considered overbought and <30 oversold.
    
    Args:
        df: DataFrame containing OHLCV data.
        period: EMA period for smoothing gains/losses (default: 14). Must be positive.
        column: Column to analyze (default: 'close').
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        pd.Series: RSI values ranging 0-100. First (period) values may be NaN or estimated.
    
    Example:
        >>> rsi = tf_rsi(df, period=14, column='close')
        >>> overbought = rsi[rsi > 70]
    """
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
    """Standard deviation of price.
    
    Measures price volatility as the standard deviation over a period.
    Used for risk analysis and to calculate Bollinger Bands.
    
    Args:
        df: DataFrame containing OHLCV data.
        period: Rolling window size (default: 14). Must be positive.
        column: Column to analyze (default: 'close').
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        pd.Series: Standard deviation values. First (period-1) values are NaN.
    
    Example:
        >>> volatility = tf_stdev(df, period=20, column='close')
    """
    values = df[column].astype(float)
    return values.rolling(window=period, min_periods=period).std(ddof=0)


def tf_bollinger_bands(
    df: pd.DataFrame,
    period: int = 14,
    deviation: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """Bollinger Bands volatility indicator.
    
    Creates an envelope around price based on moving average and standard deviation.
    Consists of middle (SMA), upper (SMA + std*deviation), and lower (SMA - std*deviation) bands.
    
    Args:
        df: DataFrame containing OHLCV data.
        period: Period for moving average (default: 14). Must be positive.
        deviation: Number of standard deviations for band width (default: 2.0). Typically 1.5-3.0.
        column: Column to analyze (default: 'close').
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        pd.DataFrame: DataFrame with 'middle', 'upper', 'lower' columns for band values.
    
    Example:
        >>> bands = tf_bollinger_bands(df, period=20, deviation=2.0)
        >>> print(bands[['upper', 'middle', 'lower']].head())
    """
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
    """Garman-Klass volatility estimator.
    
    Volatility measure using open, high, low, and close prices.
    More accurate than standard deviation but requires more data points.
    
    Args:
        df: DataFrame containing OHLCV data. Must have 'open', 'high', 'low', 'close' columns.
        period: Rolling period for averaging (default: 14). Must be positive.
    
    Returns:
        pd.Series: Annualized volatility estimates. First (period-1) values are NaN.
    
    Raises:
        ValueError: If required OHLC columns are missing.
    
    Example:
        >>> gk_vol = tf_german_klass_volatility(df, period=14)
    """
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
    """Wick-to-body range ratio indicator.
    
    Measures the ratio of wicks to bar body size, indicating price indecision.
    Higher values indicate larger wicks relative to body (indecision bars).
    
    Args:
        df: DataFrame containing OHLCV data. Must have 'open', 'high', 'low', 'close' columns.
    
    Returns:
        pd.Series: Wick-to-range ratio values. NaN where range is zero.
    
    Raises:
        ValueError: If required OHLC columns are missing.
    
    Example:
        >>> wick_ratio = tf_wick_bar_range_ratio(df)
    """
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
    """Normalize feature to 0-1 range using quantile clipping.
    
    Normalizes a column by clipping extreme values (outliers) and scaling to [0, 1].
    Robust to outliers unlike min-max normalization.
    
    Args:
        df: DataFrame containing data.
        column: Column name to normalize.
        lower_quantile: Lower quantile for clipping (default: 0.01 = 1st percentile).
            Valid range: [0, 1).
        upper_quantile: Upper quantile for clipping (default: 0.99 = 99th percentile).
            Valid range: (lower_quantile, 1].
    
    Returns:
        pd.Series: Normalized values in range [0, 1]. Returns zeros if span is 0.
    
    Raises:
        KeyError: If specified column doesn't exist.
    
    Example:
        >>> normalized_close = tf_normalize_feature(df, 'close', lower_quantile=0.01, upper_quantile=0.99)
    """
    values = df[column].astype(float)
    lower = float(values.quantile(lower_quantile))
    upper = float(values.quantile(upper_quantile))
    clipped = values.clip(lower=lower, upper=upper)
    span = upper - lower
    if span == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=df.index)
    return (clipped - lower) / span


def ma_factory(period: int = 12, column: str = "close") -> Callable:
    """Factory function to create a Moving Average indicator with fixed parameters.
    
    Returns a callable that takes a DataFrame and computes MA with the specified parameters.
    Useful for creating indicator configs with fixed parameters.
    
    Args:
        period: Window size (default: 12). Must be positive.
        column: Column name (default: 'close').
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        Callable: Function that takes DataFrame and returns MA Series.
    
    Example:
        >>> ma_20 = ma_factory(period=20, column='close')
        >>> ma_values = ma_20(df)
    """
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_ma(df, period=period, column=column)

    return indicator_func


def slope_factory(period: int = 14, column: str = "close") -> Callable:
    """Factory function to create a Slope indicator with fixed parameters.
    
    Args:
        period: Lookback period (default: 14). Must be positive.
        column: Column name (default: 'close').
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        Callable: Function that takes DataFrame and returns Slope Series.
    
    Example:
        >>> slope_20 = slope_factory(period=20, column='close')
        >>> slope_values = slope_20(df)
    """
    def indicator_func(df: pd.DataFrame, **kwargs) -> pd.Series:
        return tf_slope(df, period=period, column=column)

    return indicator_func


def atr_factory(period: int = 14) -> Callable:
    """Factory function to create an ATR indicator with fixed period.
    
    Args:
        period: ATR period (default: 14). Must be positive.
    
    Returns:
        Callable: Function that takes DataFrame and returns ATR Series.
    
    Example:
        >>> atr_14 = atr_factory(period=14)
        >>> atr_values = atr_14(df)
    """
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
