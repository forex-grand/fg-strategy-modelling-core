"""ForexGrand Strategy Modelling Core - ML framework for forex trading strategy development.

This package provides tools and utilities for:
- Loading and caching market data from multiple storage backends
- Computing technical indicators and features
- Training deep learning models on timeseries data
- Evaluating model performance and managing deployments

Core Modules:
    - data_manager: Load forex market data from S3-compatible storage
    - indicators: Technical indicators for feature engineering
    - settings: Configuration management via environment variables
    - config: Helpers for configuring Cloudflare R2-backed storage
    - pipeline: Data preparation, training, and evaluation workflows
    - storage: Storage abstraction for Cloudflare R2-compatible access

Quick Start:
    >>> from forexgrand_core import configure_r2
    >>> from forexgrand_core.data_manager import DataManager
    >>> from forexgrand_core.main import run_training
    >>>
    >>> configure_r2(
    ...     account_id="your-cloudflare-account-id",
    ...     access_key_id="your-r2-access-key-id",
    ...     secret_access_key="your-r2-secret-access-key",
    ...     bucket_name="forexgrand",
    ... )
    >>>
    >>> # Run training pipeline
    >>> results = run_training(
    ...     symbols=['EURUSD', 'GBPUSD'],
    ...     model_types=['simple', 'conservative'],
    ...     sequence_length=60,
    ... )

Environment Setup:
    See README.md for required environment variables and configuration.
"""

__version__ = "0.2.2"
__author__ = "ForexGrand Team"
__license__ = "MIT"

# Import lightweight core modules first
from forexgrand_core.config import R2Config, configure_r2
from forexgrand_core.settings import Settings

# Lazy-load modules with external dependencies
def __getattr__(name):
    """Lazy import of modules with external dependencies."""
    if name == "DataManager":
        from forexgrand_core.data_manager import DataManager
        return DataManager
    elif name == "run_training":
        from forexgrand_core.main import run_training
        return run_training
    elif name == "Trainer":
        from forexgrand_core.pipeline.trainer import Trainer
        return Trainer
    elif name == "GenerateTrainData":
        from forexgrand_core.generate_train_data import GenerateTrainData
        return GenerateTrainData
    elif name in {"SignalsBase", "BacktestResult", "ResultType", "RESULT_TYPE", "trade_result_only", "statistics_only", "trade_result_and_statistics", "trade_result", "statistics", "SLTPCalculator", "SignalExtractor", "MarketTableBuilder", "BacktestEngine", "StrategyLoadError", "run_backtest", "FIXED_LOT", "PERCENTAGE", "MARTINGALE", "ANTIMARTINGALE", "FIXED_SLTP", "RANGE", "ATR", "LOTSIZERS", "SLTP_MODES"}:
        from forexgrand_core.backtesting import (
            ANTIMARTINGALE, ATR, FIXED_LOT, FIXED_SLTP, LOTSIZERS, MARTINGALE,
            PERCENTAGE, RANGE, SLTP_MODES, BacktestEngine, BacktestResult,
            MarketTableBuilder, RESULT_TYPE, ResultType, SignalsBase, SLTPCalculator,
            SignalExtractor, StrategyLoadError, run_backtest, statistics,
            statistics_only, trade_result, trade_result_and_statistics,
            trade_result_only,
        )
        return locals()[name]
    elif name == "Evaluator":
        from forexgrand_core.pipeline.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "run_training",
    "DataManager",
    "Settings",
    "R2Config",
    "configure_r2",
    "Trainer",
    "GenerateTrainData",
    "Evaluator",
    "BacktestResult",
    "SignalsBase",
    "ResultType",
    "RESULT_TYPE",
    "trade_result_only",
    "statistics_only",
    "trade_result_and_statistics",
    "trade_result",
    "statistics",
    "SLTPCalculator",
    "SignalExtractor",
    "MarketTableBuilder",
    "BacktestEngine",
    "StrategyLoadError",
    "run_backtest",
    "FIXED_LOT",
    "PERCENTAGE",
    "MARTINGALE",
    "ANTIMARTINGALE",
    "FIXED_SLTP",
    "RANGE",
    "ATR",
    "LOTSIZERS",
    "SLTP_MODES",
]
