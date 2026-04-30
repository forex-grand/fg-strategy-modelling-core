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
    - pipeline: Data preparation, training, and evaluation workflows
    - storage: Multi-backend storage abstraction (AWS S3, MinIO, GCS, Cloudflare R2)

Quick Start:
    >>> from src.data_manager import DataManager
    >>> from src.main import run_training
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

__version__ = "0.1.0"
__author__ = "ForexGrand Team"
__license__ = "MIT"

# Validate environment on import
from src.env_validator import validate_environment_on_import
validate_environment_on_import()

# Import lightweight core modules first
from src.settings import Settings
from src.indicators import (
    tf_ma,
    tf_slope,
    tf_atr,
    tf_rsi,
    tf_stdev,
    tf_bollinger_bands,
    tf_normalize_feature,
    tf_german_klass_volatility,
    tf_wick_bar_range_ratio,
    ma_factory,
    slope_factory,
    atr_factory,
)

# Lazy-load modules with external dependencies
def __getattr__(name):
    """Lazy import of modules with external dependencies."""
    if name == "DataManager":
        from src.data_manager import DataManager
        return DataManager
    elif name == "run_training":
        from src.main import run_training
        return run_training
    elif name == "Trainer":
        from src.pipeline.trainer import Trainer
        return Trainer
    elif name == "GenerateTrainData":
        from src.pipeline.generate_train_data import GenerateTrainData
        return GenerateTrainData
    elif name == "Evaluator":
        from src.pipeline.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "run_training",
    "DataManager",
    "Settings",
    "Trainer",
    "GenerateTrainData",
    "Evaluator",
    "tf_ma",
    "tf_slope",
    "tf_atr",
    "tf_rsi",
    "tf_stdev",
    "tf_bollinger_bands",
    "tf_normalize_feature",
    "tf_german_klass_volatility",
    "tf_wick_bar_range_ratio",
    "ma_factory",
    "slope_factory",
    "atr_factory",
]
