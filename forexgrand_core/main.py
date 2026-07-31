"""Entry point for ForexGrand training orchestration.

This module provides the main entry point for training models on forex data.
It coordinates data loading, preprocessing, and model training workflows.
"""

from __future__ import annotations

import logging
from typing import Type

from forexgrand_core.settings import Settings
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.pipeline.preprocessing.example_preprocessor import ExamplePreprocessor
from forexgrand_core.pipeline.trainer import Trainer, TrainingResult


def run_training(
    *,
    symbols: list[str],
    model_types: list[str],
    preprocessor_class: Type[PreprocessBase] = ExamplePreprocessor,
    sequence_length: int | None = None,
) -> list[TrainingResult]:
    """Train ML models on forex data for specified symbols.
    
    Orchestrates the complete training pipeline for multiple symbols and model architectures.
    Loads configuration from environment variables, prepares data, trains models, and
    evaluates performance against configured benchmarks.
    
    Args:
        symbols: List of currency pair symbols to train on (e.g., ['EURUSD', 'GBPUSD']).
            Symbols are case-insensitive and will be converted to uppercase.
        model_types: List of model architectures to train (e.g., ['conservative', 'simple', 'complex']).
            Options: 'conservative', 'simple', 'complex', 'lstm', 'cnn_bi_lstm'.
        preprocessor_class: Data preprocessor class to use (default: ExamplePreprocessor).
            Must be a subclass of PreprocessBase. Custom preprocessors can implement
            custom feature engineering and data transformations.
        sequence_length: Sequence length for model input (default: None uses config value).
            If provided, overrides SEQUENCE_STRIDE environment variable for all models.
    
    Returns:
        list[TrainingResult]: Training results for each symbol/model combination, including
            model paths, performance metrics, and evaluation scores.
    
    Raises:
        ValueError: If symbols or model_types are empty, or if configuration is invalid.
        OSError: If data files cannot be accessed or models cannot be saved.
    
    Environment Variables Used:
        DATA_SOURCE: Data source type (default: 'mt5')
        S3_STORAGE_OPTION: Storage backend (default: 'minio')
        BATCH_SIZE: Training batch size (default: 64)
        EPOCHS: Number of training epochs (default: 50)
        LEARNING_RATE: Model learning rate (default: 1e-3)
        Refer to forexgrand_core.settings for complete list of variables.
    
    Example:
        >>> results = run_training(
        ...     symbols=['EURUSD', 'GBPUSD'],
        ...     model_types=['conservative', 'simple'],
        ...     preprocessor_class=ExamplePreprocessor,
        ...     sequence_length=60,
        ... )
        >>> for result in results:
        ...     print(f"{result.symbol}: {result.model_type} accuracy={result.accuracy}")
    """
    config = Settings()
    if sequence_length is not None:
        for model_type in model_types:
            config.sequence_lengths[model_type.lower()] = int(sequence_length)

    all_results: list[TrainingResult] = []
    for symbol in symbols:
        trainer = Trainer(
            symbol=symbol,
            model_types=model_types,
            preprocessor_class=preprocessor_class,
            config=config,
        )
        all_results.extend(trainer.run())
    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    run_training(
        symbols=["EURUSD", "GBPUSD"],
        model_types=["conservative", "simple", "complex", "transformer"],
        preprocessor_class=ExamplePreprocessor,
        sequence_length=60,
    )

