"""Entry point for ForexGrand training orchestration."""

from __future__ import annotations

import logging
from typing import Type

from src.settings import Settings
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase
from src.pipeline.preprocessing.example_preprocessor import ExamplePreprocessor
from src.pipeline.trainer import Trainer, TrainingResult


def run_training(
    *,
    symbols: list[str],
    model_types: list[str],
    preprocessor_class: Type[PreprocessBase] = ExamplePreprocessor,
    sequence_length: int | None = None,
) -> list[TrainingResult]:
    """Train all requested symbol/model combinations."""
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

