"""Training pipeline modules for data processing and model training.

This package contains modules for:
- Data generation and feature engineering (generate_train_data)
- Model training and optimization (trainer)
- Model evaluation and benchmarking (evaluator)
- Data preprocessing and transformation (preprocessing)
- Performance testing and validation (performance_test)
- Model deployment and uploads (pusher)
"""

__all__ = [
    "Trainer",
    "TrainingResult",
    "Evaluator",
    "GenerateTrainData",
]


def __getattr__(name):
    if name == "Trainer":
        from forexgrand_core.pipeline.trainer import Trainer

        return Trainer
    if name == "TrainingResult":
        from forexgrand_core.schemas import TrainingResult

        return TrainingResult
    if name == "Evaluator":
        from forexgrand_core.pipeline.evaluator import Evaluator

        return Evaluator
    if name == "GenerateTrainData":
        from forexgrand_core.pipeline.generate_train_data import GenerateTrainData

        return GenerateTrainData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
