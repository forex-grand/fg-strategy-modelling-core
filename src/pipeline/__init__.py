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
        from src.pipeline.trainer import Trainer

        return Trainer
    if name == "TrainingResult":
        from src.schemas import TrainingResult

        return TrainingResult
    if name == "Evaluator":
        from src.pipeline.evaluator import Evaluator

        return Evaluator
    if name == "GenerateTrainData":
        from src.pipeline.generate_train_data import GenerateTrainData

        return GenerateTrainData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
