"""Training pipeline modules for data processing and model training.

This package contains modules for:
- Data generation and feature engineering (generate_train_data)
- Model training and optimization (trainer)
- Model evaluation and benchmarking (evaluator)
- Data preprocessing and transformation (preprocessing)
- Performance testing and validation (performance_test)
- Model deployment and uploads (pusher)
"""

from src.pipeline.trainer import Trainer, TrainingResult
from src.pipeline.evaluator import Evaluator
from src.pipeline.generate_train_data import GenerateTrainData

__all__ = [
    "Trainer",
    "TrainingResult",
    "Evaluator",
    "GenerateTrainData",
]
