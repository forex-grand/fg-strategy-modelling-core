"""Model architectures for forex prediction models.

This package contains base classes and implementations for various neural network
architectures used for forex trading strategy development.

Modules:
    - base_model: Abstract base class for all models
    - no_train_model: Pre-trained models that don't require training
    - train_models: Trainable model implementations
        - base_train_model: Base class for trainable models
        - lstm_model: LSTM-based architecture
        - simple_model: Simple CNN architecture
        - conservative_model: Conservative ensemble model
        - complex_model: Complex multi-tower architecture
        - cnn_bi_lstm: Bidirectional LSTM with CNN
"""

from src.models_architecture.base_model import BaseModel
from src.models_architecture.no_train_model import NoTrainModel

__all__ = [
    "BaseModel",
    "NoTrainModel",
]
