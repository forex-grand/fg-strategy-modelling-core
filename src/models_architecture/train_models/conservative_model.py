"""Conservative low-complexity model."""

from __future__ import annotations

import tensorflow as tf
from keras import layers

from src.models_architecture.train_models.base_train_model import BaseSignalModel
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


class ConservativeModel(BaseSignalModel):
    """Shallow LSTM with regularized dense head."""

    def __init__(self, *, sequence_length: int, preprocessor: PreprocessBase) -> None:
        super().__init__(
            sequence_length=sequence_length,
            preprocessor=preprocessor,
            model_name="conservative_model",
        )
        self.lstm = layers.LSTM(
            32,
            kernel_regularizer=self.kernel_regularizer,
            return_sequences=False,
            name="lstm",
        )
        self.bn = layers.BatchNormalization(name="bn")
        self.dropout = layers.Dropout(0.25, name="dropout")
        self.dense = layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=self.kernel_regularizer,
            name="dense",
        )

    def encode(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.lstm(features, training=training)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense(x)

