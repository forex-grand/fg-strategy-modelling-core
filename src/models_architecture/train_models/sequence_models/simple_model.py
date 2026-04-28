"""Simple moderate-depth recurrent architecture."""

from __future__ import annotations

import tensorflow as tf
from keras import layers

from src.models_architecture.train_models.base_train_model import BaseSignalModel
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


class SimpleModel(BaseSignalModel):
    """Stacked GRU model with batch normalization and dropout."""

    def __init__(self, *, sequence_length: int, preprocessor: PreprocessBase) -> None:
        super().__init__(
            sequence_length=sequence_length,
            preprocessor=preprocessor,
            model_name="simple_model",
        )
        self.gru1 = layers.GRU(
            64,
            return_sequences=True,
            kernel_regularizer=self.kernel_regularizer,
            name="gru_1",
        )
        self.bn1 = layers.BatchNormalization(name="bn_1")
        self.gru2 = layers.GRU(
            48,
            return_sequences=False,
            kernel_regularizer=self.kernel_regularizer,
            name="gru_2",
        )
        self.bn2 = layers.BatchNormalization(name="bn_2")
        self.dropout = layers.Dropout(0.3, name="dropout")
        self.head = layers.Dense(
            48,
            activation="relu",
            kernel_regularizer=self.kernel_regularizer,
            name="head_dense",
        )

    def encode(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.gru1(features, training=training)
        x = self.bn1(x, training=training)
        x = self.gru2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        return self.head(x)

