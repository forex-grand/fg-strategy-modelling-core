"""Complex architecture with BiLSTM, attention, and residuals."""

from __future__ import annotations

import tensorflow as tf
from keras import layers

from src.models_architecture.train_models.base_train_model import TrainModel
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


class ComplexModel(TrainModel):
    """BiLSTM + self-attention + residual block."""

    def __init__(self, *, sequence_length: int, preprocessor: PreprocessBase) -> None:
        super().__init__(
            sequence_length=sequence_length,
            preprocessor=preprocessor,
            model_name="complex_model",
        )
        self.bilstm = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, kernel_regularizer=self.kernel_regularizer),
            name="bilstm",
        )
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.2,
            name="mha",
        )
        self.res_proj = layers.Dense(128, kernel_regularizer=self.kernel_regularizer, name="res_proj")
        self.norm1 = layers.BatchNormalization(name="bn_1")
        self.ffn = layers.Dense(128, activation="relu", kernel_regularizer=self.kernel_regularizer, name="ffn")
        self.dropout = layers.Dropout(0.3, name="dropout")
        self.pool = layers.GlobalAveragePooling1D(name="gap")
        self.head = layers.Dense(64, activation="relu", kernel_regularizer=self.kernel_regularizer, name="head")

    def encode(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.bilstm(features, training=training)
        attended = self.attention(x, x, training=training)
        residual = self.res_proj(x, training=training)
        x = self.norm1(attended + residual, training=training)
        x = self.ffn(x, training=training)
        x = self.dropout(x, training=training)
        x = self.pool(x)
        return self.head(x, training=training)

