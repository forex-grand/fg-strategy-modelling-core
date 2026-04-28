"""Transformer encoder classifier model."""

from __future__ import annotations

import tensorflow as tf
from keras import layers

from src.models_architecture.train_models.base_train_model import BaseSignalModel
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


class TransformerModel(BaseSignalModel):
    """Multi-head self-attention encoder with dense classification head."""

    def __init__(self, *, sequence_length: int, preprocessor: PreprocessBase) -> None:
        super().__init__(
            sequence_length=sequence_length,
            preprocessor=preprocessor,
            model_name="transformer_model",
        )
        self.input_proj = layers.Dense(96, kernel_regularizer=self.kernel_regularizer, name="input_proj")
        self.mha1 = layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.2, name="mha_1")
        self.mha2 = layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.2, name="mha_2")
        self.norm1 = layers.BatchNormalization(name="bn_1")
        self.norm2 = layers.BatchNormalization(name="bn_2")
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(128, activation="relu", kernel_regularizer=self.kernel_regularizer),
                layers.Dropout(0.25),
                layers.Dense(96, activation="relu", kernel_regularizer=self.kernel_regularizer),
            ],
            name="ffn",
        )
        self.dropout = layers.Dropout(0.3, name="dropout")
        self.pool = layers.GlobalAveragePooling1D(name="gap")
        self.head = layers.Dense(64, activation="relu", kernel_regularizer=self.kernel_regularizer, name="head")

    def encode(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.input_proj(features, training=training)
        attn1 = self.mha1(x, x, training=training)
        x = self.norm1(x + attn1, training=training)
        attn2 = self.mha2(x, x, training=training)
        x = self.norm2(x + attn2, training=training)
        x = self.ffn(x, training=training)
        x = self.dropout(x, training=training)
        x = self.pool(x)
        return self.head(x, training=training)

