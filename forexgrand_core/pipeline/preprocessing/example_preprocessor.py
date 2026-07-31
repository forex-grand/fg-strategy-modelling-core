"""Reference preprocessing implementation for training and inference."""

from __future__ import annotations

import tensorflow as tf

from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase


class ExamplePreprocessor(PreprocessBase):
    """Simple feature normalization preprocessor."""

    def __init__(self, sequence_length: int) -> None:
        super().__init__(sequence_length=sequence_length, name="example_preprocessor")
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def preprocess(self, inputs, training: bool = False) -> dict[str, tf.Tensor]:
        if isinstance(inputs, dict):
            raw_features = tf.cast(inputs["features"], tf.float32)
            target = inputs.get("target")
        else:
            raw_features = tf.cast(inputs, tf.float32)
            target = None

        normalized = self.norm(raw_features, training=training)
        result: dict[str, tf.Tensor] = {"features": normalized}
        if training and target is not None:
            result["target"] = tf.cast(target, tf.float32)
        return result

