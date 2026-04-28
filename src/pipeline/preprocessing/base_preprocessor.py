"""Base preprocessing layer embedded in the model graph."""

from __future__ import annotations

import abc
import tempfile
from pathlib import Path
from typing import Any
import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(name="Preprocessing_layer")
class PreprocessBase(keras.layers.Layer, metaclass=abc.ABCMeta):
    """
    Abstract preprocessing layer.

    Implementations must return:
    - Training: {"features": tensor, "target": tensor}
    - Inference: {"features": tensor}
    """

    def __init__(self, sequence_length: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        # self._aux_model_cache: dict[str, Any] = {}

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
        })
        return config
    
    def call(self, inputs, training: bool = False):
        """Run preprocessing inside the Keras graph."""
        return self.preprocess(inputs, training=training)

    @keras.utils.register_keras_serializable(name="Preproces_function")
    def get_transform_layer(self,):
        return self

    @abc.abstractmethod
    @keras.utils.register_keras_serializable(name="Preproces_function")
    def preprocess(self, inputs, training: bool = False) -> dict[str, tf.Tensor]:
        """Transform raw model inputs into downstream tensors."""

    # def load_and_run_auxiliary_model(self, model_gcs_path: str, inputs: tf.Tensor) -> tf.Tensor:
    #     """
    #     Download an auxiliary model from GCS, run inference, and return output tensor.
    #     """
    #     if model_gcs_path not in self._aux_model_cache:
    #         local_model_dir = self._download_model_from_gcs(model_gcs_path)
    #         self._aux_model_cache[model_gcs_path] = tf.saved_model.load(local_model_dir)

    #     loaded_model = self._aux_model_cache[model_gcs_path]
    #     infer = loaded_model.signatures.get("serving_default")
    #     if infer is None:
    #         raise ValueError(f"Auxiliary model '{model_gcs_path}' has no serving_default signature.")

    #     try:
    #         outputs = infer(features=inputs)
    #     except TypeError:
    #         outputs = infer(inputs)

    #     if isinstance(outputs, dict):
    #         return next(iter(outputs.values()))
    #     return outputs

    # def _download_model_from_gcs(self, model_gcs_path: str) -> str:
    #     """Recursively copy a GCS model directory to a local temp directory."""
    #     source = model_gcs_path.rstrip("/")
    #     if not tf.io.gfile.exists(source):
    #         raise FileNotFoundError(f"Auxiliary model path does not exist: {source}")

    #     temp_root = Path(tempfile.mkdtemp(prefix="fg_aux_model_"))
    #     target = temp_root / "model"
    #     self._copy_tree(source, str(target))
    #     return str(target)

    def _copy_tree(self, source: str, target: str) -> None:
        tf.io.gfile.makedirs(target)
        for item in tf.io.gfile.listdir(source):
            src_item = f"{source}/{item}"
            dst_item = str(Path(target) / item)
            if tf.io.gfile.isdir(src_item):
                self._copy_tree(src_item, dst_item)
            else:
                tf.io.gfile.copy(src_item, dst_item, overwrite=True)

