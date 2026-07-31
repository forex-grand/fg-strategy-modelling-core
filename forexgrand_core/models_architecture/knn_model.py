from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from forexgrand_core.models_architecture.base_model import BaseModel
from forexgrand_core.models_architecture.kmeans_module import Kmeans_model
from forexgrand_core.schemas import ModelBuildTrainArguments


@keras.saving.register_keras_serializable(package="ForexGrand")
class FeatureStackLayer(keras.layers.Layer):
    def __init__(self, feature_keys: list[str], **kwargs):
        super().__init__(**kwargs)
        self.feature_keys = list(feature_keys)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_keys": self.feature_keys})
        return config

    @staticmethod
    def _flatten_feature_tensor(value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, tf.float32)
        if value.shape.rank == 1:
            value = tf.expand_dims(value, axis=-1)
        batch_size = tf.shape(value)[0]
        return tf.reshape(value, (batch_size, -1))

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        columns = [self._flatten_feature_tensor(features[key]) for key in self.feature_keys]
        return tf.concat(columns, axis=-1)


class KNNModel(BaseModel):
    """Feature-only KNN-style model backed by the existing KMeans module."""

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.feature_keys: list[str] = []
        self.feature_shapes: dict[str, tuple[int, ...]] = {}
        self.knn_model: Kmeans_model | None = None
        self.feature_stack_layer: FeatureStackLayer | None = None

    @staticmethod
    def _feature_dict_from_batch(batch: Any) -> dict[str, tf.Tensor]:
        if isinstance(batch, tuple):
            return batch[0]
        return batch

    @staticmethod
    def _flatten_feature_tensor(value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, tf.float32)
        if value.shape.rank == 1:
            value = tf.expand_dims(value, axis=-1)
        batch_size = tf.shape(value)[0]
        return tf.reshape(value, (batch_size, -1))

    def _stack_features(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        if not self.feature_keys:
            self.feature_keys = list(features.keys())
        return FeatureStackLayer(self.feature_keys)(features)

    def _collect_dataset_features(self, dataset: tf.data.Dataset, num_batches: int) -> tf.Tensor:
        batches: list[tf.Tensor] = []
        for batch in dataset.take(num_batches):
            features = self._feature_dict_from_batch(batch)
            if not isinstance(features, dict):
                raise ValueError("KNNModel expects each dataset element to be a feature dictionary.")
            for key, value in features.items():
                self.feature_shapes[key] = tuple(value.shape[1:])
            batches.append(self._stack_features(features))

        if not batches:
            raise ValueError("No feature batches were available for KNN training.")
        return tf.concat(batches, axis=0)

    @staticmethod
    def _resolve_positive_int_env(name: str, default: int) -> int:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return value

    def build_train_model(
        self,
        train_ds: tf.data.Dataset,
        eval_ds: tf.data.Dataset,
        fn_args: ModelBuildTrainArguments | dict,
    ) -> keras.Model:
        cardinality = train_ds.cardinality()
        configured_steps = getattr(fn_args, "steps_per_epoch", None)
        if isinstance(fn_args, dict):
            configured_steps = fn_args.get("steps_per_epoch")
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", str(configured_steps or -1)))
        num_batches = steps_per_epoch if steps_per_epoch > 0 else (cardinality if cardinality > 0 else 100)
        if cardinality == -2:
            num_batches = -1

        ts = datetime.now()
        train_features = self._collect_dataset_features(train_ds, num_batches)
        print("KNN training data collected: ", (datetime.now().timestamp() - ts.timestamp()), "s")
        print("KNN training data shape: ", tf.shape(train_features))

        max_iters = self._resolve_positive_int_env("KNN_MAX_ITERS", 100)
        neighbors = self._resolve_positive_int_env("KNN_NEIGHBORS", min(500, int(train_features.shape[0] or 500)))
        epochs = int(getattr(fn_args, "epochs", 1) if not isinstance(fn_args, dict) else fn_args.get("epochs", 1))

        if neighbors > int(tf.shape(train_features)[0].numpy()):
            raise ValueError("KNN_NEIGHBORS cannot be greater than the collected training sample count.")

        self.knn_model = Kmeans_model(k=neighbors, max_iters=max_iters)
        self.knn_model.compile(training_strategy=os.getenv("KNN_TRAINING_STRATEGY", "randomizer"))
        self.knn_model.fit(train_features, epochs=epochs)

        self.feature_stack_layer = FeatureStackLayer(self.feature_keys, name="feature_stack")
        inputs = {inp.name: inp for inp in self._build_input_signature()}
        preprocessed = self.preprocessor(inputs)
        stacked = self.feature_stack_layer(preprocessed)
        outputs = self.knn_model(stacked)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="knn_inference")
        return self.knn_model

    def evaluate(self, eval_ds: tf.data.Dataset) -> dict[str, float]:
        if self.knn_model is None:
            raise ValueError("KNN model must be trained before evaluation.")

        cardinality = eval_ds.cardinality()
        steps = cardinality if cardinality > 0 else int(os.getenv("KNN_EVAL_STEPS", "100"))
        if cardinality == -2:
            steps = -1
        eval_features = self._collect_dataset_features(eval_ds, steps)
        clusters = self.knn_model.kmeans_layer.assign_clusters(eval_features)
        loss = self.knn_model.kmeans_layer.compute_loss(eval_features, clusters).numpy()
        return {
            "accuracy": 0.0,
            "precision_buy": 0.0,
            "precision_sell": 0.0,
            "recall_buy": 0.0,
            "recall_sell": 0.0,
            "val_loss": float(loss),
            "train_loss": 0.0,
        }

    def get_serving_signature(self):
        input_signature = {
            "time": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.int64, name="time"),
            "open": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="open"),
            "high": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="high"),
            "close": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="close"),
            "low": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="low"),
            "spread": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="spread"),
            "real_volume": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="real_volume"),
            "tick_volume": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="tick_volume"),
        }

        @tf.function(input_signature=[input_signature])
        def serve(examples):
            return {"output":self.model(examples)}

        return serve
