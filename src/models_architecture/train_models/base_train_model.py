from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Any

import keras
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.models_architecture.base_model import BaseModel
from src.schemas import ModelBuildTrainArguments
from abc import abstractmethod

class TrainModel(BaseModel):
    def __init__(
        self,
        preprocessor: keras.Layer,
        sequence_length: int,
    ) -> None:
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.history: keras.callbacks.History | None = None
        self.strategy = tf.distribute.MirroredStrategy()

    @abstractmethod
    def build_model(self, input_spec: dict) -> tf.keras.Model:
        """Should be implemented by individual subclasses"""

    def _get_metrics(self) -> list[keras.metrics.Metric]:
        return [
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.Precision(name="precision_buy", class_id=0),
            keras.metrics.Precision(name="precision_sell", class_id=1),
            keras.metrics.Precision(name="precision_hold", class_id=2),
            keras.metrics.Recall(name="recall_buy", class_id=0),
            keras.metrics.Recall(name="recall_sell", class_id=1),
            keras.metrics.Recall(name="recall_hold", class_id=2),
        ]

    def build_train_model(self, train_ds: tf.data.Dataset, eval_ds: tf.data.Dataset, fn_args: ModelBuildTrainArguments) -> keras.Model:
        train_ds = self.strategy.experimental_distribute_dataset(train_ds)
        eval_ds = self.strategy.experimental_distribute_dataset(eval_ds)

        with self.strategy.scope():
            model = self.build_model(input_spec=train_ds.element_spec[0])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=fn_args.learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=self._get_metrics(),
            )

        # confirm model is on GPU
        if model.weights:
            print(f"[INFO] Model device: {model.weights[0].device}")

        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]

        self.history = model.fit(
            train_ds,
            validation_data=eval_ds,
            epochs=fn_args.epochs,
            callbacks=callbacks,
            steps_per_epoch=fn_args.steps_per_epoch,
            verbose=1,
        )

        # Build inference model inside strategy scope
        with self.strategy.scope():
            inputs = {inp.name: inp for inp in self._build_input_signature()}
            preprocess = self.preprocessor(inputs)
            inference = model(preprocess)
            output = keras.layers.Lambda(lambda x: tf.argmax(x, axis=-1))(inference)
            self.model = keras.Model(inputs, output)
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=fn_args.learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=self._get_metrics(),
            )

        return model

    def _get_inference_model(self) -> keras.Model:
        if self.model is None:
            raise ValueError("Training model must be built and trained before exporting inference.")

        raw_inputs = self._build_input_signature(self.sequence_length)
        features = self.preprocessor(raw_inputs)
        predictions = self.model(features)
        return keras.Model(inputs=raw_inputs, outputs=predictions, name="train_inference")