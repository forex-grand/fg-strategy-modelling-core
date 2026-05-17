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
from imblearn.over_sampling import SMOTE
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

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
        cardinality = train_ds.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH","-1"))
        num_batches = steps_per_epoch if steps_per_epoch>0 else (cardinality if cardinality>0 else 100)
        if cardinality==-2:
            num_batches = -1

        y = None
        ts = datetime.now()
        first_batch_seen = False
        for batch_x, batch_y in train_ds.take(num_batches):
            yd = batch_y.numpy()
            
            if first_batch_seen:
                y  = np.concatenate([y, yd], axis=0)                
            else:
                y = yd
                first_batch_seen = True

        y_logits = np.argmax(y, axis=1)
        weights = compute_sample_weight(class_weight='balanced', y=y_logits)
        # If you already have train_ds as (x, y) tuples
        sample_weight_ds = tf.data.Dataset.from_tensor_slices(weights)
        train_ds_weighted = tf.data.Dataset.zip((train_ds, sample_weight_ds))
        # This gives ((x, y), weight) — need to flatten to (x, y, weight)
        train_ds_weighted = train_ds_weighted.map(lambda xy, w: (xy[0], xy[1], w))

        with self.strategy.scope():
            model = self.build_model(input_spec=train_ds.element_spec[0])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=fn_args.learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=self._get_metrics(),
            )

        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(os.getenv("EARLY_STOPPING_PATIENCE","10")),
                restore_best_weights=True,
            )
        ]
        
        self.history = model.fit(
            train_ds_weighted,
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