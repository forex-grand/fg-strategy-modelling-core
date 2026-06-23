from __future__ import annotations

import os
from abc import abstractmethod
from datetime import datetime
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight

from src.models_architecture.base_model import BaseModel
from src.pipeline.feature_selection import auto_expand_feature_fe
from src.schemas import ModelBuildTrainArguments

tf.get_logger().setLevel("ERROR")


class TrainModel(BaseModel):
    def __init__(
        self,
        preprocessor: keras.Layer,
        sequence_length: int,
    ) -> None:
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.history: keras.callbacks.History | None = None
        self.strategy = tf.distribute.MirroredStrategy()
        self.nn_model: keras.Model | None = None
        self.openfe_nn_model: keras.Model | None = None

    @abstractmethod
    def build_model(self, input_spec: dict, num_classes: int) -> tf.keras.Model:
        """Should be implemented by individual subclasses"""

    def _get_metrics(self, num_classes: int | None = None) -> list[keras.metrics.Metric]:
        metrics: list[keras.metrics.Metric] = [
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ]

        if num_classes is None or num_classes >= 1:
            metrics.append(keras.metrics.Precision(name="precision_buy", class_id=0))
            metrics.append(keras.metrics.Recall(name="recall_buy", class_id=0))

        if num_classes is None or num_classes >= 2:
            metrics.append(keras.metrics.Precision(name="precision_sell", class_id=1))
            metrics.append(keras.metrics.Recall(name="recall_sell", class_id=1))

        if num_classes is None or num_classes >= 3:
            metrics.append(keras.metrics.Precision(name="precision_hold", class_id=2))
            metrics.append(keras.metrics.Recall(name="recall_hold", class_id=2))

        return metrics

    @staticmethod
    def _feature_generator() -> str:
        return str(os.getenv("FEATURE_GENERATOR", "")).strip().upper()

    @staticmethod
    def _sampling_strategy() -> str:
        return (
            os.getenv("TRAIN_DATA_SAMPLING")
            or os.getenv("SAMPLING_STRATEGY")
            or os.getenv("DATA_SAMPLING_STRATEGY")
            or "weights"
        ).strip().lower().replace("-", "_")

    @staticmethod
    def _labels_to_class_ids(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[-1] > 1:
            return np.argmax(y, axis=-1).astype(np.int64)
        return np.reshape(y, (-1,)).astype(np.int64)

    @staticmethod
    def _class_ids_to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        return keras.utils.to_categorical(
            y.astype(np.int64),
            num_classes=num_classes,
        ).astype("float32")

    @staticmethod
    def _collect_dataset_as_frame(dataset: tf.data.Dataset, num_batches: int):
        frames: list[pd.DataFrame] = []
        labels: list[np.ndarray] = []
        feature_shapes: dict[str, tuple[int, ...]] = {}
        feature_dtypes: dict[str, tf.dtypes.DType] = {}

        for batch_x, batch_y in dataset.take(num_batches):
            batch_columns: dict[str, np.ndarray] = {}
            for key, value in batch_x.items():
                arr = value.numpy()
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))

                row_count = arr.shape[0]
                trailing_shape = tuple(arr.shape[1:])
                feature_shapes[key] = trailing_shape
                feature_dtypes[key] = value.dtype
                flat = arr.reshape((row_count, -1))

                if flat.shape[1] == 1:
                    batch_columns[key] = flat[:, 0]
                else:
                    for idx in range(flat.shape[1]):
                        batch_columns[f"{key}__{idx}"] = flat[:, idx]

            frames.append(pd.DataFrame(batch_columns))
            labels.append(batch_y.numpy())

        if not frames:
            raise ValueError("No batches were available to build training data.")

        return (
            pd.concat(frames, axis=0, ignore_index=True),
            np.concatenate(labels, axis=0),
            feature_shapes,
            feature_dtypes,
        )

    @staticmethod
    def _frame_to_feature_dict(
        frame: pd.DataFrame,
        feature_shapes: dict[str, tuple[int, ...]] | None = None,
        feature_dtypes: dict[str, tf.dtypes.DType] | None = None,
    ) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        if not feature_shapes:
            for column in frame.columns:
                features[column] = frame[column].to_numpy(dtype=np.float32).reshape((-1, 1))
            return features

        for key, shape in feature_shapes.items():
            size = int(np.prod(shape)) if shape else 1
            if size == 1 and key in frame.columns:
                arr = frame[key].to_numpy()
            else:
                columns = [f"{key}__{idx}" for idx in range(size)]
                arr = frame[columns].to_numpy()

            dtype = feature_dtypes.get(key, tf.float32) if feature_dtypes else tf.float32
            np_dtype = dtype.as_numpy_dtype if hasattr(dtype, "as_numpy_dtype") else np.float32
            features[key] = arr.reshape((-1, *shape)).astype(np_dtype)

        return features

    @staticmethod
    def _frame_input_spec(frame: pd.DataFrame) -> dict[str, tf.TensorSpec]:
        return {
            column: tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name=column)
            for column in frame.columns
        }

    def _apply_sampling(self, X: pd.DataFrame, y_class: np.ndarray):
        strategy = self._sampling_strategy()
        sample_weight = None

        if strategy in {"none", "off", "false", "0"}:
            return X, y_class, sample_weight

        if strategy in {"weights", "class_weight", "class_weights", "weighted"}:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_class)
            return X, y_class, sample_weight

        if strategy in {"oversample", "over", "upsample", "up"}:
            sampler = RandomOverSampler(random_state=44)
            X_res, y_res = sampler.fit_resample(X, y_class)
        elif strategy in {"undersample", "under", "downsample", "down"}:
            sampler = RandomUnderSampler(random_state=44)
            X_res, y_res = sampler.fit_resample(X, y_class)
        elif strategy in {"over_under", "upsample_undersample", "both"}:
            over = RandomOverSampler(random_state=44)
            under = RandomUnderSampler(random_state=44)
            X_res, y_res = over.fit_resample(X, y_class)
            X_res, y_res = under.fit_resample(X_res, y_res)
        elif strategy in {"smote", "smote_over"}:
            sampler = SMOTE(random_state=44)
            X_res, y_res = sampler.fit_resample(X, y_class)
        elif strategy in {"smote_under", "smote_undersample"}:
            over = SMOTE(random_state=44)
            under = RandomUnderSampler(random_state=44)
            X_res, y_res = over.fit_resample(X, y_class)
            X_res, y_res = under.fit_resample(X_res, y_res)
        else:
            raise ValueError(
                "Unsupported TRAIN_DATA_SAMPLING value. Use one of: "
                "none, weights, oversample, undersample, over_under, smote, smote_under."
            )

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_res)
        return X_res, y_res.astype(np.int64), sample_weight

    def _make_training_dataset(
        self,
        X: pd.DataFrame,
        y_class: np.ndarray,
        num_classes: int,
        batch_size: int,
        sample_weight: np.ndarray | None = None,
        feature_shapes: dict[str, tuple[int, ...]] | None = None,
        feature_dtypes: dict[str, tf.dtypes.DType] | None = None,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        y = self._class_ids_to_categorical(y_class, num_classes=num_classes)
        feature_dict = self._frame_to_feature_dict(X, feature_shapes, feature_dtypes)
        tensors: tuple[Any, ...]
        if sample_weight is None:
            tensors = (feature_dict, y)
        else:
            tensors = (feature_dict, y, sample_weight.astype("float32"))

        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        if shuffle:
            dataset = dataset.shuffle(
                int(os.getenv("SHUFFLE_BUFFER_SIZE", "10000")),
                reshuffle_each_iteration=True,
            )
        return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    def build_train_model(
        self,
        train_ds: tf.data.Dataset,
        eval_ds: tf.data.Dataset,
        fn_args: ModelBuildTrainArguments,
    ) -> keras.Model:
        cardinality = train_ds.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "-1"))
        num_batches = steps_per_epoch if steps_per_epoch > 0 else (cardinality if cardinality > 0 else 100)
        if cardinality == -2:
            num_batches = -1

        ts = datetime.now()
        X_train, y_train_raw, feature_shapes, feature_dtypes = self._collect_dataset_as_frame(
            train_ds,
            num_batches,
        )
        X_eval, y_eval_raw, _, _ = self._collect_dataset_as_frame(eval_ds, num_batches)
        print("Training data collected: ", (datetime.now().timestamp() - ts.timestamp()), "s")

        y_train_class = self._labels_to_class_ids(y_train_raw)
        y_eval_class = self._labels_to_class_ids(y_eval_raw)

        unique_classes = np.union1d(np.unique(y_train_class), np.unique(y_eval_class))
        if unique_classes.size == 0:
            raise ValueError("No target classes found in training or evaluation data.")

        if not np.array_equal(unique_classes, np.arange(unique_classes.max() + 1)):
            class_id_map = {label: idx for idx, label in enumerate(unique_classes)}
            y_train_class = np.vectorize(class_id_map.get)(y_train_class)
            y_eval_class = np.vectorize(class_id_map.get)(y_eval_class)

        num_classes = int(unique_classes.size)
        use_openfe = self._feature_generator() == "OPENFE"
        if use_openfe:
            X_train, X_eval, self.feature_transformer = auto_expand_feature_fe(
                X_train,
                y_train_class,
                X_eval,
                metadata=fn_args.model_dump() if hasattr(fn_args, "model_dump") else {},
            )
            feature_shapes = None
            feature_dtypes = None

        X_train, y_train_class, sample_weight = self._apply_sampling(X_train, y_train_class)

        batch_size = int(os.getenv("BATCH_SIZE", "128"))
        train_model_ds = self._make_training_dataset(
            X_train,
            y_train_class,
            num_classes=num_classes,
            batch_size=batch_size,
            sample_weight=sample_weight,
            feature_shapes=feature_shapes,
            feature_dtypes=feature_dtypes,
            shuffle=True,
        )
        eval_model_ds = self._make_training_dataset(
            X_eval,
            y_eval_class,
            num_classes=num_classes,
            batch_size=batch_size,
            feature_shapes=feature_shapes,
            feature_dtypes=feature_dtypes,
            shuffle=False,
        )
        input_spec = self._frame_input_spec(X_train) if use_openfe else train_ds.element_spec[0]

        with self.strategy.scope():
            model = self.build_model(input_spec=input_spec, num_classes=num_classes)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=fn_args.learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=self._get_metrics(num_classes=num_classes),
            )

        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(os.getenv("EARLY_STOPPING_PATIENCE", "10")),
                restore_best_weights=True,
            )
        ]

        fit_kwargs: dict[str, Any] = {
            "x": train_model_ds,
            "validation_data": eval_model_ds,
            "epochs": fn_args.epochs,
            "callbacks": callbacks,
            "verbose": 1,
        }
        if fn_args.steps_per_epoch and fn_args.steps_per_epoch > 0:
            fit_kwargs["steps_per_epoch"] = fn_args.steps_per_epoch

        self.history = model.fit(**fit_kwargs)

        with self.strategy.scope():
            self.nn_model = model
            inputs = {inp.name: inp for inp in self._build_input_signature()}
            preprocess = self.preprocessor(inputs)
            if use_openfe:
                self.openfe_nn_model = model
                self.model = keras.Model(inputs, preprocess)
            else:
                inference = model(preprocess)
                output = keras.layers.Lambda(
                    lambda x: tf.argmax(x, axis=-1),
                    name="class_id",
                )(inference)
                self.model = keras.Model(inputs, output)
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=fn_args.learning_rate),
                    loss=keras.losses.CategoricalCrossentropy(),
                    metrics=self._get_metrics(num_classes=num_classes),
                )

        return model

    def _get_inference_model(self) -> keras.Model:
        if self.model is None:
            raise ValueError("Training model must be built and trained before exporting inference.")

        raw_inputs = self._build_input_signature(self.sequence_length)
        features = self.preprocessor(raw_inputs)
        predictions = self.model(features)
        return keras.Model(inputs=raw_inputs, outputs=predictions, name="train_inference")
