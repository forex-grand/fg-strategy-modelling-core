from __future__ import annotations

import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
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
from src.pipeline.feature_selection import auto_expand_feature_fe, transform_fe
from src.schemas import ModelBuildTrainArguments

try:
    import keras_tuner as kt
except ImportError:
    kt = None

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
        self.num_classes: int = 0
        self.initial_output_bias: np.ndarray | None = None

    @abstractmethod
    def build_model(
        self,
        input_spec: dict,
        num_classes: int,
        hp: kt.HyperParameters | None = None,
    ) -> tf.keras.Model:
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

    def evaluate(self, eval_ds: tf.data.Dataset) -> dict[str, float]:
        if self.num_classes <= 0:
            raise ValueError("Model must be trained before evaluation; num_classes is not set.")

        precision_buy = keras.metrics.Precision(class_id=0)
        precision_sell = keras.metrics.Precision(class_id=1)
        recall_buy = keras.metrics.Recall(class_id=0)
        recall_sell = keras.metrics.Recall(class_id=1)
        accuracy = keras.metrics.CategoricalAccuracy()

        cardinality = eval_ds.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "-1"))
        num_batches = steps_per_epoch if steps_per_epoch > 0 else (cardinality if cardinality > 0 else 100)
        num_eval_batches = cardinality if cardinality > 0 else num_batches
        if cardinality == -2:
            num_eval_batches = -1

        use_openfe = self._feature_generator() == "OPENFE"
        predict_model = self.openfe_nn_model if use_openfe and self.openfe_nn_model is not None else self.nn_model
        if predict_model is None:
            raise ValueError("No trained neural network model is available for evaluation.")

        for batch_x, batch_y in eval_ds.take(num_eval_batches):
            if use_openfe:
                X = np.stack([tf.squeeze(value).numpy() for value in batch_x.values()], axis=-1)
                X = pd.DataFrame(X, columns=list(batch_x.keys()))
                X = transform_fe(X, self.feature_transformer)
                X_input = {
                    name: X[name].to_numpy(dtype=np.float32).reshape((-1, 1))
                    for name in X.columns
                }
                predictions = predict_model.predict(X_input, verbose=0)
            else:
                predictions = predict_model.predict(batch_x, verbose=0)

            y_true = self._labels_to_class_ids(batch_y)
            y_pred = np.asarray(predictions)
            if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
                y_pred = np.argmax(y_pred, axis=-1)
            else:
                y_pred = np.reshape(y_pred, (-1,))

            y_true_onehot = tf.one_hot(y_true, depth=self.num_classes)
            y_pred_onehot = tf.one_hot(y_pred.astype(np.int32), depth=self.num_classes)

            precision_buy.update_state(y_true_onehot, y_pred_onehot)
            precision_sell.update_state(y_true_onehot, y_pred_onehot)
            recall_buy.update_state(y_true_onehot, y_pred_onehot)
            recall_sell.update_state(y_true_onehot, y_pred_onehot)
            accuracy.update_state(y_true_onehot, y_pred_onehot)

        val_loss = float(self.history.history["val_loss"][-1]) if self.history and "val_loss" in self.history.history else 0.0
        train_loss = float(self.history.history["loss"][-1]) if self.history and "loss" in self.history.history else 0.0

        return {
            "accuracy": float(accuracy.result().numpy()),
            "precision_buy": float(precision_buy.result().numpy()),
            "precision_sell": float(precision_sell.result().numpy()),
            "recall_buy": float(recall_buy.result().numpy()),
            "recall_sell": float(recall_sell.result().numpy()),
            "val_loss": val_loss,
            "train_loss": train_loss,
        }

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
    def _resolve_excluded_class_id() -> int | None:
        exclude_id = os.getenv("EXCLUDE_CLASS_ID", "-1").strip()
        if exclude_id == "":
            return None
        return int(exclude_id)

    @staticmethod
    def _resolve_num_batches(dataset: tf.data.Dataset, fn_args: ModelBuildTrainArguments) -> int:
        configured_steps = getattr(fn_args, "steps_per_epoch", None)
        if configured_steps and configured_steps > 0:
            return int(configured_steps)

        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "-1"))
        cardinality = dataset.cardinality()
        num_batches = steps_per_epoch if steps_per_epoch > 0 else (cardinality if cardinality > 0 else 100)
        if cardinality == -2:
            num_batches = -1
        return int(num_batches)

    @staticmethod
    def _class_ids_to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        return keras.utils.to_categorical(
            y.astype(np.int64),
            num_classes=num_classes,
        ).astype("float32")

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _use_initial_output_bias(self, fn_args: ModelBuildTrainArguments) -> bool:
        if "INITIALIZE_OUTPUT_BIAS" in os.environ:
            return self._env_flag("INITIALIZE_OUTPUT_BIAS")
        if "USE_INITIAL_OUTPUT_BIAS" in os.environ:
            return self._env_flag("USE_INITIAL_OUTPUT_BIAS")
        return bool(getattr(fn_args, "initialize_output_bias", False))

    @staticmethod
    def _compute_initial_output_bias(y_class: np.ndarray, num_classes: int) -> np.ndarray:
        counts = np.bincount(y_class.astype(np.int64), minlength=num_classes).astype("float32")
        total = float(np.sum(counts))
        if total <= 0:
            raise ValueError("Cannot initialize output bias because the training labels are empty.")

        probabilities = counts / total
        probabilities = np.clip(probabilities, np.finfo("float32").tiny, 1.0)
        probabilities = probabilities / np.sum(probabilities)
        return np.log(probabilities).astype("float32")

    def _output_bias_initializer(self) -> keras.initializers.Initializer | str:
        if self.initial_output_bias is None:
            return "zeros"
        return keras.initializers.Constant(self.initial_output_bias)

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

    @staticmethod
    def _keras_tuner_enabled() -> bool:
        return os.getenv("KERAS_TUNER_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _require_keras_tuner():
        if kt is None:
            raise ImportError(
                "keras-tuner is required for neural-network hyperparameter search. "
                "Install project dependencies or set KERAS_TUNER_ENABLED=false."
            )
        return kt

    @staticmethod
    def _resolve_tuner_trials() -> int:
        trials = int(os.getenv("KERAS_TUNER_TRIALS", os.getenv("NN_KERAS_TUNER_TRIALS", "20")))
        if trials <= 0:
            raise ValueError("KERAS_TUNER_TRIALS must be greater than zero.")
        return trials

    @staticmethod
    def _resolve_tuner_executions_per_trial() -> int:
        executions = int(os.getenv("KERAS_TUNER_EXECUTIONS_PER_TRIAL", "1"))
        if executions <= 0:
            raise ValueError("KERAS_TUNER_EXECUTIONS_PER_TRIAL must be greater than zero.")
        return executions

    @staticmethod
    def _resolve_tuner_seed() -> int:
        return int(os.getenv("KERAS_TUNER_RANDOM_STATE", "44"))

    @staticmethod
    def _resolve_tuner_objective() -> kt.Objective:
        tuner_module = TrainModel._require_keras_tuner()
        objective_name = os.getenv("KERAS_TUNER_OBJECTIVE", "val_loss").strip()
        direction = os.getenv("KERAS_TUNER_OBJECTIVE_DIRECTION", "min").strip().lower()
        if direction not in {"min", "max"}:
            raise ValueError("KERAS_TUNER_OBJECTIVE_DIRECTION must be either 'min' or 'max'.")
        return tuner_module.Objective(objective_name, direction=direction)

    def _build_loss(self, hp: kt.HyperParameters | None = None) -> keras.losses.Loss:
        if hp is None:
            return keras.losses.CategoricalCrossentropy()

        loss_name = "categorical_focal_crossentropy"
        if loss_name == "categorical_focal_crossentropy":
            return keras.losses.CategoricalFocalCrossentropy(
                alpha=hp.Float("focal_alpha", 0.1, 0.75, step=0.05, default=0.25),
                gamma=hp.Float("focal_gamma", 1.0, 5.0, step=0.5, default=2.0),
            )
        return keras.losses.CategoricalCrossentropy()

    def _compile_model(
        self,
        model: keras.Model,
        num_classes: int,
        hp: kt.HyperParameters | None = None,
        learning_rate: float = 1e-3,
    ) -> keras.Model:
        if hp is not None:
            learning_rate = hp.Float(
                "learning_rate",
                min_value=1e-5,
                max_value=3e-3,
                sampling="log",
                default=learning_rate,
            )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._build_loss(hp),
            metrics=self._get_metrics(num_classes=num_classes),
        )
        return model

    def _build_compiled_hypermodel(
        self,
        hp: kt.HyperParameters,
        input_spec: dict,
        num_classes: int,
        learning_rate: float,
    ) -> keras.Model:
        model = self.build_model(input_spec=input_spec, num_classes=num_classes, hp=hp)
        return self._compile_model(
            model,
            num_classes=num_classes,
            hp=hp,
            learning_rate=learning_rate,
        )

    def _find_best_hyperparameters(
        self,
        train_model_ds: tf.data.Dataset,
        eval_model_ds: tf.data.Dataset,
        input_spec: dict,
        num_classes: int,
        fn_args: ModelBuildTrainArguments,
        callbacks: list[keras.callbacks.Callback],
    ) -> kt.HyperParameters | None:
        if not self._keras_tuner_enabled():
            return None

        tuner_module = self._require_keras_tuner()
        tuner_dir = Path(os.getenv("KERAS_TUNER_DIRECTORY", ".keras_tuner"))
        project_name = os.getenv("KERAS_TUNER_PROJECT_NAME", self.__class__.__name__.lower())
        tuner = tuner_module.RandomSearch(
            hypermodel=lambda hp: self._build_compiled_hypermodel(
                hp,
                input_spec=input_spec,
                num_classes=num_classes,
                learning_rate=fn_args.learning_rate,
            ),
            objective=self._resolve_tuner_objective(),
            max_trials=self._resolve_tuner_trials(),
            distribution_strategy=self.strategy,
            executions_per_trial=self._resolve_tuner_executions_per_trial(),
            overwrite=os.getenv("KERAS_TUNER_OVERWRITE", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            directory=str(tuner_dir),
            project_name=project_name,
            seed=self._resolve_tuner_seed(),
        )

        tuner_epochs = int(os.getenv("KERAS_TUNER_EPOCHS", str(min(int(fn_args.epochs), 20))))
        search_kwargs: dict[str, Any] = {
            "x": train_model_ds,
            "validation_data": eval_model_ds,
            "epochs": tuner_epochs,
            "callbacks": callbacks,
            "verbose": int(os.getenv("KERAS_TUNER_VERBOSE", "1")),
        }
        if fn_args.steps_per_epoch and fn_args.steps_per_epoch > 0:
            search_kwargs["steps_per_epoch"] = fn_args.steps_per_epoch

        tuner.search(**search_kwargs)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hps:
            return None

        print("Best Keras Tuner hyperparameters: ", best_hps[0].values)
        return best_hps[0]

    def build_train_model(
        self,
        train_ds: tf.data.Dataset,
        eval_ds: tf.data.Dataset,
        fn_args: ModelBuildTrainArguments,
    ) -> keras.Model:
        num_batches = self._resolve_num_batches(train_ds, fn_args)

        ts = datetime.now()
        X_train, y_train_raw, feature_shapes, feature_dtypes = self._collect_dataset_as_frame(
            train_ds,
            num_batches,
        )
        X_eval, y_eval_raw, _, _ = self._collect_dataset_as_frame(eval_ds, num_batches)
        print("Training data collected: ", (datetime.now().timestamp() - ts.timestamp()), "s")

        y_train_class = self._labels_to_class_ids(y_train_raw)
        y_eval_class = self._labels_to_class_ids(y_eval_raw)
        exclude_id = self._resolve_excluded_class_id()
        if exclude_id is not None:
            train_mask = y_train_class != exclude_id
            eval_mask = y_eval_class != exclude_id
            X_train = X_train.loc[train_mask].reset_index(drop=True)
            y_train_class = y_train_class[train_mask]
            X_eval = X_eval.loc[eval_mask].reset_index(drop=True)
            y_eval_class = y_eval_class[eval_mask]

        unique_classes = np.union1d(np.unique(y_train_class), np.unique(y_eval_class))
        if unique_classes.size == 0:
            raise ValueError("No target classes found in training or evaluation data.")

        if not np.array_equal(unique_classes, np.arange(unique_classes.max() + 1)):
            class_id_map = {label: idx for idx, label in enumerate(unique_classes)}
            y_train_class = np.vectorize(class_id_map.get)(y_train_class)
            y_eval_class = np.vectorize(class_id_map.get)(y_eval_class)

        num_classes = int(unique_classes.size)
        if self._use_initial_output_bias(fn_args):
            self.initial_output_bias = self._compute_initial_output_bias(
                y_train_class,
                num_classes=num_classes,
            )
            train_counts = np.bincount(y_train_class.astype(np.int64), minlength=num_classes)
            print(
                "Initial output bias enabled. Training class counts: ",
                train_counts.tolist(),
            )
        else:
            self.initial_output_bias = None

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

        self.num_classes = num_classes
        with self.strategy.scope():
            warmup_model = self.build_model(input_spec=input_spec, num_classes=num_classes)

        def make_callbacks() -> list[keras.callbacks.Callback]:
            user_callbacks = [
                callback
                for callback in (fn_args.callbacks or [])
                if isinstance(callback, keras.callbacks.Callback)
            ]
            return user_callbacks + [
                keras.callbacks.EarlyStopping(
                    monitor=str(os.getenv("KERAS_TUNER_OBJECTIVE", "val_loss").strip()),
                    patience=int(os.getenv("EARLY_STOPPING_PATIENCE", "10")),
                    restore_best_weights=True,
                    mode=os.getenv("KERAS_TUNER_OBJECTIVE_DIRECTION", "min").strip().lower(),
                )
            ]

        best_hp = self._find_best_hyperparameters(
            train_model_ds=train_model_ds,
            eval_model_ds=eval_model_ds,
            input_spec=input_spec,
            num_classes=num_classes,
            fn_args=fn_args,
            callbacks=make_callbacks(),
        )

        with self.strategy.scope():
            if best_hp is None:
                model = self._compile_model(
                    warmup_model,
                    num_classes=num_classes,
                    learning_rate=fn_args.learning_rate,
                )
            else:
                model = self._build_compiled_hypermodel(
                    best_hp,
                    input_spec=input_spec,
                    num_classes=num_classes,
                    learning_rate=fn_args.learning_rate,
                )

        fit_kwargs: dict[str, Any] = {
            "x": train_model_ds,
            "validation_data": eval_model_ds,
            "epochs": fn_args.epochs,
            "callbacks": make_callbacks(),
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
