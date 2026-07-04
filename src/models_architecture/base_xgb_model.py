import os
import warnings
from typing import Optional

from src.models_architecture.base_model import BaseModel
import tensorflow as tf
import keras
from xgboost import XGBClassifier
import numpy as np
from datetime import datetime
import optuna
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.pipeline.feature_selection import auto_expand_feature_fe, transform_fe
import pandas as pd

# NOTE: objective / num_class / eval_metric are now resolved dynamically based on
# num_classes (see build_xgb_model). Do not hardcode them here.
_COMMON = dict(
    use_label_encoder=False,
    random_state=44,
    tree_method="hist",
    device=os.getenv("XGB_DEVICE", "cpu").strip().lower(),
    verbosity=0,
    n_jobs=-1,
)

cpu_counts = os.cpu_count()

class XGBTrainModel(BaseModel):
    """
      The model on the base model object is the preprocessing model object that outputs an array of numpy.
    """

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length
        self.train_loss = 0.0
        self.eval_loss  = 0.0
        self.num_classes = 0
        self.class_id_map = None
        self.class_id_reverse_map = None

    @staticmethod
    def _sampling_strategy() -> str:
        return (
            os.getenv("TRAIN_DATA_SAMPLING")
            or os.getenv("SAMPLING_STRATEGY")
            or os.getenv("DATA_SAMPLING_STRATEGY")
            or "smote_under"
        ).strip().lower().replace("-", "_")

    def _apply_sampling(self, X, y):
        strategy = self._sampling_strategy()
        sample_weight = None

        if strategy in {"none", "off", "false", "0"}:
            return X, y, sample_weight

        if strategy in {"weights", "class_weight", "class_weights", "weighted"}:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
            return X, y, sample_weight

        if strategy in {"oversample", "over", "upsample", "up"}:
            sampler = RandomOverSampler(random_state=44)
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy in {"undersample", "under", "downsample", "down"}:
            sampler = RandomUnderSampler(random_state=44)
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy in {"over_under", "upsample_undersample", "both"}:
            over = RandomOverSampler(random_state=44)
            under = RandomUnderSampler(random_state=44)
            X_res, y_res = over.fit_resample(X, y)
            X_res, y_res = under.fit_resample(X_res, y_res)
        elif strategy in {"smote", "smote_over"}:
            sampler = SMOTE(random_state=44)
            X_res, y_res = sampler.fit_resample(X, y)
        elif strategy in {"smote_under", "smote_undersample"}:
            over = SMOTE(random_state=44)
            under = RandomUnderSampler(random_state=44)
            X_res, y_res = over.fit_resample(X, y)
            X_res, y_res = under.fit_resample(X_res, y_res)
        else:
            raise ValueError(
                "Unsupported TRAIN_DATA_SAMPLING value. Use one of: "
                "none, weights, oversample, undersample, over_under, smote, smote_under."
            )

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_res)
        return X_res, y_res, sample_weight

    def build_model(self, input_spec:dict[str,tf.TensorSpec]):
        inputs = {key:keras.Input(shape=(spec.shape[-1] if spec.shape.rank>1 else 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        y = keras.layers.concatenate(list(inputs.values()))
        return keras.Model(inputs, y)

    @staticmethod
    def _to_class_ids(y):
        y_np = np.asarray(y)
        if y_np.ndim > 1 and y_np.shape[-1] > 1:
            return np.argmax(y_np, axis=-1).astype(np.int32)
        return np.reshape(y_np, (-1,)).astype(np.int32)

    @staticmethod
    def _eval_metric_key(num_classes: int) -> str:
        """Metric key used to read validation loss out of evals_result()."""
        return "logloss" if num_classes <= 2 else "mlogloss"

    def build_xgb_model(self, params: Optional[dict] = None, num_classes: int = 2):
        """
        Builds an XGBClassifier with objective/num_class/eval_metric chosen based on
        num_classes. binary:logistic must NOT receive num_class - passing it causes
        XGBoost to emit num_classes predictions per sample, which blows up shape
        checks against the (single-column) labels array.
        """
        params = dict(params or {})
        common = dict(_COMMON)

        if num_classes <= 2:
            common["objective"] = "binary:logistic"
            common["eval_metric"] = "logloss"
            common.pop("num_class", None)
        else:
            common["objective"] = "multi:softprob"
            common["eval_metric"] = "mlogloss"
            common["num_class"] = num_classes

        merged_params = {**common, **params}
        # Guard against a caller accidentally injecting num_class via params/best_params
        # when we're in binary mode (e.g. stale Optuna best_params from a prior multiclass run).
        if num_classes <= 2:
            merged_params.pop("num_class", None)

        return XGBClassifier(**merged_params)

    def _resolve_optuna_trials(self) -> int:
        trials = int(os.getenv("XGB_OPTUNA_TRIALS", "30"))
        if trials <= 0:
            raise ValueError("XGB_OPTUNA_TRIALS must be greater than zero.")
        return trials

    @staticmethod
    def _optuna_enabled() -> bool:
        return os.getenv("XGB_OPTUNA_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _resolve_num_batches(dataset: tf.data.Dataset, fn_args) -> int:
        configured_steps = None
        if isinstance(fn_args, dict):
            configured_steps = fn_args.get("steps_per_epoch")
        else:
            configured_steps = getattr(fn_args, "steps_per_epoch", None)
        if configured_steps and configured_steps > 0:
            return int(configured_steps)

        cardinality = dataset.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "-1"))
        num_batches = steps_per_epoch if steps_per_epoch > 0 else (cardinality if cardinality > 0 else 100)
        if cardinality == -2:
            num_batches = -1
        return int(num_batches)

    @staticmethod
    def _resolve_validation_fraction() -> float:
        fraction = float(os.getenv("XGB_OPTUNA_VALIDATION_FRACTION", "0.2"))
        if not 0.0 < fraction < 1.0:
            raise ValueError("XGB_OPTUNA_VALIDATION_FRACTION must be between 0 and 1.")
        return fraction

    @staticmethod
    def _suggest_params(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        }

    def _fit_xgb_model(self, model, X_train, y_train, eval_set):
        X_res, y_res, weights = self._apply_sampling(X_train, y_train)
        model.fit(
            X_res,
            y_res,
            eval_set=eval_set,
            sample_weight=weights,
            verbose=int(os.getenv("XGB_VERBOSE", "0")),
        )
        return model

    def _find_best_params(self, X, y, num_classes: int) -> dict:
        validation_fraction = self._resolve_validation_fraction()
        split_index = int(len(X) * (1.0 - validation_fraction))
        if split_index <= 0 or split_index >= len(X):
            raise ValueError("Not enough XGBoost training samples for Optuna train/validation split.")

        X_train, X_valid = X[:split_index], X[split_index:]
        y_train, y_valid = y[:split_index], y[split_index:]
        metric_key = self._eval_metric_key(num_classes)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            model = self.build_xgb_model(params, num_classes=num_classes)
            self._fit_xgb_model(model, X_train, y_train, eval_set=[(X_valid, y_valid)])
            results = model.evals_result()
            return float(results["validation_0"][metric_key][-1])

        sampler = optuna.samplers.TPESampler(
            seed=int(os.getenv("XGB_OPTUNA_RANDOM_STATE", "44"))
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        timeout_value = os.getenv("XGB_OPTUNA_TIMEOUT")
        timeout = int(timeout_value) if timeout_value else None
        study.optimize(
            objective,
            n_trials=self._resolve_optuna_trials(),
            timeout=timeout,
            show_progress_bar=False,
        )
        print("Best XGBoost Optuna params: ", study.best_params)
        print("Best XGBoost Optuna validation loss: ", study.best_value)
        return dict(study.best_params)

    def evaluate(self, eval_ds):
        if self.num_classes <= 0:
            raise ValueError("Model must be trained before evaluation; num_classes is not set.")

        precision_buy = keras.metrics.Precision(class_id=0)
        precision_sell= keras.metrics.Precision(class_id=1)
        recall_buy = keras.metrics.Recall(class_id=0)
        recall_sell= keras.metrics.Recall(class_id=1)
        accuracy   = keras.metrics.CategoricalAccuracy()

        cardinality = eval_ds.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH","-1"))
        num_batches = steps_per_epoch if steps_per_epoch>0 else (cardinality if cardinality>0 else 100)
        
        num_eval_batches = cardinality if cardinality>0 else num_batches
        if cardinality==-2:
            num_eval_batches = -1

        for batch_x, batch_y in eval_ds.take(num_eval_batches):
            X = np.stack([tf.squeeze(value) for value in batch_x.values()], axis=-1)

            if str(os.getenv('FEATURE_GENERATOR')).upper() == "OPENFE":
                if not self.feature_transformer:
                    raise ValueError("Feature Transformer does not exist for transformer openfe")
                X = pd.DataFrame(X, columns=eval_ds.element_spec[0].keys())
                X = transform_fe(X, self.feature_transformer)

            y_true = self._to_class_ids(batch_y)
            y_pred = np.asarray(self.xgb_model.predict(X))
            if y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=-1)
            else:
                y_pred = (y_pred > 0.5).astype(np.int32)  # For binary classification, threshold at 0.5

            y_pred = y_pred.astype(np.int32)

            y_true = tf.one_hot(y_true, depth=self.num_classes)
            y_pred = tf.one_hot(y_pred, depth=self.num_classes)

            precision_buy.update_state(y_true, y_pred)
            precision_sell.update_state(y_true, y_pred)
            recall_buy.update_state(y_true, y_pred)
            recall_sell.update_state(y_true, y_pred)
            accuracy.update_state(y_true, y_pred)

        metrics = {
            'accuracy':accuracy.result().numpy(),
            'precision_buy':precision_buy.result().numpy(),
            'precision_sell':precision_sell.result().numpy(),
            'recall_buy':recall_buy.result().numpy(),
            'recall_sell':recall_sell.result().numpy(),
            'val_loss':self.eval_loss,
            'train_loss':self.train_loss,
        }
        return metrics

    def build_train_model(self, train_ds, eval_ds, fn_args):
        inputs = self._build_input_signature()
        inputs_dict = {inp.name: inp for inp in inputs}
        x = self.preprocessor(inputs_dict)
        y = self.build_model(train_ds.element_spec[0])(x)
        model_ = keras.Model(inputs, y)
        self.model = model_
        
        ###training loop
        num_batches = self._resolve_num_batches(train_ds, fn_args)

        X = None
        y = None
        ts = datetime.now()
        first_batch_seen = False
        exclude_id = os.getenv("EXCLUDE_CLASS_ID", "-1")
        if exclude_id:
            exclude_id = int(exclude_id)
        for batch_x, batch_y in train_ds.take(num_batches):
            Xd = np.stack([tf.squeeze(value) for value in batch_x.values()], axis=-1)
            yd = self._to_class_ids(batch_y)
            exclude_mask = (yd!=exclude_id)
            Xd = Xd[exclude_mask]
            yd = yd[exclude_mask]
            if first_batch_seen:
                X  = np.concatenate([X, Xd], axis=0)
                y  = np.concatenate([y, yd], axis=0)
            else:
                X = np.array(Xd)
                y = np.array(yd)
                first_batch_seen = True

        print("Training data collected: ",(datetime.now().timestamp() - ts.timestamp()),"s")
        Xe = []
        ye = []
        first_batch_seen = False
        for batch_x, batch_y in eval_ds.take(num_batches):          
            Xd = np.stack([tf.squeeze(value) for value in batch_x.values()], axis=-1)
            yd = self._to_class_ids(batch_y)
            exclude_mask = (yd!=exclude_id)
            Xd = Xd[exclude_mask]
            yd = yd[exclude_mask]
            if first_batch_seen:
                Xe = np.concatenate([Xe, Xd], axis=0)
                ye = np.concatenate([ye, yd], axis=0)
            else:
                Xe = np.array(Xd)
                ye = np.array(yd)
                first_batch_seen = True

        unique_train_classes = np.unique(y)
        unique_eval_classes = np.unique(ye)
        unique_all_classes = np.union1d(unique_train_classes, unique_eval_classes)

        if unique_all_classes.size == 0:
            raise ValueError("No target classes found in training or evaluation data.")

        if not np.array_equal(unique_all_classes, np.arange(unique_all_classes.max() + 1)):
            warnings.warn(
                "Target labels are not contiguous zero-based integers. "
                "Non-zero or missing class IDs will be remapped to contiguous class IDs for XGBoost training."
            )
            class_id_map = {label: idx for idx, label in enumerate(unique_all_classes)}
            y = np.vectorize(class_id_map.get)(y)
            ye = np.vectorize(class_id_map.get)(ye)
            self.class_id_map = class_id_map
            self.class_id_reverse_map = {idx: label for label, idx in class_id_map.items()}

        num_classes = int(len(unique_all_classes))
        self.num_classes = num_classes

        print(f"Train data length: {X.shape[0]}, Eval data len: {Xe.shape[0]}")
        train_target_dist = np.unique_counts(y)
        eval_target_dist  = np.unique_counts(ye)
        print("Target Distribution, Train",train_target_dist," Eval: ",eval_target_dist)
        
        if str(os.getenv('FEATURE_GENERATOR')).upper()=="OPENFE":
          X = pd.DataFrame(X, columns=train_ds.element_spec[0].keys())
          Xe = pd.DataFrame(Xe, columns=train_ds.element_spec[0].keys())
          X, Xe, self.feature_transformer = auto_expand_feature_fe(X, y, Xe, metadata=fn_args)

        best_params = self._find_best_params(X, y, num_classes=num_classes) if self._optuna_enabled() else {}
        xgb_model = self.build_xgb_model(best_params, num_classes=num_classes)
        self._fit_xgb_model(xgb_model, X, y, eval_set=[(X, y), (Xe, ye)])
        results = xgb_model.evals_result()
        metric_key = self._eval_metric_key(num_classes)
        self.train_loss = results["validation_0"][metric_key][-1]
        self.eval_loss  = results["validation_1"][metric_key][-1]
        self.xgb_model = xgb_model
        return self.xgb_model

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
            return {"output": self.model(examples)}

        return serve