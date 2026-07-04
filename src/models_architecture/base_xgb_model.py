
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
from sklearn.metrics import precision_score, recall_score, accuracy_score


_COMMON = dict(
    objective="binary:logistic",
    use_label_encoder=False,
    random_state=44,
    tree_method="hist",
    device=os.getenv("XGB_DEVICE", "cpu").strip().lower(),
    eval_metric="logloss",
    verbosity=0,
    n_jobs=-1,
)


class XGBTrainModel(BaseModel):
    """
    Binary XGBoost model for trading signals.
    - Class 1: Buy / TP hit first (positive class)
    - Class 0: No-trade / SL hit first
    Preserves exclude_id filtering, sampling, OpenFE, etc.
    """

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length
        self.train_loss = 0.0
        self.eval_loss  = 0.0
        self.num_classes = 2  # Fixed for binary
        self.class_id_map = None
        self.class_id_reverse_map = None
    @staticmethod
    def _resolve_num_batches(dataset: tf.data.Dataset, fn_args) -> int:
        """Determine how many batches to process from the dataset."""
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
        if cardinality == -2:  # Unknown cardinality
            num_batches = -1
    
        return int(num_batches)
    
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

    def build_model(self, input_spec: dict[str, tf.TensorSpec]):
        inputs = {key: keras.Input(shape=(spec.shape[-1] if spec.shape.rank > 1 else 1,), name=key, dtype=spec.dtype)
                  for key, spec in input_spec.items()}
        y = keras.layers.concatenate(list(inputs.values()))
        return keras.Model(inputs, y)

    @staticmethod
    def _to_class_ids(y):
        """Ensure binary 0/1 labels after exclude_id filtering."""
        y_np = np.asarray(y).astype(np.int32).ravel()
        unique = np.unique(y_np)
        if len(unique) > 2:
            class_map = {label: idx for idx, label in enumerate(sorted(unique))}
            y_np = np.vectorize(class_map.get)(y_np)
        return y_np

    def build_xgb_model(self, params: Optional[dict] = None):
        params = dict(params or {})
        merged_params = {**_COMMON, **params}
        return XGBClassifier(**merged_params)

    @staticmethod
    def _resolve_optuna_trials() -> int:
        trials = int(os.getenv("XGB_OPTUNA_TRIALS", "30"))
        if trials <= 0:
            raise ValueError("XGB_OPTUNA_TRIALS must be greater than zero.")
        return trials

    @staticmethod
    def _optuna_enabled() -> bool:
        return os.getenv("XGB_OPTUNA_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _suggest_params(trial: optuna.Trial) -> dict:
        """Optimized suggestions for binary trading signal prediction."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 15.0),  # Crucial for imbalance
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

    def _find_best_params(self, X, y) -> dict:
        validation_fraction = 0.2  # or pull from env
        split_index = int(len(X) * (1.0 - validation_fraction))
        X_train, X_valid = X[:split_index], X[split_index:]
        y_train, y_valid = y[:split_index], y[split_index:]

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            model = self.build_xgb_model(params)
            self._fit_xgb_model(model, X_train, y_train, eval_set=[(X_valid, y_valid)])
            results = model.evals_result()
            return float(results["validation_0"]["logloss"][-1])

        sampler = optuna.samplers.TPESampler(seed=int(os.getenv("XGB_OPTUNA_RANDOM_STATE", "44")))
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
        """Binary evaluation focused on precision_buy for class 1."""
        if not hasattr(self, 'xgb_model'):
            raise ValueError("Model must be trained before evaluation.")

        y_true_list, y_pred_list, y_proba_list = [], [], []

        cardinality = eval_ds.cardinality()
        num_batches = int(os.getenv("STEPS_PER_EPOCH", "-1")) or (cardinality if cardinality > 0 else 100)
        num_eval_batches = cardinality if cardinality > 0 else num_batches
        if cardinality == -2:
            num_eval_batches = -1

        for batch_x, batch_y in eval_ds.take(num_eval_batches):
            X = np.stack([tf.squeeze(value) for value in batch_x.values()], axis=-1)

            if str(os.getenv('FEATURE_GENERATOR')).upper() == "OPENFE":
                if not hasattr(self, 'feature_transformer') or self.feature_transformer is None:
                    raise ValueError("Feature Transformer missing for OPENFE")
                X = pd.DataFrame(X, columns=eval_ds.element_spec[0].keys())
                X = transform_fe(X, self.feature_transformer)

            y_true = self._to_class_ids(batch_y)

            y_proba = self.xgb_model.predict_proba(X)[:, 1] if hasattr(self.xgb_model, 'predict_proba') else self.xgb_model.predict(X)
            y_pred = (y_proba > 0.5).astype(np.int32)

            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            y_proba_list.append(y_proba)

        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        y_proba_all = np.concatenate(y_proba_list)

        metrics = {
            'accuracy': float(accuracy_score(y_true_all, y_pred_all)),
            'precision_buy': float(precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)),
            'precision_sell': float(precision_score(y_true_all, y_pred_all, pos_label=0, zero_division=0)),
            'recall_buy': float(recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)),
            'recall_sell': float(recall_score(y_true_all, y_pred_all, pos_label=0, zero_division=0)),
            'val_loss': self.eval_loss,
            'train_loss': self.train_loss,
        }
        return metrics

    def build_train_model(self, train_ds, eval_ds, fn_args):
        inputs = self._build_input_signature()
        inputs_dict = {inp.name: inp for inp in inputs}
        x = self.preprocessor(inputs_dict)
        y = self.build_model(train_ds.element_spec[0])(x)
        model_ = keras.Model(inputs, y)
        self.model = model_

        num_batches = self._resolve_num_batches(train_ds, fn_args)  # assume this method exists in BaseModel

        # === Training data collection (preserve exclude_id) ===
        X, y = None, None
        ts = datetime.now()
        first_batch_seen = False
        exclude_id = int(os.getenv("EXCLUDE_CLASS_ID", "-1"))
        for batch_x, batch_y in train_ds.take(num_batches):
            Xd = np.stack([tf.squeeze(value) for value in batch_x.values()], axis=-1)
            yd = self._to_class_ids(batch_y)
            exclude_mask = (yd != exclude_id)
            Xd = Xd[exclude_mask]
            yd = yd[exclude_mask]
            if first_batch_seen:
                X = np.concatenate([X, Xd], axis=0)
                y = np.concatenate([y, yd], axis=0)
            else:
                X = Xd
                y = yd
                first_batch_seen = True

        print("Training data collected in", (datetime.now().timestamp() - ts.timestamp()), "s")

        # === Eval data collection ===
        Xe, ye = None, None
        first_batch_seen = False
        for batch_x, batch_y in eval_ds.take(num_batches):
            Xd = np.stack([tf.squeeze(value) for value in batch_x.values()], axis=-1)
            yd = self._to_class_ids(batch_y)
            exclude_mask = (yd != exclude_id)
            Xd = Xd[exclude_mask]
            yd = yd[exclude_mask]
            if first_batch_seen:
                Xe = np.concatenate([Xe, Xd], axis=0)
                ye = np.concatenate([ye, yd], axis=0)
            else:
                Xe = Xd
                ye = yd
                first_batch_seen = True

        # Binary label handling
        unique_train = np.unique(y)
        unique_eval = np.unique(ye)
        if len(np.union1d(unique_train, unique_eval)) > 2:
            warnings.warn("Remapping to binary 0/1")
            class_map = {label: i for i, label in enumerate(sorted(np.union1d(unique_train, unique_eval)))}
            y = np.vectorize(class_map.get)(y)
            ye = np.vectorize(class_map.get)(ye)
            self.class_id_map = class_map
            self.class_id_reverse_map = {v: k for k, v in class_map.items()}

        self.num_classes = 2
        print(f"Train shape: {X.shape}, Eval: {Xe.shape}")
        print("Target dist Train:", np.unique(y, return_counts=True), "Eval:", np.unique(ye, return_counts=True))

        if str(os.getenv('FEATURE_GENERATOR')).upper() == "OPENFE":
            X = pd.DataFrame(X, columns=train_ds.element_spec[0].keys())
            Xe = pd.DataFrame(Xe, columns=eval_ds.element_spec[0].keys())
            X, Xe, self.feature_transformer = auto_expand_feature_fe(X, y, Xe, metadata=fn_args)

        best_params = self._find_best_params(X, y) if self._optuna_enabled() else {}
        xgb_model = self.build_xgb_model(best_params)
        self._fit_xgb_model(xgb_model, X, y, eval_set=[(X, y), (Xe, ye)])

        results = xgb_model.evals_result()
        self.train_loss = results["validation_0"]["logloss"][-1]
        self.eval_loss = results["validation_1"]["logloss"][-1]
        self.xgb_model = xgb_model
        return self.xgb_model

    def get_serving_signature(self):
        # Unchanged - adapt if you need probability output
        input_signature = {
            "time": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.int64, name="time"),
            "open": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="open"),
            # ... (rest of fields)
        }

        @tf.function(input_signature=[input_signature])
        def serve(examples):
            return {"output": self.model(examples)}
        return serve
