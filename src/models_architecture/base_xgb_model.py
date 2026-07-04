import os
import warnings
from typing import Optional

from src.models_architecture.base_model import BaseModel
import tensorflow as tf
import keras
import xgboost as xgb
import numpy as np
from datetime import datetime
import optuna
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.pipeline.feature_selection import auto_expand_feature_fe, transform_fe
import pandas as pd

# NOTE: objective / num_class / eval_metric / device / scale_pos_weight are resolved
# dynamically based on num_classes and the training data (see build_xgb_params and
# _compute_scale_pos_weight). Do not hardcode them here.
_COMMON = dict(
    random_state=44,
    tree_method="hist",
    verbosity=0,
    n_jobs=-1,
)

cpu_counts = os.cpu_count()

class XGBTrainModel(BaseModel):
    """
      The model on the base model object is the preprocessing model object that outputs an array of numpy.

      Uses the native xgboost.train()/DMatrix API (Booster) rather than the XGBClassifier
      sklearn wrapper. Empirically, native xgb.train() + scale_pos_weight produced
      meaningfully better validation precision on this project's imbalanced binary
      targets than the classifier-wrapper + SMOTE/sample_weight approach.
    """

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length
        self.train_loss = 0.0
        self.eval_loss  = 0.0
        self.num_classes = 0
        self.class_id_map = None
        self.class_id_reverse_map = None
        self.xgb_model: Optional[xgb.Booster] = None
        self.best_iteration: Optional[int] = None

    @staticmethod
    def _sampling_strategy() -> str:
        return (
            os.getenv("TRAIN_DATA_SAMPLING")
            or os.getenv("SAMPLING_STRATEGY")
            or os.getenv("DATA_SAMPLING_STRATEGY")
            or "weights"
        ).strip().lower().replace("-", "_")

    def _apply_sampling(self, X, y):
        """
        Returns (X, y, sample_weight).

        Only used for num_classes > 2. scale_pos_weight (the mechanism used for binary
        targets, see _compute_scale_pos_weight) is a binary-only xgboost parameter, so
        multiclass targets still go through this per-row sample_weight / resampling path.
        """
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
    def _resolve_metric_config(num_classes: int) -> tuple[str, bool]:
        """
        Returns (eval_metric, higher_is_better).

        Binary defaults to aucpr rather than logloss. logloss rewards well-calibrated
        probabilities across the *whole* distribution, which under class imbalance is
        dominated by the majority class - a model can have great logloss while having
        mediocre precision on the minority (buy/sell) classes you actually care about.
        aucpr (area under the precision-recall curve) tracks the precision/recall
        trade-off directly and is the standard XGBoost metric for imbalanced binary
        classification. Override via XGB_EVAL_METRIC if you want logloss/auc/etc back.
        """
        override = os.getenv("XGB_EVAL_METRIC", "").strip().lower()
        if num_classes <= 2:
            metric = override or "aucpr"
        else:
            # aucpr's multi-class support is inconsistent across xgboost versions;
            # stick with mlogloss for >2 classes unless explicitly overridden.
            metric = override or "mlogloss"
        higher_is_better = metric in {"aucpr", "auc", "map", "ndcg"}
        return metric, higher_is_better

    @staticmethod
    def _resolve_early_stopping_rounds() -> Optional[int]:
        rounds = int(os.getenv("XGB_EARLY_STOPPING_ROUNDS", "50"))
        return rounds if rounds > 0 else None

    @staticmethod
    def _resolve_num_boost_round() -> int:
        rounds = int(os.getenv("XGB_NUM_BOOST_ROUND", "500"))
        if rounds <= 0:
            raise ValueError("XGB_NUM_BOOST_ROUND must be greater than zero.")
        return rounds

    @staticmethod
    def _resolve_best_iteration(booster: xgb.Booster) -> int:
        """
        best_iteration only exists on the Booster when early stopping actually ran
        (confirmed empirically: accessing it otherwise raises AttributeError with
        "best_iteration is only defined when early stopping is used"). Falls back to
        the last boosted round when early stopping was disabled.
        """
        best_iteration = getattr(booster, "best_iteration", None)
        if best_iteration is not None:
            return int(best_iteration)
        return int(booster.num_boosted_rounds()) - 1

    @staticmethod
    def _resolve_best_score(booster: xgb.Booster, evals_result: dict, dataset_key: str, metric_key: str) -> float:
        """
        best_score has the same early-stopping-only availability caveat as
        best_iteration (see _resolve_best_iteration). Falls back to reading the last
        recorded value for dataset_key/metric_key out of evals_result.
        """
        best_score = getattr(booster, "best_score", None)
        if best_score is not None:
            return float(best_score)
        return float(evals_result[dataset_key][metric_key][-1])

    def build_xgb_params(
        self,
        params: Optional[dict] = None,
        num_classes: int = 2,
        scale_pos_weight: Optional[float] = None,
    ) -> dict:
        """
        Builds the native xgboost params dict (for xgb.train/Booster), with
        objective/eval_metric/num_class/device chosen based on num_classes.

        binary:logistic must NOT receive num_class - confirmed empirically this raises
        "Check failed: info.labels.Size() == preds.Size()" because XGBoost then emits
        num_classes predictions per sample against single-column binary labels.

        scale_pos_weight is a binary-only xgboost parameter; it is ignored (popped)
        for num_classes > 2, where per-row sample_weight at DMatrix construction time
        should be used instead (see _apply_sampling).

        device is resolved here (not at module import time) so that setting the
        XGB_DEVICE env var after this module has already been imported - a common
        Colab pattern - still takes effect.
        """
        params = dict(params or {})
        common = dict(_COMMON)
        common["device"] = os.getenv("XGB_DEVICE", "cpu").strip().lower()
        metric, _ = self._resolve_metric_config(num_classes)

        if num_classes <= 2:
            common["objective"] = "binary:logistic"
            common["eval_metric"] = metric
            common.pop("num_class", None)
            if scale_pos_weight is not None:
                common["scale_pos_weight"] = scale_pos_weight
        else:
            common["objective"] = "multi:softprob"
            common["eval_metric"] = metric
            common["num_class"] = num_classes
            common.pop("scale_pos_weight", None)

        merged = {**common, **params}
        # num_boost_round/n_estimators are xgb.train() arguments, not booster params -
        # confirmed empirically that leaving them in the params dict doesn't error, but
        # does trigger a "Parameters: {...} are not used" UserWarning on every call.
        merged.pop("num_boost_round", None)
        merged.pop("n_estimators", None)
        merged.pop("use_label_encoder", None)
        if num_classes <= 2:
            merged.pop("num_class", None)
        return merged

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
        """
        Suggests booster hyperparameters plus num_boost_round. num_boost_round is not
        a native xgboost booster param - it's an argument to xgb.train() - so callers
        must pop it out of this dict before passing the rest through build_xgb_params
        (build_xgb_params also defensively strips it, in case a caller forgets).
        """
        return {
            "num_boost_round": trial.suggest_int("num_boost_round", 100, 1200),
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

    @staticmethod
    def _compute_scale_pos_weight(y) -> Optional[float]:
        """
        Binary-only. Mirrors the working reference script's approach: compute_class_weight
        ('balanced', ...) then take the ratio of the two resulting weights. Returns None
        if y doesn't have exactly 2 classes (e.g. a batch that's missing one class, or a
        genuinely multiclass target - callers should only use this when num_classes == 2).
        """
        classes = np.unique(y)
        if len(classes) != 2:
            return None
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        # np.unique returns classes sorted ascending; this pipeline guarantees contiguous
        # zero-based labels upstream (see the remapping block in build_train_model), so
        # classes[0] == 0 and classes[1] == 1 here.
        return float(class_weights[1] / class_weights[0])

    @staticmethod
    def _make_dmatrix(X, y=None, sample_weight=None) -> xgb.DMatrix:
        if y is not None:
            return xgb.DMatrix(X, label=y, weight=sample_weight)
        return xgb.DMatrix(X)

    def _fit_booster(self, params: dict, dtrain: xgb.DMatrix, evals: list, num_boost_round: int):
        """
        Trains via the native xgb.train() API. Returns (booster, evals_result).

        early_stopping_rounds is only passed when there's at least one eval set,
        matching xgb.train()'s requirement that early stopping needs something to
        monitor. Confirmed empirically that when there IS more than one entry in
        `evals`, xgboost uses the LAST entry for early-stopping decisions - so callers
        should always put the validation set last (as build_train_model does).
        """
        early_stopping_rounds = self._resolve_early_stopping_rounds() if evals else None
        evals_result: dict = {}
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=int(os.getenv("XGB_VERBOSE", "0")),
            evals_result=evals_result,
        )
        return booster, evals_result

    def _predict_proba(self, booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
        """
        Predicts using the booster's best_iteration boundary explicitly.

        Confirmed empirically: Booster.predict() does NOT automatically limit itself
        to best_iteration (unlike the XGBClassifier sklearn wrapper's .predict()) - by
        default it uses every boosted round, including the extra `early_stopping_rounds`
        of rounds trained past the best one. Passing iteration_range explicitly is
        required to actually get the benefit of early stopping at inference time;
        without it, early stopping only limits training time, not prediction quality.
        """
        best_iteration = self._resolve_best_iteration(booster)
        return booster.predict(dmatrix, iteration_range=(0, best_iteration + 1))

    def _find_best_params(
        self,
        X,
        y,
        num_classes: int,
        scale_pos_weight: Optional[float],
    ) -> dict:
        validation_fraction = self._resolve_validation_fraction()
        split_index = int(len(X) * (1.0 - validation_fraction))
        if split_index <= 0 or split_index >= len(X):
            raise ValueError("Not enough XGBoost training samples for Optuna train/validation split.")

        X_train, X_valid = X[:split_index], X[split_index:]
        y_train, y_valid = y[:split_index], y[split_index:]
        metric, higher_is_better = self._resolve_metric_config(num_classes)

        train_weight = None
        if num_classes > 2:
            X_train, y_train, train_weight = self._apply_sampling(X_train, y_train)

        dtrain = self._make_dmatrix(X_train, y_train, sample_weight=train_weight)
        dvalid = self._make_dmatrix(X_valid, y_valid)

        def objective(trial: optuna.Trial) -> float:
            suggested = self._suggest_params(trial)
            num_boost_round = suggested["num_boost_round"]
            params = self.build_xgb_params(
                suggested, num_classes=num_classes, scale_pos_weight=scale_pos_weight
            )
            booster, evals_result = self._fit_booster(
                params, dtrain, evals=[(dvalid, "validation")], num_boost_round=num_boost_round
            )
            return self._resolve_best_score(booster, evals_result, "validation", metric)

        sampler = optuna.samplers.TPESampler(
            seed=int(os.getenv("XGB_OPTUNA_RANDOM_STATE", "44"))
        )
        study = optuna.create_study(
            direction="maximize" if higher_is_better else "minimize",
            sampler=sampler,
        )
        timeout_value = os.getenv("XGB_OPTUNA_TIMEOUT")
        timeout = int(timeout_value) if timeout_value else None
        study.optimize(
            objective,
            n_trials=self._resolve_optuna_trials(),
            timeout=timeout,
            show_progress_bar=False,
        )
        print("Best XGBoost Optuna params: ", study.best_params)
        print(f"Best XGBoost Optuna validation {metric}: ", study.best_value)
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
            dmatrix = self._make_dmatrix(X)
            y_proba = np.asarray(self._predict_proba(self.xgb_model, dmatrix))

            if y_proba.ndim > 1 and y_proba.shape[-1] > 1:
                # multi:softprob -> (n, num_classes) probability rows
                y_pred = np.argmax(y_proba, axis=-1)
            else:
                # binary:logistic -> (n,) probability of the positive class
                y_pred = (y_proba > 0.5).astype(np.int32)

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

        # scale_pos_weight (binary-only) mirrors the working reference script; for
        # num_classes > 2 this is None and _apply_sampling's per-row sample_weight is
        # used instead (see below).
        scale_pos_weight = self._compute_scale_pos_weight(y) if num_classes <= 2 else None

        X_fit, y_fit = X, y
        train_weight = None
        if num_classes > 2:
            X_fit, y_fit, train_weight = self._apply_sampling(X, y)

        best_params = (
            self._find_best_params(X_fit, y_fit, num_classes=num_classes, scale_pos_weight=scale_pos_weight)
            if self._optuna_enabled() else {}
        )
        num_boost_round = best_params.pop("num_boost_round", None) or self._resolve_num_boost_round()
        params = self.build_xgb_params(best_params, num_classes=num_classes, scale_pos_weight=scale_pos_weight)

        dtrain = self._make_dmatrix(X_fit, y_fit, sample_weight=train_weight)
        dvalid = self._make_dmatrix(Xe, ye)

        booster, evals_result = self._fit_booster(
            params,
            dtrain,
            evals=[(dtrain, "train"), (dvalid, "validation")],
            num_boost_round=num_boost_round,
        )

        metric, _ = self._resolve_metric_config(num_classes)
        best_iteration = self._resolve_best_iteration(booster)
        # NOTE: despite the attribute names (kept for backward compat with evaluate()'s
        # output dict), these now hold whatever metric is configured - aucpr by default
        # for binary, so higher is better here, not lower.
        self.train_loss = float(evals_result["train"][metric][best_iteration])
        self.eval_loss = self._resolve_best_score(booster, evals_result, "validation", metric)
        self.best_iteration = best_iteration
        self.xgb_model = booster
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
