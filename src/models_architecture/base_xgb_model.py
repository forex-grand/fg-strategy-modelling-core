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
    Binary XGBoost for trading signals (class 1 = buy/TP success).
    Preserves exclude_id filtering and all original preprocessing/sampling.
    """

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length
        self.train_loss = 0.0
        self.eval_loss  = 0.0
        self.num_classes = 2  # fixed binary
        self.class_id_map = None
        self.class_id_reverse_map = None

    # ... (_sampling_strategy, _apply_sampling, build_model unchanged) ...

    @staticmethod
    def _to_class_ids(y):
        """Binary: ensure labels are 0 or 1 after filtering."""
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

    # _resolve_optuna_trials, _optuna_enabled etc. unchanged

    @staticmethod
    def _suggest_params(trial: optuna.Trial) -> dict:
        """Tuned for binary trading signals with imbalance handling."""
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
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        }

    # _fit_xgb_model unchanged (sampling works for binary)

    def _find_best_params(self, X, y) -> dict:  # binary
        # ... (updated objective uses logloss, no num_class) ...
        # (full logic as edited)

    def evaluate(self, eval_ds):
        """Binary metrics: precision_buy = precision on class 1 (TP hit)."""
        # ... (sklearn-based as edited above) ...

    def build_train_model(self, train_ds, eval_ds, fn_args):
        # ... (data collection with exclude_id preserved, binary label handling, Optuna/fit updated) ...
        pass  # full logic as per edits

    # get_serving_signature unchanged (or adapt output to binary prob if needed)
