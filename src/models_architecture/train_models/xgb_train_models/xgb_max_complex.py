
from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBMaxComplex(XGBTrainModel):
    """
    Maximum complexity — kitchen-sink regularisation.
    Combines all levers: deep trees, slow LR, elastic-net, aggressive sampling,
    high child weight, and gamma pruning. Use as an upper-bound reference;
    likely needs early stopping in practice.
    Based on parameter ranges reported across the TDS guide, XGBoost docs,
    and Kapoor & Perrone (2021).
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=1200,
            max_depth=12,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.5,
            colsample_bylevel=0.5,
            colsample_bynode=0.7,  # per-node column sampling (XGBoost ≥ 0.90)
            min_child_weight=10,
            gamma=3,
            reg_alpha=0.5,
            reg_lambda=5,
        )