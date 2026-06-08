from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBColumnSampled(XGBTrainModel):
    """
    Column-level + feature-level sub-sampling.
    Anyscale multiclass tuning found colsample_bytree=0.6, colsample_bylevel=0.7
    improved accuracy to 85.8% on a 10-class problem.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.6,
            colsample_bylevel=0.7,   # per-level column sampling
            min_child_weight=3,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
        )