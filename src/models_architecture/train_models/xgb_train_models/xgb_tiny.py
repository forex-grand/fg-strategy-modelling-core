from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBTiny(XGBTrainModel):
    """Absolute minimum — sanity-check baseline."""
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.3,
            subsample=1.0,
            colsample_bytree=1.0,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,   # XGBoost default L2
        )

