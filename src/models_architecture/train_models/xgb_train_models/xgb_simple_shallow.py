
from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBSimpleShallow(XGBTrainModel):
    """
    Shallow trees + moderate estimators.
    max_depth=4 is a common sweet-spot for low-noise tabular data.
    Matches configs reported in the industry-classification study (Fatihah et al., 2024).
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
        )

