from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBL1Regularised(XGBTrainModel):
    """
    L1 (Lasso) regularisation — drives sparse feature use.
    Useful when many features are weakly relevant; alpha>0 effectively
    prunes noisy splits. Range 0.1–1.0 cited in Analytics Vidhya guide.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0.5,   # L1
            reg_lambda=0,
        )