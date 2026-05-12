from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBHighChildWeight(XGBTrainModel):
    """
    Very high min_child_weight — strong noise resistance.
    The OpenReview ablation study showed min_child_weight=10–20 substantially
    reduces overfitting on noisy or small datasets with complex feature spaces.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=800,
            max_depth=9,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.5,
            colsample_bylevel=0.6,
            min_child_weight=15,  # very conservative splitting
            gamma=2,
            reg_alpha=0.1,
            reg_lambda=3,
        )