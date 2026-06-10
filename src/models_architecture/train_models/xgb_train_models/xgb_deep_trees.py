from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBDeepTrees(XGBTrainModel):
    """
    Deep trees (max_depth=10) for complex, high-order interactions.
    Used in the Anyscale guide's best config for multiclass (max_depth=10,
    n_estimators=500, lr=0.1). Requires strong subsampling to avoid overfit.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=500,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.6,
            colsample_bylevel=0.7,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0,
            reg_lambda=2,
        )

