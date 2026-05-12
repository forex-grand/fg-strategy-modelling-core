from src.models_architecture.base_xgb_model import XGBTrainModel, _COMMON
from xgboost import XGBClassifier

class XGBElasticNet(XGBTrainModel):
    """
    Elastic-net regularisation (L1 + L2 combined).
    Combining alpha and lambda is recommended in the XGBoost param docs
    when both feature sparsity and weight magnitude control are needed.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=800,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.6,
            colsample_bylevel=0.6,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.3,   # L1
            reg_lambda=2,    # L2
        )

