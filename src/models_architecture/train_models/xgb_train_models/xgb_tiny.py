

from src.models_architecture.base_xgb_model import XGBTrainModel
from xgboost import XGBClassifier

_COMMON = dict(
    objective="multi:softmax",
    num_class=3,
    use_label_encoder=False,
    random_state=42,
    tree_method="hist",
    eval_metric="mlogloss",
    verbosity=0,
)

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

