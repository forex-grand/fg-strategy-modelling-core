from src.models_architecture.base_xgb_model import XGBTrainModel,_COMMON
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

class XGBHighCapacity(XGBTrainModel):
    """
    High-capacity config — many deep trees, slow learner.
    n_estimators=1000 + lr=0.01 follows the Kapoor & Perrone (2021) finding
    that lower-LR configs maintain ranking advantages on large tabular datasets.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.75,
            colsample_bytree=0.6,
            colsample_bylevel=0.6,
            min_child_weight=3,
            gamma=0.5,
            reg_alpha=0.1,
            reg_lambda=2,
        )
