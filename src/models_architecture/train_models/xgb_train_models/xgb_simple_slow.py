
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


class XGBSimpleSlow(XGBTrainModel):
    """
    Slow learner — low LR + more trees.
    The TDS tuning guide recommends lr=0.05 with n_estimators=500 as a strong
    default that generalises well across tabular datasets.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
        )