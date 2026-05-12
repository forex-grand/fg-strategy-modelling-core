
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


class XGBBalanced(XGBTrainModel):
    """
    Well-regularised mid-range config.
    colsample_bytree=0.7, lr=0.05 came out best in the TDS random-search example.
    min_child_weight=5 adds noise robustness per the OpenReview ablation study.
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
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0,
            reg_lambda=1,
        )
