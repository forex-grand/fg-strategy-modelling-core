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

class XGBGammaPruned(XGBTrainModel):
    """
    Gamma (min split loss) pruning — conservative tree growth.
    gamma=1 removes splits that don't improve the objective by at least 1 unit.
    TDS guide notes gamma=1–5 as commonly effective; 10+ is very aggressive.
    """
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)

    def build_xgb_model(self):
        return XGBClassifier(
            **_COMMON,
            n_estimators=600,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=3,
            gamma=2,         # conservative split pruning
            reg_alpha=0,
            reg_lambda=1,
        )