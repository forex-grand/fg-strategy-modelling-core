import xgboost

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGS  (baseline → conservative generalisation, balanced → class-weighted,
#            deep_reg → high-capacity + strong regularisation)
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "baseline": dict(
        n_estimators=400, max_depth=6,  learning_rate=0.10,
        subsample=0.80, colsample_bytree=0.80, min_child_weight=5,
        gamma=0.0,  reg_alpha=0.0, reg_lambda=1.0,
    ),
    "balanced": dict(
        n_estimators=600, max_depth=6,  learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.70, min_child_weight=3,
        gamma=0.10, reg_alpha=0.10, reg_lambda=1.0,
        # sample_weight is computed automatically for this config
        _use_class_weight=True,
    ),
    "deep_reg": dict(
        n_estimators=800, max_depth=8,  learning_rate=0.03,
        subsample=0.70, colsample_bytree=0.60, min_child_weight=10,
        gamma=0.20, reg_alpha=0.50, reg_lambda=2.0,
    ),
}

CLASS_NAMES = ["long (0)", "short (1)", "neutral (2)"]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _to_numpy(v):
    return v.numpy() if hasattr(v, "numpy") else np.array(v)

def _features_to_matrix(features: dict):
    """Stack all non-target keys → (N, F) float32 matrix + column names."""
    cols, names = [], []
    for k, v in features.items():
        if k == "target":
            continue
        arr = _to_numpy(v).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        n = arr.shape[1]
        cols.append(arr)
        names += [k] if n == 1 else [f"{k}[{i}]" for i in range(n)]
    return np.concatenate(cols, axis=1), names

def _class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    return np.array([len(y) / (len(classes) * freq[c]) for c in y], dtype=np.float32)

def _sep(char="─", w=64): print(char * w)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(
    train_features: dict,
    eval_features:  dict,
    config:         str  = "balanced",   # "baseline" | "balanced" | "deep_reg" | "all"
    early_stopping: int  = 40,
    save_models:    bool = True,
):
    """
    Train XGBoost and print per-class precision / recall / F1 on eval data.

    Args:
        train_features  : dict returned by Preprocess.preprocess(data, training=True)
        eval_features   : same structure, for the eval split
        config          : one of the CONFIGS keys, or "all" to run every config
        early_stopping  : stop after this many rounds with no improvement
        save_models     : save each model as xgb_<config>.ubj

    Returns:
        dict  {config_name: {"model": XGBClassifier, "report": str, ...}}
    """

    # ── 1. Parse inputs ───────────────────────────────────────────────────────
    X_train, col_names = _features_to_matrix(train_features)
    y_train = _to_numpy(train_features["target"]).ravel().astype(np.int32)

    X_eval, _ = _features_to_matrix(eval_features)
    y_eval    = _to_numpy(eval_features["target"]).ravel().astype(np.int32)

    n_classes = len(np.unique(np.concatenate([y_train, y_eval])))

    _sep("═")
    print(f"  XGBoost Training")
    _sep("═")
    print(f"  Train : {X_train.shape[0]:,} samples  |  {X_train.shape[1]} features")
    print(f"  Eval  : {X_eval.shape[0]:,} samples")
    print(f"  Features: {col_names}")
    print()

    # class distribution
    print("  Train class distribution:")
    for cls, cnt in sorted(Counter(y_train.tolist()).items()):
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        print(f"    class {cls} {label:<14} {cnt:>7,}  ({100*cnt/len(y_train):.1f}%)")

    # ── 2. Decide which configs to run ────────────────────────────────────────
    run = list(CONFIGS.keys()) if config == "all" else [config]
    all_results = {}

    for cfg_name in run:
        _sep()
        print(f"  Config: {cfg_name}")
        _sep()

        params = {k: v for k, v in CONFIGS[cfg_name].items()
                  if not k.startswith("_")}
        use_weights = CONFIGS[cfg_name].get("_use_class_weight", False)

        model = XGBClassifier(
            **params,
            objective       = "multi:softprob",
            num_class       = n_classes,
            eval_metric     = ["mlogloss", "merror"],
            tree_method     = "hist",
            early_stopping_rounds = early_stopping,
            verbosity       = 1,
            random_state    = 42,
        )

        fit_kwargs = {"sample_weight": _class_weights(y_train)} if use_weights else {}

        model.fit(
            X_train, y_train,
            eval_set = [(X_train, y_train), (X_eval, y_eval)],
            verbose  = 100,
            **fit_kwargs,
        )

        print(f"  Best iteration: {model.best_iteration}")

        # ── 3. Evaluation ─────────────────────────────────────────────────────
        y_pred = model.predict(X_eval)

        present_classes   = sorted(np.unique(np.concatenate([y_eval, y_pred])).tolist())
        present_names     = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
                             for c in present_classes]

        report = classification_report(
            y_eval, y_pred,
            labels       = present_classes,
            target_names = present_names,
            zero_division= 0,
        )

        cm = confusion_matrix(y_eval, y_pred, labels=present_classes)

        print(f"\n  ── Eval: Precision / Recall / F1 per class ──")
        print(report)

        print(f"  ── Confusion matrix (rows=true, cols=pred) ──")
        header = f"  {'':>12}" + "".join(f"{n:>12}" for n in present_names)
        print(header)
        for i, row in enumerate(cm):
            print(f"  {present_names[i]:>12}" + "".join(f"{v:>12}" for v in row))

        # ── 4. Feature importance ─────────────────────────────────────────────
        imp = model.get_booster().get_score(importance_type="gain")
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        max_score  = max(v for _, v in sorted_imp) if sorted_imp else 1

        print(f"\n  ── Feature importance (gain) ────────────────")
        for feat, score in sorted_imp:
            idx  = int(feat[1:]) if feat[1:].isdigit() else -1
            name = col_names[idx] if 0 <= idx < len(col_names) else feat
            bar  = "█" * int(score / max_score * 28)
            print(f"  {name:<28} {score:>8.1f}  {bar}")

        # ── 5. Save ───────────────────────────────────────────────────────────
        if save_models:
            path = f"xgb_{cfg_name}.ubj"
            model.save_model(path)
            print(f"\n  Model saved → {path}")

        all_results[cfg_name] = {
            "model":    model,
            "report":   report,
            "cm":       cm,
            "y_pred":   y_pred,
        }

    # ── 6. Summary table (only when running all configs) ─────────────────────
    if len(all_results) > 1:
        _sep("═")
        print("  SUMMARY")
        _sep("═")
        from sklearn.metrics import precision_score, recall_score, f1_score
        print(f"  {'Config':<18} {'Macro-P':>9} {'Macro-R':>9} {'Macro-F1':>10}")
        _sep()
        for cfg_name, res in all_results.items():
            yp = res["y_pred"]
            mp = precision_score(y_eval, yp, average="macro", zero_division=0)
            mr = recall_score   (y_eval, yp, average="macro", zero_division=0)
            mf = f1_score       (y_eval, yp, average="macro", zero_division=0)
            print(f"  {cfg_name:<18} {mp:>9.4f} {mr:>9.4f} {mf:>10.4f}")
        _sep("═")

    return all_results
