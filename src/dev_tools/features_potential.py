"""
Feature Potential Test
======================
Tests whether preprocessed features have learning signal by checking
if XGBoost can overfit on them. If even one aggressive config memorises
the training data with good metrics, the features carry useful information.
"""

import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# ── XGBoost configs – all tuned to overfit aggressively ───────────────────────
CONFIGS = {
    "deep_dense": dict(
        n_estimators=800,
        max_depth=12,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=0,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=0,
    ),
    "shallow_many_trees": dict(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.5,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=0,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=0,
    ),
    "extreme_depth": dict(
        n_estimators=500,
        max_depth=20,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=0,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=0,
    ),
    "high_lr_no_reg": dict(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.8,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=0,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=0,
    ),
}

CLASS_LABELS = {0: "Long  (>+10)", 1: "Short (<-10)", 2: "Neutral"}

# A config is considered "passed" when it hits these thresholds ON TRAINING DATA
OVERFIT_PRECISION_THRESHOLD = 0.80
OVERFIT_RECALL_THRESHOLD    = 0.80


def _tensors_to_numpy(transformed: dict):
    """Convert all tensor values to numpy arrays."""
    return {k: (v.numpy() if hasattr(v, "numpy") else np.array(v))
            for k, v in transformed.items()}


def _flatten_features(features_np: dict, exclude=("target",)):
    """
    Flatten all features into a single 2-D matrix (n_samples, n_cols).
    - 1-D tensors  → (n, 1)
    - 2-D tensors  → kept as-is  (n, length)
    - Higher rank  → (n, -1)
    Returns X (np.ndarray) and a list of column names.
    """
    cols, names = [], []
    for name, arr in features_np.items():
        if name in exclude:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        n_cols = arr.shape[1]
        cols.append(arr)
        if n_cols == 1:
            names.append(name)
        else:
            names.extend([f"{name}[{i}]" for i in range(n_cols)])
    X = np.concatenate(cols, axis=1).astype(np.float32)
    return X, names


def _print_header(text, width=62, char="─"):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(char * width)


def _per_class_metrics(y_true, y_pred, classes):
    """Return dict: class → {precision, recall, f1}"""
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score   (y_true, y_pred, labels=classes, average=None, zero_division=0)
    f = f1_score       (y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {cls: {"precision": p[i], "recall": r[i], "f1": f[i]}
            for i, cls in enumerate(classes)}


def test_feature_potential(
    example_data,
    sequence_length: int = 100,
    eval_fraction: float = 0.001,
):
    """
    Test whether preprocessed features carry learning signal by trying to
    overfit several aggressive XGBoost configurations on the training data.

    Args:
        preprocess_cls:   The Preprocess class (not an instance).
        example_data:     Raw data dict consumed by preprocess(data, training=True).
        sequence_length:  Passed to the preprocessor constructor.
        eval_fraction:    Fraction of data held out for gap logging only (default 0.1%).

    Returns:
        dict with per-config overfit scores and a top-level 'passed' flag.
    """

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    np_data      = _tensors_to_numpy(example_data)

    target = np_data["target"].ravel().astype(int)
    X, col_names = _flatten_features(np_data, exclude=("target",))

    n_samples, n_cols = X.shape
    print(f"  Samples   : {n_samples}")
    print(f"  Features  : {len([k for k in np_data if k != 'target'])} raw  →  {n_cols} columns after flatten")

    # ── 2. Class distribution ──────────────────────────────────────────────────
    _print_header("Step 2 · Target class distribution")
    counts = Counter(target.tolist())
    classes = sorted(counts.keys())
    for cls in classes:
        n   = counts[cls]
        pct = 100.0 * n / n_samples
        print(f"  Class {cls}  {CLASS_LABELS.get(cls, str(cls)):<16}  {n:>6}  ({pct:.1f} %)")

    # ── 3. Train / eval split (eval is tiny – for gap logging only) ───────────
    _print_header("Step 3 · Train / eval split")
    n_eval  = max(1, int(n_samples * eval_fraction))
    n_train = n_samples - n_eval

    # Shuffle with fixed seed for reproducibility
    rng  = np.random.default_rng(42)
    idx  = rng.permutation(n_samples)
    train_idx, eval_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[train_idx], target[train_idx]
    X_eval,  y_eval  = X[eval_idx],  target[eval_idx]

    print(f"  Train samples : {n_train}  ({100*(1-eval_fraction):.1f} %)")
    print(f"  Eval  samples : {n_eval}   ({100*eval_fraction:.2f} % – gap logging only)")

    # ── 4. Train all configs ───────────────────────────────────────────────────
    _print_header("Step 4 · Training XGBoost configs")

    results      = {}
    passed_any   = False

    for cfg_name, cfg_params in CONFIGS.items():
        print(f"\n  ▸ Config: {cfg_name}")

        model = XGBClassifier(
            **cfg_params,
            objective="multi:softmax",
            num_class=len(classes),
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # ── Train metrics (overfit target) ─────────────────────────────────
        pred_train = model.predict(X_train)
        train_acc  = (pred_train == y_train).mean()
        train_metrics = _per_class_metrics(y_train, pred_train, classes)

        # ── Eval metrics (gap logging) ──────────────────────────────────────
        pred_eval = model.predict(X_eval)
        eval_acc  = (pred_eval == y_eval).mean()
        eval_metrics  = _per_class_metrics(y_eval, pred_eval, classes)

        # ── Overfit gap ─────────────────────────────────────────────────────
        overfit_gap = train_acc - eval_acc

        # ── Pass/fail check ─────────────────────────────────────────────────
        # A config passes if EVERY class clears the precision AND recall bar
        per_class_pass = {}
        for cls in classes:
            p = train_metrics[cls]["precision"]
            r = train_metrics[cls]["recall"]
            per_class_pass[cls] = (p >= OVERFIT_PRECISION_THRESHOLD and
                                   r >= OVERFIT_RECALL_THRESHOLD)

        config_passed = all(per_class_pass.values())
        if config_passed:
            passed_any = True

        # ── Print summary ───────────────────────────────────────────────────
        status = "✅ PASSED" if config_passed else "❌ did not fully overfit"
        print(f"    Status        : {status}")
        print(f"    Train acc     : {train_acc:.4f}   |   Eval acc : {eval_acc:.4f}   |   Gap : {overfit_gap:+.4f}")
        print()
        print(f"    {'Class':<10} {'Label':<18} {'Tr-Prec':>8} {'Tr-Rec':>8} {'Tr-F1':>7}  "
              f"{'Ev-Prec':>8} {'Ev-Rec':>8}  {'Pass?':>6}")
        print(f"    {'─'*10} {'─'*18} {'─'*8} {'─'*8} {'─'*7}  {'─'*8} {'─'*8}  {'─'*6}")
        for cls in classes:
            tm = train_metrics[cls]
            em = eval_metrics[cls]
            ok = "✓" if per_class_pass[cls] else "✗"
            print(f"    {cls:<10} {CLASS_LABELS.get(cls,str(cls)):<18} "
                  f"{tm['precision']:>8.3f} {tm['recall']:>8.3f} {tm['f1']:>7.3f}  "
                  f"{em['precision']:>8.3f} {em['recall']:>8.3f}  {ok:>6}")

        results[cfg_name] = {
            "model":          model,
            "train_acc":      train_acc,
            "eval_acc":       eval_acc,
            "overfit_gap":    overfit_gap,
            "train_metrics":  train_metrics,
            "eval_metrics":   eval_metrics,
            "per_class_pass": per_class_pass,
            "passed":         config_passed,
        }

    # ── 5. Final verdict ───────────────────────────────────────────────────────
    _print_header("Step 5 · Final verdict", char="═")

    if passed_any:
        winning = [n for n, r in results.items() if r["passed"]]
        print(f"\n  ✅  FEATURES HAVE POTENTIAL")
        print(f"  At least one config successfully overfit the training data.")
        print(f"  Winning config(s): {', '.join(winning)}")
        print(f"\n  Interpretation:")
        print(f"  ├─ Features carry sufficient signal for the model to memorise.")
        print(f"  ├─ This is a necessary (not sufficient) condition for real training.")
        print(f"  └─ Next step: run with regularisation + proper train/val/test split.")
    else:
        # Partial info: best config by macro train F1
        macro_f1s = {
            n: np.mean([r["train_metrics"][c]["f1"] for c in classes])
            for n, r in results.items()
        }
        best_name = max(macro_f1s, key=macro_f1s.get)
        best_f1   = macro_f1s[best_name]
        print(f"\n  ⚠️  NO CONFIG FULLY OVERFIT — features may lack sufficient signal.")
        print(f"  Best macro train F1 : {best_f1:.3f}  ({best_name})")
        print(f"\n  Possible reasons:")
        print(f"  ├─ Features are too similar / collinear.")
        print(f"  ├─ Target is too noisy or classes heavily imbalanced.")
        print(f"  ├─ Sequence window ({sequence_length}) may be too short/long.")
        print(f"  └─ Consider adding more or different indicator periods.")

    print(f"\n{'═'*62}\n")

    return {
        "passed":           passed_any,
        "per_config":       results,
        "feature_columns":  col_names,
        "class_counts":     dict(counts),
    }