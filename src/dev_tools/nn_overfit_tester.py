"""
Neural Network Overfit Tester
==============================
Builds a simple Dense model and tries to overfit on preprocessed features.
If the model can memorise the training data, the features carry learning signal.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import keras
import tensorflow as tf
from collections import Counter


# ── Thresholds ─────────────────────────────────────────────────────────────────
OVERFIT_ACCURACY_THRESHOLD = 0.90      # train accuracy to consider "overfit achieved"
EVAL_FRACTION              = 0.001     # fraction held out for gap logging only


CLASS_LABELS = {0: "Long  (>+10)", 1: "Short (<-10)", 2: "Neutral"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(v):
    return v.numpy() if hasattr(v, "numpy") else np.array(v)


def _flatten_and_concat(features_np: dict, exclude=("target",)):
    """
    Flatten every feature tensor to 2-D (batch, cols) then hstack.
    Returns X (float32 ndarray) and column name list.
    """
    cols, names = [], []
    for name, arr in features_np.items():
        if name in exclude:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        # arr is now (batch, n_cols)
        n = arr.shape[1]
        cols.append(arr.astype(np.float32))
        names += [name] if n == 1 else [f"{name}[{i}]" for i in range(n)]

    X = np.concatenate(cols, axis=1)
    return X, names


def _print_section(title, char="─", width=64):
    print(f"\n{char*width}\n  {title}\n{char*width}")


def _safe_metrics(y_true, y_pred, classes):
    """Per-class precision, recall, f1 (pure numpy, no sklearn needed)."""
    result = {}
    for cls in classes:
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        result[cls] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
    return result


def _build_model(input_dim: int, n_classes: int, config: dict) -> keras.Model:
    """Build a Sequential Dense model from a config dict."""
    layers_cfg = config["layers"]            # list of unit counts
    dropout     = config.get("dropout", 0.0)
    activation  = config.get("activation", "relu")
    l2_reg      = config.get("l2", 0.0)

    reg = keras.regularizers.l2(l2_reg) if l2_reg else None

    model = keras.Sequential(name=config["name"])
    model.add(keras.layers.Input(shape=(input_dim,)))

    for units in layers_cfg:
        model.add(keras.layers.Dense(
            units, activation=activation,
            kernel_regularizer=reg,
        ))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Model configs  – all push toward overfitting
# ──────────────────────────────────────────────────────────────────────────────
def _get_configs(input_dim: int):
    return [
        {
            "name":       "wide_shallow",
            "layers":     [512, 512],
            "lr":         1e-3,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     300,
            "batch_size": 32,
            "activation": "relu",
            "note":       "Wide, no regularisation – quick memoriser",
        },
        {
            "name":       "deep_narrow",
            "layers":     [256, 256, 256, 256, 128],
            "lr":         5e-4,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     400,
            "batch_size": 16,
            "activation": "relu",
            "note":       "Deep, tiny batches – forces memorisation",
        },
        {
            "name":       "massive_overfit",
            "layers":     [1024, 1024, 512, 256],
            "lr":         2e-3,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     500,
            "batch_size": 64,
            "activation": "relu",
            "note":       "Huge capacity, high LR – brute-force memoriser",
        },
        {
            "name":       "leaky_deep",
            "layers":     [512, 256, 256, 128, 128, 64],
            "lr":         1e-3,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     400,
            "batch_size": 32,
            "activation": "leaky_relu",
            "note":       "Deep with leaky relu – avoids dying neurons",
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def test_nn_overfit(
    preprocess_cls,
    example_data,
    sequence_length: int = 100,
    eval_fraction:   float = EVAL_FRACTION,
    verbose_keras:   int   = 0,          # 0=silent, 1=progress bar, 2=one line/epoch
):
    """
    Test whether preprocessed features can be memorised by a Dense network.

    Args:
        preprocess_cls:  The Preprocess class (not an instance).
        example_data:    Raw data dict passed to preprocess(data, training=True).
        sequence_length: Passed to the preprocessor constructor.
        eval_fraction:   Fraction held out for overfit-gap logging only.
        verbose_keras:   Keras fit verbosity level.

    Returns:
        dict with per-config results and a top-level 'passed' flag.
    """

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    _print_section("Step 1 · Preprocessing")
    preprocessor = preprocess_cls(sequence_length=sequence_length)
    transformed  = preprocessor.preprocess(example_data, training=True)
    np_data      = {k: _to_numpy(v) for k, v in transformed.items()}

    target   = np_data["target"].ravel().astype(np.int32)
    X, names = _flatten_and_concat(np_data, exclude=("target",))

    n_samples, n_cols = X.shape
    n_classes         = len(set(target.tolist()))
    print(f"  Samples        : {n_samples}")
    print(f"  Input columns  : {n_cols}  (after flattening 2-D features)")
    print(f"  Classes        : {n_classes}")

    # ── 2. Class distribution ──────────────────────────────────────────────────
    _print_section("Step 2 · Target distribution")
    counts  = Counter(target.tolist())
    classes = sorted(counts.keys())
    for cls in classes:
        n   = counts[cls]
        pct = 100.0 * n / n_samples
        print(f"  Class {cls}  {CLASS_LABELS.get(cls, str(cls)):<18}  {n:>6}  ({pct:.1f} %)")

    # ── 3. Split ───────────────────────────────────────────────────────────────
    _print_section("Step 3 · Train / eval split")
    n_eval  = max(1, int(n_samples * eval_fraction))
    n_train = n_samples - n_eval

    rng        = np.random.default_rng(42)
    idx        = rng.permutation(n_samples)
    tr_idx, ev_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[tr_idx], target[tr_idx]
    X_eval,  y_eval  = X[ev_idx],  target[ev_idx]
    print(f"  Train : {n_train}   Eval (gap log only) : {n_eval}")

    # Normalise inputs (z-score on train stats)
    mu    = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sigma
    X_eval  = (X_eval  - mu) / sigma

    # ── 4. Train configs ───────────────────────────────────────────────────────
    _print_section("Step 4 · Training configs")
    configs  = _get_configs(n_cols)
    results  = {}
    passed_any = False

    for cfg in configs:
        print(f"\n  ▸ {cfg['name']}  —  {cfg['note']}")
        print(f"    layers={cfg['layers']}  lr={cfg['lr']}  "
              f"epochs={cfg['epochs']}  batch={cfg['batch_size']}")

        model = _build_model(n_cols, n_classes, cfg)

        # Early stop only if train acc is already perfect (save time)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="accuracy",
            patience=40,
            restore_best_weights=True,
            baseline=0.999,
        )

        history = model.fit(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=[early_stop],
            verbose=verbose_keras,
        )

        epochs_ran = len(history.history["loss"])

        # ── Train metrics ──────────────────────────────────────────────────
        pred_train_prob  = model.predict(X_train, verbose=0)
        pred_train       = pred_train_prob.argmax(axis=1)
        train_acc        = (pred_train == y_train).mean()
        train_loss       = history.history["loss"][-1]

        # ── Eval metrics (gap only) ────────────────────────────────────────
        pred_eval_prob   = model.predict(X_eval, verbose=0)
        pred_eval        = pred_eval_prob.argmax(axis=1)
        eval_acc         = (pred_eval == y_eval).mean()

        # ── Per-class metrics ──────────────────────────────────────────────
        train_cls_metrics = _safe_metrics(y_train, pred_train, classes)
        eval_cls_metrics  = _safe_metrics(y_eval,  pred_eval,  classes)

        overfit_gap = train_acc - eval_acc
        config_passed = train_acc >= OVERFIT_ACCURACY_THRESHOLD
        if config_passed:
            passed_any = True

        status = "✅ PASSED" if config_passed else "❌ did not overfit"

        # ── Print ──────────────────────────────────────────────────────────
        print(f"\n    Status       : {status}   (ran {epochs_ran}/{cfg['epochs']} epochs)")
        print(f"    Train acc    : {train_acc:.4f}   Train loss : {train_loss:.4f}")
        print(f"    Eval  acc    : {eval_acc:.4f}   Overfit gap: {overfit_gap:+.4f}")
        print()
        print(f"    {'Class':<6} {'Label':<18} {'Tr-Prec':>8} {'Tr-Rec':>8} "
              f"{'Tr-F1':>7}  {'Ev-Prec':>8} {'Ev-Rec':>7}")
        print(f"    {'─'*6} {'─'*18} {'─'*8} {'─'*8} {'─'*7}  {'─'*8} {'─'*7}")
        for cls in classes:
            tm = train_cls_metrics[cls]
            em = eval_cls_metrics[cls]
            print(f"    {cls:<6} {CLASS_LABELS.get(cls, str(cls)):<18} "
                  f"{tm['precision']:>8.3f} {tm['recall']:>8.3f} {tm['f1']:>7.3f}  "
                  f"{em['precision']:>8.3f} {em['recall']:>7.3f}")

        results[cfg["name"]] = {
            "model":             model,
            "train_acc":         train_acc,
            "eval_acc":          eval_acc,
            "overfit_gap":       overfit_gap,
            "train_cls_metrics": train_cls_metrics,
            "eval_cls_metrics":  eval_cls_metrics,
            "epochs_ran":        epochs_ran,
            "history":           history.history,
            "passed":            config_passed,
        }

    # ── 5. Final verdict ───────────────────────────────────────────────────────
    _print_section("Step 5 · Final verdict", char="═")

    if passed_any:
        winners = [n for n, r in results.items() if r["passed"]]
        best    = max(results, key=lambda n: results[n]["train_acc"])
        best_acc = results[best]["train_acc"]
        print(f"\n  ✅  FEATURES HAVE LEARNING SIGNAL")
        print(f"  The network successfully overfit the training data.")
        print(f"  Passing config(s)  : {', '.join(winners)}")
        print(f"  Best train accuracy: {best_acc:.4f}  ({best})")
        print()
        print(f"  Interpretation:")
        print(f"  ├─ A Dense model can memorise these features → signal exists.")
        print(f"  ├─ High overfit gap is expected and desired here.")
        print(f"  └─ Next: train with proper regularisation + validation split.")
    else:
        best     = max(results, key=lambda n: results[n]["train_acc"])
        best_acc = results[best]["train_acc"]
        print(f"\n  ⚠️  NO CONFIG OVERFIT — features may lack sufficient signal.")
        print(f"  Best train accuracy: {best_acc:.4f}  ({best})")
        print()
        print(f"  Possible causes:")
        print(f"  ├─ Too few samples to memorise (try a larger subset).")
        print(f"  ├─ Features are too collinear or uninformative.")
        print(f"  ├─ Target noise is too high relative to feature signal.")
        print(f"  ├─ Class imbalance preventing minority class learning.")
        print(f"  └─ Try increasing epochs or adding more indicator periods.")

    print(f"\n{'═'*64}\n")

    return {
        "passed":          passed_any,
        "per_config":      results,
        "feature_columns": names,
        "class_counts":    dict(counts),
        "input_dim":       n_cols,
    }
