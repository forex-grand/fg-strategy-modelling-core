import numpy as np
import warnings
warnings.filterwarnings("ignore")

import keras
import tensorflow as tf
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


# ── Thresholds ─────────────────────────────────────────────────────────────────
OVERFIT_ACCURACY_THRESHOLD = 0.90
EVAL_FRACTION              = 0.001

CLASS_LABELS = {0: "Long  (>+10)", 1: "Short (<-10)", 2: "Neutral"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(v):
    return v.numpy() if hasattr(v, "numpy") else np.array(v)


def _flatten_and_concat(features_np: dict, exclude=("target",)):
    cols, names = [], []
    for name, arr in features_np.items():
        if name in exclude:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        n = arr.shape[1]
        cols.append(arr.astype(np.float32))
        names += [name] if n == 1 else [f"{name}[{i}]" for i in range(n)]
    X = np.concatenate(cols, axis=1)
    return X, names


def _print_section(title, char="─", width=64):
    print(f"\n{char*width}\n  {title}\n{char*width}")


def _safe_metrics(y_true, y_pred, classes):
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
    layers_cfg = config["layers"]
    dropout    = config.get("dropout", 0.0)
    activation = config.get("activation", "relu")
    l2_reg     = config.get("l2", 0.0)
    reg        = keras.regularizers.l2(l2_reg) if l2_reg else None

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
# Model configs
# ──────────────────────────────────────────────────────────────────────────────
def _get_configs():
    return [
        {
            "name":       "wide_shallow",
            "layers":     [512, 512],
            "lr":         1e-3,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     500,
            "batch_size": 16,
            "activation": "relu",
            "note":       "Wide, no reg, small batch",
        },
        {
            "name":       "deep_narrow",
            "layers":     [256, 256, 256, 256, 128],
            "lr":         5e-4,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     600,
            "batch_size": 8,
            "activation": "relu",
            "note":       "Deep, tiny batch — forces minority class learning",
        },
        {
            "name":       "massive_overfit",
            "layers":     [1024, 1024, 512, 256],
            "lr":         2e-3,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     600,
            "batch_size": 16,
            "activation": "relu",
            "note":       "Huge capacity, high LR",
        },
        {
            "name":       "leaky_deep",
            "layers":     [512, 256, 256, 128, 128, 64],
            "lr":         1e-3,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     600,
            "batch_size": 8,
            "activation": "leaky_relu",
            "note":       "Deep leaky relu — avoids dying neurons on imbalanced data",
        },
        {
            "name":       "tiny_batch_brute",
            "layers":     [256, 128, 128, 64],
            "lr":         5e-4,
            "dropout":    0.0,
            "l2":         0.0,
            "epochs":     1000,
            "batch_size": 4,
            "activation": "relu",
            "note":       "Batch=4 guarantees minority class exposure every few steps",
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
    verbose_keras:   int   = 0,
):
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
    print(f"  Input columns  : {n_cols}")
    print(f"  Classes        : {n_classes}")

    # ── 2. Class distribution ──────────────────────────────────────────────────
    _print_section("Step 2 · Target distribution")
    counts  = Counter(target.tolist())
    classes = sorted(counts.keys())
    for cls in classes:
        n   = counts[cls]
        pct = 100.0 * n / n_samples
        print(f"  Class {cls}  {CLASS_LABELS.get(cls, str(cls)):<18}  {n:>6}  ({pct:.1f} %)")

    # ── 3. Class weights ───────────────────────────────────────────────────────
    _print_section("Step 3 · Class weights (balanced)")
    cw = compute_class_weight("balanced", classes=np.array(classes), y=target)
    class_weight_dict = dict(zip(classes, cw))
    for cls, w in class_weight_dict.items():
        print(f"  Class {cls}  {CLASS_LABELS.get(cls, str(cls)):<18}  weight = {w:.4f}")

    # ── 4. Split ───────────────────────────────────────────────────────────────
    _print_section("Step 4 · Train / eval split")
    n_eval  = max(1, int(n_samples * eval_fraction))
    n_train = n_samples - n_eval

    rng            = np.random.default_rng(42)
    idx            = rng.permutation(n_samples)
    tr_idx, ev_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[tr_idx], target[tr_idx]
    X_eval,  y_eval  = X[ev_idx], target[ev_idx]
    print(f"  Train : {n_train}   Eval (gap log only) : {n_eval}")

    # Z-score normalise on train stats
    mu    = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0,  keepdims=True) + 1e-8
    X_train = (X_train - mu) / sigma
    X_eval  = (X_eval  - mu) / sigma

    # ── 5. Train configs ───────────────────────────────────────────────────────
    _print_section("Step 5 · Training configs")
    configs    = _get_configs()
    results    = {}
    passed_any = False

    for cfg in configs:
        print(f"\n  ▸ {cfg['name']}  —  {cfg['note']}")
        print(f"    layers={cfg['layers']}  lr={cfg['lr']}  "
              f"epochs={cfg['epochs']}  batch={cfg['batch_size']}")

        model = _build_model(n_cols, n_classes, cfg)

        # Early stopping — no baseline, generous patience
        early_stop = keras.callbacks.EarlyStopping(
            monitor="accuracy",
            patience=80,
            restore_best_weights=True,
            min_delta=1e-4,
        )

        # Reduce LR on plateau to squeeze out last accuracy gains
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=30,
            min_lr=1e-6,
            verbose=0,
        )

        history = model.fit(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            class_weight=class_weight_dict,   # ← handles imbalance
            callbacks=[early_stop, reduce_lr],
            verbose=verbose_keras,
        )

        epochs_ran = len(history.history["loss"])

        # ── Metrics ────────────────────────────────────────────────────────
        pred_train = model.predict(X_train, verbose=0).argmax(axis=1)
        pred_eval  = model.predict(X_eval,  verbose=0).argmax(axis=1)

        train_acc  = (pred_train == y_train).mean()
        eval_acc   = (pred_eval  == y_eval ).mean()
        train_loss = history.history["loss"][-1]
        overfit_gap = train_acc - eval_acc

        train_cls = _safe_metrics(y_train, pred_train, classes)
        eval_cls  = _safe_metrics(y_eval,  pred_eval,  classes)

        config_passed = train_acc >= OVERFIT_ACCURACY_THRESHOLD
        if config_passed:
            passed_any = True

        status = "✅ PASSED" if config_passed else "❌ did not overfit"

        print(f"\n    Status       : {status}   (ran {epochs_ran}/{cfg['epochs']} epochs)")
        print(f"    Train acc    : {train_acc:.4f}   Train loss : {train_loss:.4f}")
        print(f"    Eval  acc    : {eval_acc:.4f}   Overfit gap: {overfit_gap:+.4f}")
        print()
        print(f"    {'Class':<6} {'Label':<18} {'Tr-Prec':>8} {'Tr-Rec':>8} "
              f"{'Tr-F1':>7}  {'Ev-Prec':>8} {'Ev-Rec':>7}")
        print(f"    {'─'*6} {'─'*18} {'─'*8} {'─'*8} {'─'*7}  {'─'*8} {'─'*7}")
        for cls in classes:
            tm = train_cls[cls]
            em = eval_cls[cls]
            print(f"    {cls:<6} {CLASS_LABELS.get(cls, str(cls)):<18} "
                  f"{tm['precision']:>8.3f} {tm['recall']:>8.3f} {tm['f1']:>7.3f}  "
                  f"{em['precision']:>8.3f} {em['recall']:>7.3f}")

        results[cfg["name"]] = {
            "model":        model,
            "train_acc":    train_acc,
            "eval_acc":     eval_acc,
            "overfit_gap":  overfit_gap,
            "train_cls":    train_cls,
            "eval_cls":     eval_cls,
            "epochs_ran":   epochs_ran,
            "history":      history.history,
            "passed":       config_passed,
        }

    # ── 6. Final verdict ───────────────────────────────────────────────────────
    _print_section("Step 6 · Final verdict", char="═")

    best     = max(results, key=lambda n: results[n]["train_acc"])
    best_acc = results[best]["train_acc"]

    if passed_any:
        winners = [n for n, r in results.items() if r["passed"]]
        print(f"\n  ✅  FEATURES HAVE LEARNING SIGNAL")
        print(f"  Passing config(s)  : {', '.join(winners)}")
        print(f"  Best train accuracy: {best_acc:.4f}  ({best})")
        print(f"\n  Interpretation:")
        print(f"  ├─ Dense network can memorise these features → signal exists.")
        print(f"  ├─ High overfit gap is expected and desired here.")
        print(f"  └─ Next: train with regularisation + proper train/val/test split.")
    else:
        print(f"\n  ⚠️  NO CONFIG OVERFIT — features may lack sufficient signal.")
        print(f"  Best train accuracy: {best_acc:.4f}  ({best})")
        print(f"\n  Possible causes:")
        print(f"  ├─ Too few samples (try a larger subset).")
        print(f"  ├─ Features are collinear or uninformative.")
        print(f"  ├─ Target noise too high relative to feature signal.")
        print(f"  └─ Consider adding more indicator periods or raw price features.")

    print(f"\n{'═'*64}\n")

    return {
        "passed":          passed_any,
        "per_config":      results,
        "feature_columns": names,
        "class_counts":    dict(counts),
        "class_weights":   class_weight_dict,
        "input_dim":       n_cols,
    }
