"""
Feature Importance Map for Trading ML Preprocessing
=====================================================
Single horizontal bar chart ranked best to lowest.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import f_oneway


def compute_feature_importance_map(
    example_data,
    sequence_length: int = 100,
    feature_names: list = None,
):
    """
    Compute and plot feature importance for trading ML preprocessing features.

    Args:
        preprocess_cls:   The Preprocess class (not an instance).
        example_data:     Raw data dict passed to preprocess(data, training=True).
        sequence_length:  Sequence length used to instantiate the preprocessor.
        feature_names:    Optional list of feature names to restrict analysis to.
    """
    features_np = {k: (v.numpy() if hasattr(v, "numpy") else np.array(v))
                   for k, v in example_data.items()}

    target = features_np.pop("target").ravel().astype(int)

    # ── 2. Filter features ─────────────────────────────────────────────────────
    available = list(features_np.keys())
    selected  = [f for f in feature_names if f in available] if feature_names else available

    if not selected:
        raise ValueError("No valid features to analyse.")

    # ── 3. Log class distribution ──────────────────────────────────────────────
    counts = Counter(target.tolist())
    total  = len(target)
    class_labels = {0: "Long (>+10)", 1: "Short (<-10)", 2: "Neutral"}

    print("\n── Target class distribution ──────────────────────────")
    for cls in sorted(class_labels):
        n = counts.get(cls, 0)
        print(f"  Class {cls}  {class_labels.get(cls, str(cls)):<16}  {n:>6}  ({100*n/total:.1f}%)")
    print(f"  Total                          {total:>6}\n")

    # ── 4. Flatten 2-D features ────────────────────────────────────────────────
    flat = {}
    for name in selected:
        arr = features_np[name]
        if arr.ndim == 1:
            flat[name] = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            flat[name] = arr
        else:
            flat[name] = arr.reshape(arr.shape[0], -1)

    # ── 5. Compute importance (ANOVA F-score, mean across columns) ─────────────
    classes = sorted(set(target.tolist()))
    scores  = {}

    for name, arr in flat.items():
        col_scores = []
        for col in range(arr.shape[1]):
            groups = [arr[target == cls, col] for cls in classes if (target == cls).sum() > 1]
            if len(groups) < 2:
                col_scores.append(0.0)
                continue
            try:
                f_stat, _ = f_oneway(*groups)
                col_scores.append(float(f_stat) if not np.isnan(f_stat) else 0.0)
            except Exception:
                col_scores.append(0.0)
        scores[name] = float(np.mean(col_scores))

    # Normalise to [0, 1]
    max_score     = max(scores.values()) or 1.0
    norm          = {k: v / max_score for k, v in scores.items()}
    sorted_names  = sorted(norm, key=norm.get, reverse=True)
    sorted_scores = [norm[n] for n in sorted_names]

    print("── Feature importance (normalised) ────────────────────")
    for n, s in zip(sorted_names, sorted_scores):
        print(f"  {n:<22}  {s:.4f}  {'█' * int(s * 30)}")
    print()

    # ── 6. Plot ────────────────────────────────────────────────────────────────
    n_feats = len(sorted_names)
    fig_h   = max(4, 0.55 * n_feats + 2)

    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, n_feats))
    bars   = ax.barh(sorted_names, sorted_scores, color=colors, edgecolor="none", height=0.6)

    for bar, score in zip(bars, sorted_scores):
        ax.text(score + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", color="white", fontsize=9)

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Normalised Importance (F-score)", color="#aaaaaa", fontsize=10)
    ax.set_title("Feature Importance  ·  Best → Lowest", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#888888", labelsize=9)
    plt.setp(ax.get_yticklabels(), color="white")
    ax.spines[:].set_color("#333344")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    return {"importance_scores": scores, "normalised_scores": norm}