"""
Feature Importance Map for Trading ML Preprocessing
=====================================================
Computes and visualizes feature importance relative to target classes (0, 1, 2).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter


def compute_feature_importance_map(
    preprocess_cls,
    example_data,
    sequence_length: int = 100,
    feature_names: list = None,
):
    """
    Compute and plot feature importance for trading ML preprocessing features.

    Args:
        preprocess_cls:     The Preprocess class (not an instance).
        example_data:       Raw data dict passed to preprocess(data, training=True).
        sequence_length:    Sequence length used to instantiate the preprocessor.
        feature_names:      Optional list of feature names to restrict analysis to.
                            If None, all features (except 'target') are used.
    """

    # ── 1. Instantiate preprocessor & transform data ──────────────────────────
    preprocessor = preprocess_cls(sequence_length=sequence_length)
    transformed  = preprocessor.preprocess(example_data, training=True)

    # Convert every tensor to a numpy array
    features_np = {}
    for k, v in transformed.items():
        arr = v.numpy() if hasattr(v, "numpy") else np.array(v)
        features_np[k] = arr

    target = features_np.pop("target")          # shape (batch,)
    target = np.asarray(target).ravel().astype(int)

    # ── 2. Filter to requested features ───────────────────────────────────────
    available = list(features_np.keys())
    if feature_names is not None:
        missing = [f for f in feature_names if f not in available]
        if missing:
            print(f"[Warning] Requested features not found and will be skipped: {missing}")
        selected = [f for f in feature_names if f in available]
    else:
        selected = available

    if not selected:
        raise ValueError("No valid features to analyse.")

    # ── 3. Log target-class distribution ──────────────────────────────────────
    class_labels = {0: "Long (>+10)", 1: "Short (<-10)", 2: "Neutral"}
    counts = Counter(target.tolist())
    total  = len(target)

    print("\n── Target class distribution ──────────────────────────────────")
    for cls in sorted(class_labels):
        n   = counts.get(cls, 0)
        pct = 100.0 * n / total if total else 0
        print(f"  Class {cls}  {class_labels[cls]:<16}  {n:>6} samples  ({pct:.1f}%)")
    print(f"  Total                          {total:>6} samples")
    print()

    # ── 4. Flatten 2-D features to 1-D per sample ─────────────────────────────
    # Feature tensor shapes:
    #   1-D per sample:  (batch,)
    #   2-D per sample:  (batch, length)  → flatten to (batch, length)
    # We keep the multi-column version and compute per-column importance,
    # then aggregate (mean) across columns for a single importance score.

    flat_features = {}
    for name in selected:
        arr = features_np[name]
        if arr.ndim == 1:
            flat_features[name] = arr.reshape(-1, 1)       # (batch, 1)
        elif arr.ndim == 2:
            flat_features[name] = arr                      # (batch, length)
        else:
            # Higher-rank tensors: reshape to (batch, -1)
            flat_features[name] = arr.reshape(arr.shape[0], -1)
            print(f"[Info] Feature '{name}' had shape {arr.shape}, "
                  f"reshaped to {flat_features[name].shape}")

    # ── 5. Compute importance scores ──────────────────────────────────────────
    # Method: mutual-information-like class separability per feature column.
    # We use the ANOVA F-score (fast, no extra dependencies) per column and
    # aggregate with the mean.  Results are normalised to [0, 1].

    from scipy.stats import f_oneway

    importance_scores   = {}   # {feature_name: scalar score}
    per_column_scores   = {}   # {feature_name: array of scores per column}

    classes = sorted(set(target.tolist()))

    for name, arr in flat_features.items():
        col_scores = []
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            groups = [col[target == cls] for cls in classes]
            # Filter out empty groups
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                col_scores.append(0.0)
                continue
            try:
                f_stat, _ = f_oneway(*groups)
                col_scores.append(float(f_stat) if not np.isnan(f_stat) else 0.0)
            except Exception:
                col_scores.append(0.0)

        per_column_scores[name] = np.array(col_scores)
        importance_scores[name] = float(np.mean(col_scores))

    # Normalise to [0, 1]
    max_score = max(importance_scores.values()) or 1.0
    norm_scores = {k: v / max_score for k, v in importance_scores.items()}

    # Sorted for bar chart
    sorted_names  = sorted(norm_scores, key=norm_scores.get, reverse=True)
    sorted_scores = [norm_scores[n] for n in sorted_names]

    print("── Feature importance (normalised F-score) ─────────────────────")
    for n, s in zip(sorted_names, sorted_scores):
        bar = "█" * int(s * 30)
        print(f"  {n:<20}  {s:.4f}  {bar}")
    print()

    # ── 6. Class-conditional mean per feature ─────────────────────────────────
    class_means = {}   # {feature_name: {cls: mean_value}}
    for name, arr in flat_features.items():
        class_means[name] = {}
        for cls in classes:
            mask = target == cls
            if mask.sum() == 0:
                class_means[name][cls] = 0.0
            else:
                class_means[name][cls] = float(arr[mask].mean())

    # ── 7. Plot ────────────────────────────────────────────────────────────────
    n_feats = len(selected)
    palette = ["#2ecc71", "#e74c3c", "#3498db"]   # green / red / blue → Long / Short / Neutral

    fig = plt.figure(figsize=(14, 4 + 3 * n_feats), facecolor="#0f1117")
    fig.suptitle(
        "Feature Importance Map  ·  Trading ML",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    rows = 2 + n_feats
    gs   = gridspec.GridSpec(rows, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax_style = dict(facecolor="#1a1d27", frameon=True)
    label_kw = dict(color="#aaaaaa", fontsize=9)
    tick_kw  = dict(colors="#888888", labelsize=8)

    # ── 7a. Horizontal bar chart – overall importance ──────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.set_facecolor("#1a1d27")
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_names)))
    bars = ax_bar.barh(sorted_names, sorted_scores, color=colors_bar, edgecolor="none", height=0.55)
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Normalised Importance (F-score)", **label_kw)
    ax_bar.set_title("Overall Feature Importance", color="white", fontsize=11, pad=8)
    ax_bar.tick_params(axis="both", **tick_kw)
    ax_bar.spines[:].set_color("#333344")
    for bar, score in zip(bars, sorted_scores):
        ax_bar.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", color="white", fontsize=8)
    ax_bar.yaxis.label.set_color("#aaaaaa")
    plt.setp(ax_bar.get_yticklabels(), color="white")

    # ── 7b. Target class distribution ─────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor("#1a1d27")
    pie_vals   = [counts.get(c, 0) for c in classes]
    pie_labels = [f"{class_labels[c]}\n({counts.get(c,0)})" for c in classes]
    ax_pie.pie(pie_vals, labels=pie_labels, colors=palette[:len(classes)],
               autopct="%1.1f%%", textprops={"color": "white", "fontsize": 8},
               wedgeprops={"edgecolor": "#0f1117", "linewidth": 1.5})
    ax_pie.set_title("Target Distribution", color="white", fontsize=10)

    # ── 7c. Class-conditional means heatmap ───────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.set_facecolor("#1a1d27")
    heat_data = np.array([[class_means[n].get(c, 0) for c in classes] for n in selected])
    # Z-score normalise rows for display
    row_std  = heat_data.std(axis=1, keepdims=True) + 1e-8
    row_mean = heat_data.mean(axis=1, keepdims=True)
    heat_z   = (heat_data - row_mean) / row_std

    sns.heatmap(
        heat_z,
        ax=ax_heat,
        xticklabels=[class_labels[c] for c in classes],
        yticklabels=selected,
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="#0f1117",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7, "color": "white"},
        cbar_kws={"shrink": 0.8},
    )
    ax_heat.set_title("Class-Conditional Mean (z-scored)", color="white", fontsize=10)
    plt.setp(ax_heat.get_xticklabels(), color="white", fontsize=7)
    plt.setp(ax_heat.get_yticklabels(), color="white", fontsize=7)
    ax_heat.tick_params(axis="both", colors="#888888")

    # ── 7d. Per-feature: distribution by class (one row per feature) ──────────
    for row_idx, name in enumerate(selected):
        ax_l = fig.add_subplot(gs[2 + row_idx, 0])
        ax_r = fig.add_subplot(gs[2 + row_idx, 1])

        arr = flat_features[name]   # (batch, cols)

        # Left: KDE / violin of mean value per sample, coloured by class
        sample_means = arr.mean(axis=1)
        for i, cls in enumerate(classes):
            vals = sample_means[target == cls]
            if len(vals) > 1:
                sns.kdeplot(vals, ax=ax_l, color=palette[i % len(palette)],
                            fill=True, alpha=0.35, linewidth=1.2,
                            label=class_labels[cls])
        ax_l.set_facecolor("#1a1d27")
        ax_l.set_title(f"{name}  ·  sample mean KDE", color="white", fontsize=9)
        ax_l.tick_params(axis="both", **tick_kw)
        ax_l.spines[:].set_color("#333344")
        ax_l.yaxis.label.set_color("#aaaaaa")
        ax_l.xaxis.label.set_color("#aaaaaa")
        ax_l.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white",
                    edgecolor="#333344", loc="upper right")

        # Right: per-column importance across the feature's time dimension
        col_s = per_column_scores[name]
        ax_r.set_facecolor("#1a1d27")
        if len(col_s) > 1:
            xs = np.arange(len(col_s))
            ax_r.fill_between(xs, col_s, alpha=0.4, color="#3498db")
            ax_r.plot(xs, col_s, color="#3498db", linewidth=1.5)
            ax_r.set_xlabel("Time step (column index)", **label_kw)
            ax_r.set_ylabel("F-score", **label_kw)
            ax_r.set_title(f"{name}  ·  per-step importance", color="white", fontsize=9)
        else:
            ax_r.bar([name], col_s, color="#3498db", edgecolor="none")
            ax_r.set_title(f"{name}  ·  importance", color="white", fontsize=9)
        ax_r.tick_params(axis="both", **tick_kw)
        ax_r.spines[:].set_color("#333344")
        ax_r.yaxis.label.set_color("#aaaaaa")
        ax_r.xaxis.label.set_color("#aaaaaa")

    plt.savefig("feature_importance_map.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print("[✓] Plot saved to feature_importance_map.png")

    return {
        "importance_scores": importance_scores,
        "normalised_scores": norm_scores,
        "class_counts":      dict(counts),
        "class_means":       class_means,
    }