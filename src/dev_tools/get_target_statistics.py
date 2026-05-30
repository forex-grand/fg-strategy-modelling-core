import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import os

def get_target_statistics(file_path):
    features = {
        'target_value': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'target_highest': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'target_lowest': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    }

    split_name = os.path.basename(file_path)
    files = sorted(glob.glob(str(file_path) + f"/{split_name}_*.gz"))
    data = tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
    data = data.map(lambda x: tf.io.parse_example(x, features=features), num_parallel_calls=tf.data.AUTOTUNE)

    # Collect all values into arrays
    target_value_list = []
    target_highest_list = []
    target_lowest_list = []

    for record in data:
        target_value_list.append(record['target_value'].numpy())
        target_highest_list.append(record['target_highest'].numpy())
        target_lowest_list.append(record['target_lowest'].numpy())

    target_value   = np.array(target_value_list)
    target_highest = np.array(target_highest_list)
    target_lowest  = np.array(target_lowest_list)

    # ── Quantile statistics ──────────────────────────────────────────────────
    percentiles = [25, 50, 75, 90, 95]
    stats = {}
    for name, arr in [('target_value', target_value),
                      ('target_highest', target_highest),
                      ('target_lowest', target_lowest)]:
        stats[name] = np.percentile(arr, percentiles)

    # ── Plot 1: Quantile bar chart ───────────────────────────────────────────
    x = np.arange(len(percentiles))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, stats['target_value'],   width, label='target_value',   color='steelblue')
    ax.bar(x,         stats['target_highest'], width, label='target_highest', color='tomato')
    ax.bar(x + width, stats['target_lowest'],  width, label='target_lowest',  color='seagreen')

    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in percentiles])
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Value')
    ax.set_title(f'Quantile Statistics — {split_name}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── Plot 2: highest vs lowest comparison counts ──────────────────────────
    n_highest_gt_lowest = int(np.sum(target_highest > target_lowest))
    n_lowest_gt_highest = int(np.sum(target_lowest  > target_highest))
    n_equal             = len(target_highest) - n_highest_gt_lowest - n_lowest_gt_highest

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['highest > lowest', 'lowest > highest', 'equal']
    counts = [n_highest_gt_lowest, n_lowest_gt_highest, n_equal]
    colors = ['tomato', 'seagreen', 'slategray']

    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Count')
    ax.set_title(f'target_highest vs target_lowest Comparison — {split_name}')
    plt.tight_layout()
    plt.show()

    # ── Return summary dict ──────────────────────────────────────────────────
    return {
        'split': split_name,
        'n_records': len(target_value),
        'quantiles': {name: dict(zip(percentiles, vals.tolist()))
                      for name, vals in stats.items()},
        'highest_gt_lowest': n_highest_gt_lowest,
        'lowest_gt_highest': n_lowest_gt_highest,
        'equal':             n_equal,
    }
