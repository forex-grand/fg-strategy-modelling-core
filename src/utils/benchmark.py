"""Training eligibility checks for sequence datasets."""

from __future__ import annotations

import logging

import tensorflow as tf

from src.settings import Settings

LOGGER = logging.getLogger(__name__)


class BenchmarkError(RuntimeError):
    """Raised when the training dataset fails benchmark quality gates."""


def validate_dataset_eligibility(dataset: tf.data.Dataset, config: Settings) -> dict[str, float]:
    """
    Validate dataset before model training begins.

    Required checks:
    - buy + sell signals > 400
    - class imbalance ratio < 4:1
    - sequence count > 1000

    Additional checks:
    - label diversity (both classes present)
    - non-degenerate features (finite values and non-zero variance)
    """
    sequence_count = 0
    buy_count = 0
    sell_count = 0
    sum_features = 0.0
    sumsq_features = 0.0
    feature_count = 0

    for batch_x, batch_y in dataset:
        labels = tf.reshape(batch_y, (-1,))
        labels_int = tf.cast(tf.round(labels), tf.int32)
        buy_count += int(tf.reduce_sum(tf.cast(labels_int == 1, tf.int32)).numpy())
        sell_count += int(tf.reduce_sum(tf.cast(labels_int == 0, tf.int32)).numpy())
        sequence_count += int(labels.shape[0])

        flat_features = tf.reshape(tf.cast(batch_x, tf.float32), (-1,))
        if not tf.reduce_all(tf.math.is_finite(flat_features)):
            raise BenchmarkError("Features contain non-finite values.")
        sum_features += float(tf.reduce_sum(flat_features).numpy())
        sumsq_features += float(tf.reduce_sum(tf.square(flat_features)).numpy())
        feature_count += int(flat_features.shape[0])

    total_signals = buy_count + sell_count
    if total_signals <= config.benchmark_min_signals:
        raise BenchmarkError(
            f"Insufficient signals: {total_signals} <= {config.benchmark_min_signals}."
        )
    if sequence_count <= config.benchmark_min_sequences:
        raise BenchmarkError(
            f"Insufficient sequences: {sequence_count} <= {config.benchmark_min_sequences}."
        )
    if buy_count == 0 or sell_count == 0:
        raise BenchmarkError("Label diversity check failed: one class is missing.")

    imbalance_ratio = max(buy_count, sell_count) / max(1, min(buy_count, sell_count))
    if imbalance_ratio >= config.benchmark_max_imbalance_ratio:
        raise BenchmarkError(
            f"Class imbalance too high: {imbalance_ratio:.3f} >= {config.benchmark_max_imbalance_ratio:.3f}."
        )

    mean_feature = sum_features / max(feature_count, 1)
    variance = (sumsq_features / max(feature_count, 1)) - (mean_feature**2)
    if variance <= 1e-10:
        raise BenchmarkError("Feature variance is near zero; dataset appears degenerate.")

    stats = {
        "sequence_count": float(sequence_count),
        "buy_count": float(buy_count),
        "sell_count": float(sell_count),
        "imbalance_ratio": float(imbalance_ratio),
        "feature_variance": float(variance),
    }
    LOGGER.info("Benchmark checks passed: %s", stats)
    return stats

