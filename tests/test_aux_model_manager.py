import os

os.environ.setdefault("DATA_SOURCE", "mt5")
os.environ.setdefault("S3_STORAGE_OPTION", "minio")
os.environ.setdefault("S3_ENDPOINT", "http://localhost")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")
os.environ.setdefault("S3_BUCKET_NAME", "bucket")
os.environ.setdefault("TRAIN_BUCKET_NAME", "train")
os.environ.setdefault("EVAL_BUCKET_NAME", "eval")

import numpy as np
import tensorflow as tf

from src.aux_model_manager import AuxilaryModelManager


class DummyManager(AuxilaryModelManager):
    def __init__(self):
        self.model = {"metadata": {"sequence_length": 3}}


def test_ohlc_to_feature_dict_uses_direct_feature_dictionary():
    manager = DummyManager()
    row_dict = {
        "time": tf.constant([[1, 2, 3, 4]], dtype=tf.int64),
        "open": tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32),
        "high": tf.constant([[2.0, 3.0, 4.0, 5.0]], dtype=tf.float32),
        "close": tf.constant([[3.0, 4.0, 5.0, 6.0]], dtype=tf.float32),
        "low": tf.constant([[0.0, 1.0, 2.0, 3.0]], dtype=tf.float32),
        "spread": tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32),
        "tick_volume": tf.constant([[10, 20, 30, 40]], dtype=tf.float32),
        "real_volume": tf.constant([[100, 200, 300, 400]], dtype=tf.float32),
    }

    feature_dict = manager._ohlc_to_feature_dict(row_dict)

    assert set(feature_dict.keys()) == {"time", "open", "high", "close", "low", "spread", "tick_volume", "real_volume"}
    assert feature_dict["time"].shape == (1, 3)
    assert feature_dict["close"].shape == (1, 3)
    assert np.array_equal(feature_dict["close"].numpy(), np.array([[4.0, 5.0, 6.0]], dtype=np.float32))
