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

from src.models_architecture.no_train_model import NoTrainModel
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


class DummyPreprocessor(PreprocessBase):
    def __init__(self, sequence_length: int):
        super().__init__(sequence_length=sequence_length, name="dummy_preprocessor")

    def preprocess(self, inputs, training: bool = False):
        return inputs

    def features_metadata(self):
        return {}


class TargetPreprocessor(PreprocessBase):
    def __init__(self, sequence_length: int):
        super().__init__(sequence_length=sequence_length, name="target_preprocessor")

    def preprocess(self, inputs, training: bool = False):
        return {
            "features": tf.constant([[0.1, 0.2]], dtype=tf.float32),
            "target": tf.constant([1], dtype=tf.int32),
        }

    def features_metadata(self):
        return {}


def test_no_train_evaluate_supports_non_contiguous_direction_ids():
    model = NoTrainModel(preprocessor=DummyPreprocessor(sequence_length=4), sequence_length=4)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "direction": tf.constant([1, 3], dtype=tf.int32),
            },
            tf.constant([1, 3], dtype=tf.int32),
        )
    ).batch(2)

    metrics = model.evaluate(dataset)

    assert metrics["accuracy"].numpy() == 1.0
    assert metrics["precision_buy"].numpy() == 1.0
    assert metrics["precision_sell"].numpy() == 1.0
    assert metrics["recall_buy"].numpy() == 1.0
    assert metrics["recall_sell"].numpy() == 1.0


def test_no_train_inference_model_extracts_target_values_from_preprocessor_output():
    model = NoTrainModel(preprocessor=TargetPreprocessor(sequence_length=2), sequence_length=2)
    inference_model = model._build_inference()

    inputs = {
        "time": tf.constant([[1, 2]], dtype=tf.int64),
        "open": tf.constant([[1.0, 2.0]], dtype=tf.float32),
        "high": tf.constant([[1.0, 2.0]], dtype=tf.float32),
        "close": tf.constant([[1.0, 2.0]], dtype=tf.float32),
        "low": tf.constant([[1.0, 2.0]], dtype=tf.float32),
        "spread": tf.constant([[1.0, 2.0]], dtype=tf.float32),
        "real_volume": tf.constant([[1.0, 2.0]], dtype=tf.float32),
        "tick_volume": tf.constant([[1.0, 2.0]], dtype=tf.float32),
    }

    predictions = inference_model(inputs)

    assert np.array_equal(predictions.numpy(), np.array([1], dtype=np.int32))
