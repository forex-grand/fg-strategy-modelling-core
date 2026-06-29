import os

os.environ.setdefault("DATA_SOURCE", "mt5")
os.environ.setdefault("S3_STORAGE_OPTION", "minio")
os.environ.setdefault("S3_ENDPOINT", "http://localhost")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")
os.environ.setdefault("S3_BUCKET_NAME", "bucket")
os.environ.setdefault("TRAIN_BUCKET_NAME", "train")
os.environ.setdefault("EVAL_BUCKET_NAME", "eval")

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
