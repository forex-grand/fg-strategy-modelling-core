"""Feature-only training orchestration for KNN/KMeans models."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Type

import pandas as pd
import tensorflow as tf
from keras import Layer

from forexgrand_core.models_architecture.knn_model import KNNModel
from forexgrand_core.pipeline.generate_train_data import GenerateTrainData
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.pipeline.pusher import ModelPusher
from forexgrand_core.schemas import ModelBuildTrainArguments, SymbolIn, TrainingResult
from forexgrand_core.settings import Settings

LOGGER = logging.getLogger(__name__)


class KNNTrainer:
    """Runs unsupervised KNN/KMeans training on feature-only datasets."""

    def __init__(
        self,
        *,
        symbols: list[SymbolIn],
        preprocessor_class: Type[PreprocessBase],
        sequence_length: int,
        hot_reload_data: bool = False,
        upload_models: bool = False,
        use_dataframe_format: bool = False,
    ) -> None:
        self.symbols = symbols
        self.preprocessor_class = preprocessor_class
        self.sequence_length = sequence_length
        self.hot_reload_data = hot_reload_data
        self.upload_models = upload_models
        self.use_dataframe_format = use_dataframe_format
        self.config = Settings()
        self.pusher = ModelPusher(self.config)
        self.data_gen = GenerateTrainData(
            train_base_bucket=self.config.train_bucket_name,
            eval_base_bucket=self.config.eval_bucket_name,
            preprocess_data=True,
            preprocess_layer=preprocessor_class(sequence_length),
            use_dataframe_format=use_dataframe_format,
        )

    def preprocess(self, data, preprocess_layer: Layer):
        return data

    def deserialize(self, data, preprocessor: PreprocessBase):
        return tf.io.parse_example(data, features=preprocessor.features_metadata())

    def get_training_data(self, file_pattern: str, preprocessor: Layer):
        if self.use_dataframe_format:
            split_name = os.path.basename(file_pattern)
            parquet_file = Path(file_pattern) / f"{split_name}_data.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            dtype_mappings = {
                "float32": tf.float32,
                "float64": tf.float32,
                "int32": tf.int64,
                "int64": tf.int64,
            }
            dataset_dict = {
                col: tf.convert_to_tensor(df[col].values, dtype=dtype_mappings[str(df[col].dtype)])
                for col in df.columns
            }
            data = tf.data.Dataset.from_tensor_slices(dataset_dict)
        else:
            split_name = os.path.basename(file_pattern)
            files = sorted(glob.glob(str(file_pattern) + f"/{split_name}_*.gz"))
            data = tf.data.TFRecordDataset(
                files,
                compression_type="GZIP",
                buffer_size=100 * 1024 * 1024,
                num_parallel_reads=tf.data.AUTOTUNE,
            )
            data = data.map(lambda x: self.deserialize(x, preprocessor), num_parallel_calls=tf.data.AUTOTUNE)

        if self.config.shuffle_data:
            data = data.shuffle(self.config.shuffle_buffer_size, reshuffle_each_iteration=True)
        data = data.map(lambda x: self.preprocess(x, preprocess_layer=preprocessor), num_parallel_calls=tf.data.AUTOTUNE)
        data = data.cache()
        data = data.batch(self.config.batch_size, drop_remainder=True)
        return data.prefetch(tf.data.AUTOTUNE)

    def run(self) -> list[TrainingResult]:
        results: list[TrainingResult] = []
        for symbol in self.symbols:
            preprocessor = self.preprocessor_class(sequence_length=self.sequence_length)
            self.data_gen.preprocess_layer = preprocessor
            train_path, eval_path = self.data_gen.load_data(
                symbol_pair=symbol.symbol.strip(),
                instrument_group=symbol.group.strip(),
                sequence_length=self.sequence_length,
                stride=self.config.generated_data_strides,
                hot_reload=self.hot_reload_data,
                target_model=None,
                use_dataframe_format=self.use_dataframe_format,
            )
            train_ds = self.get_training_data(train_path, preprocessor=preprocessor)
            eval_ds = self.get_training_data(eval_path, preprocessor=preprocessor)

            model = KNNModel(sequence_length=self.sequence_length, preprocessor=preprocessor)
            fn_args = ModelBuildTrainArguments(
                learning_rate=self.config.learning_rate,
                epochs=int(os.getenv("KNN_EPOCHS", str(self.config.epochs))),
                callbacks=[],
                steps_per_epoch=self.config.steps_per_epoch,
            )
            model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
            metrics = model.evaluate(eval_ds)
            model_id = None
            if self.upload_models:
                model_id = self.pusher.push(
                    model=model,
                    symbol=symbol.symbol.strip(),
                    model_type="knn",
                    metrics=metrics,
                    sequence_length=self.sequence_length,
                    data_range={
                        "start": self.data_gen.train_properties.data_start,
                        "end": self.data_gen.train_properties.data_end,
                    },
                    benchmark_passed=True,
                    features_keys=list(train_ds.element_spec.keys()),
                )

            result = TrainingResult(
                symbol=symbol.symbol.strip(),
                model_type="knn",
                benchmark_passed=True,
                evaluator_passed=True,
                metrics=metrics,
                model=model,
                model_id=model_id,
            )
            LOGGER.info("KNN result for %s: %s", symbol.symbol, metrics)
            results.append(result)
        return results
