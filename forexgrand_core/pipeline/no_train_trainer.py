"""Dedicated training orchestration for no-train direction-based models."""

from __future__ import annotations

import logging
import os
from typing import Type

import tensorflow as tf

from forexgrand_core.models_architecture.no_train_model import NoTrainModel
from forexgrand_core.pipeline.evaluator import Evaluator
from forexgrand_core.pipeline.generate_train_data import GenerateTrainData
from forexgrand_core.pipeline.performance_test import test_model_live_performance
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.pipeline.pusher import ModelPusher
from forexgrand_core.pipeline.statistics_gen import get_target_statistics
from forexgrand_core.schemas import ModelBuildTrainArguments, SymbolIn, TARGET_MODEL_TYPES, TrainingResult
from forexgrand_core.settings import Settings

LOGGER = logging.getLogger(__name__)


class NoTrainTrainer:
    def __init__(
        self,
        *,
        symbols: list[SymbolIn],
        preprocessor_class: Type[PreprocessBase],
        sequence_length: int,
        target_model_type: TARGET_MODEL_TYPES,
        hot_reload_data: bool = False,
        run_performance_test: bool = False,
        upload_models: bool = False,
        target_percentile: int = 95,
        use_dataframe_format: bool = False,
    ) -> None:
        self.symbols = symbols
        self.preprocessor_class = preprocessor_class
        self.sequence_length = sequence_length
        self.target_model_type = target_model_type
        self.hot_reload_data = hot_reload_data
        self.run_performance_test = run_performance_test
        self.upload_models = upload_models
        self.target_percentile = target_percentile
        self.use_dataframe_format = use_dataframe_format
        self.config = Settings()
        self.preprocessor = preprocessor_class(sequence_length)
        self.data_gen = GenerateTrainData(
            train_base_bucket=self.config.train_bucket_name,
            eval_base_bucket=self.config.eval_bucket_name,
            preprocess_data=True,
            preprocess_layer=self.preprocessor,
            use_dataframe_format=use_dataframe_format,
        )
        self.evaluator = Evaluator(self.config)
        self.pusher = ModelPusher(self.config)

    def deserialize(self, data, preprocessor: PreprocessBase):
        try:
            return tf.io.parse_example(data, features=preprocessor.features_metadata())
        except Exception:
            return tf.io.parse_example(
                data,
                features={
                    "time": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.int64),
                    "open": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                    "high": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                    "close": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                    "low": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                    "spread": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                    "real_volume": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                    "tick_volume": tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
                },
            )

    def get_training_data(self, file_pattern: str, preprocessor: PreprocessBase):
        from pathlib import Path
        import glob
        import pandas as pd
        import tensorflow as tf

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

        data = data.map(lambda x: self._preprocess_dataset(x, preprocessor=preprocessor), num_parallel_calls=tf.data.AUTOTUNE)
        data = data.cache()
        data = data.batch(self.config.batch_size, drop_remainder=True)
        data = data.prefetch(tf.data.AUTOTUNE)
        return data

    def _preprocess_dataset(self, data, preprocessor: PreprocessBase):
        if isinstance(data, tuple) and len(data) == 2:
            return data

        if isinstance(data, dict):
            if "direction" in data:
                return data, data["direction"]
            if "target" in data:
                return data, data["target"]

        return data, None

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
                target_model=self.target_model_type,
                use_dataframe_format=self.use_dataframe_format,
            )

            target_data = self.data_gen.load_target_data(
                bucket_name=self.config.train_bucket_name,
                symbol_pair=symbol.symbol.strip(),
                instrument_group=symbol.group.strip(),
                sequence_length=self.sequence_length,
                stride=self.config.generated_data_strides,
                target_model=self.target_model_type,
            )
            statistics = get_target_statistics(target_data, percentiles=[self.target_percentile])
            mean_target_min_value = int(
                (statistics["quantiles"]["target_highest"][self.target_percentile] + statistics["quantiles"]["target_lowest"][self.target_percentile]) / 2
            )
            preprocessor.min_target_points = mean_target_min_value

            train_ds = self.get_training_data(train_path, preprocessor=preprocessor)
            eval_ds = self.get_training_data(eval_path, preprocessor=preprocessor)

            model = NoTrainModel(sequence_length=self.sequence_length, preprocessor=preprocessor)
            fn_args = ModelBuildTrainArguments(
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                callbacks=[],
                steps_per_epoch=self.config.steps_per_epoch,
            )

            model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
            metrics = model.evaluate(eval_ds)

            evaluator_passed, reasons = self.evaluator.evaluate(metrics)
            model_id = None
            if self.upload_models:
                model_id = self.pusher.push(
                    model=model,
                    symbol=symbol.symbol.strip(),
                    model_type="no-train",
                    metrics=metrics,
                    sequence_length=self.sequence_length,
                    data_range={
                        "start": self.data_gen.train_properties.data_start,
                        "end": self.data_gen.train_properties.data_end,
                    },
                    benchmark_passed=True,
                    features_keys=list(train_ds.element_spec[0].keys()),
                )

            if self.run_performance_test:
                test_model_live_performance(
                    model,
                    symbol.symbol.strip(),
                    symbol.group.strip(),
                    self.sequence_length,
                    self.config,
                    model_id,
                    stride=self.config.test_generator_stride,
                    eval_metrics=reasons,
                    min_target_points=mean_target_min_value,
                    feature_keys=train_ds.element_spec[0],
                )

            result = TrainingResult(
                symbol=symbol.symbol.strip(),
                model_type="no-train",
                benchmark_passed=True,
                evaluator_passed=evaluator_passed,
                metrics=metrics,
                model=model,
                model_id=model_id,
            )
            LOGGER.info("No-train result for %s: %s", symbol.symbol, metrics)
            results.append(result)

        return results
