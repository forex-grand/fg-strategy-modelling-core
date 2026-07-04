"""Dedicated training orchestration for Optuna-tuned XGBoost models."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional, Type

import pandas as pd
import tensorflow as tf
from keras import Layer

from src.models_architecture.base_xgb_model import XGBTrainModel
from src.pipeline.evaluator import Evaluator
from src.pipeline.generate_train_data import GenerateTrainData
from src.pipeline.performance_test import test_model_live_performance
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase
from src.pipeline.pusher import ModelPusher
from src.pipeline.statistics_gen import get_target_statistics
from src.schemas import SymbolIn, TARGET_MODEL_TYPES, TrainingResult
from src.settings import Settings

LOGGER = logging.getLogger(__name__)


class XGBTrainer:
    """Runs one Optuna-tuned XGBoost training job per symbol."""

    MODEL_TYPE = "xgb-optuna"

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
        preprocess_at_data_generation = True,
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
        self.preprocess_at_datagen = preprocess_at_data_generation
        self.config = Settings()
        self.preprocessor = preprocessor_class(sequence_length)
        self.data_gen = GenerateTrainData(
            train_base_bucket=self.config.train_bucket_name,
            eval_base_bucket=self.config.eval_bucket_name,
            preprocess_data=preprocess_at_data_generation,
            preprocess_layer=self.preprocessor,
            use_dataframe_format=use_dataframe_format,
        )
        self.evaluator = Evaluator(self.config)
        self.pusher = ModelPusher(self.config)

    def deserialize(self, data, preprocessor: PreprocessBase):
        return tf.io.parse_example(data, features=preprocessor.features_metadata())

    def preprocess(self, data, preprocess_layer: Layer):
        if not self.preprocess_at_datagen:
            data = preprocess_layer(data, training=True)
        target = data.pop("target")
        return data, target

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
       
        data = data.batch(self.config.batch_size, drop_remainder=True)
        data = data.map(
            lambda x: self.preprocess(x, preprocess_layer=preprocessor),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        data = data.cache()
        return data.prefetch(tf.data.AUTOTUNE)

    def run(self) -> list[TrainingResult]:
        results: list[TrainingResult] = []
        for symbol in self.symbols:
            result = self._run_symbol(symbol)
            if result is not None:
                results.append(result)
        return results

    def _run_symbol(self, symbol: SymbolIn) -> Optional[TrainingResult]:
        preprocessor = self.preprocessor_class(sequence_length=self.sequence_length)
        use_aux_model = symbol.aux_model_id is not None
        self.data_gen = GenerateTrainData(
            train_base_bucket=self.config.train_bucket_name,
            eval_base_bucket=self.config.eval_bucket_name,
            preprocess_data=self.preprocess_at_datagen,
            preprocess_layer=preprocessor,
            use_dataframe_format=self.use_dataframe_format,
            filter_by_model=use_aux_model,
            filter_model_id=symbol.aux_model_id,
            target_label=symbol.aux_target_label,
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
            (
                statistics["quantiles"]["target_highest"][self.target_percentile]
                + statistics["quantiles"]["target_lowest"][self.target_percentile]
            )
            / 2
        )
        preprocessor.min_target_points = mean_target_min_value
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
        train_ds = self.get_training_data(train_path, preprocessor=preprocessor)
        eval_ds = self.get_training_data(eval_path, preprocessor=preprocessor)
        model = XGBTrainModel(sequence_length=self.sequence_length, preprocessor=preprocessor)
        metric_values: dict[str, float] = {}
        reasons = {}

        try:
            fn_args = {
                "symbol": symbol.symbol.strip(),
                "group": symbol.group.strip(),
                "train_ds_keys": train_ds.element_spec[0].keys(),
                "min_target_point": mean_target_min_value,
                "data_start": self.data_gen.train_properties.data_start,
                "data_end": self.data_gen.train_properties.data_end,
                "steps_per_epoch": self.config.steps_per_epoch,
            }
            model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
            metric_values = model.evaluate(eval_ds)
            evaluator_passed, reasons = self.evaluator.evaluate(metric_values)
        except Exception as error:
            LOGGER.warning("Error training XGBoost model for %s: %s", symbol.symbol, error)
            return TrainingResult(
                symbol=symbol.symbol.strip(),
                model_type=self.MODEL_TYPE,
                benchmark_passed=False,
                evaluator_passed=False,
                metrics=metric_values,
                model=model,
            )

        if not evaluator_passed:
            LOGGER.warning("Evaluator rejected XGBoost model for %s", symbol.symbol)
            LOGGER.warning("Failure Reasons: %s", reasons)
            return TrainingResult(
                symbol=symbol.symbol.strip(),
                model_type=self.MODEL_TYPE,
                benchmark_passed=True,
                evaluator_passed=False,
                metrics=metric_values,
                model=model,
            )

        model_id = None
        if self.upload_models:
            model_id = self.pusher.push(
                model=model,
                symbol=symbol.symbol.strip(),
                model_type=self.MODEL_TYPE,
                metrics=metric_values,
                sequence_length=self.sequence_length,
                data_range={
                    "start": self.data_gen.train_properties.data_start,
                    "end": self.data_gen.train_properties.data_end,
                },
                benchmark_passed=True,
                features_keys=list(train_ds.element_spec[0].keys()),
            )

        if self.run_performance_test:
            try:
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
            except Exception as error:
                LOGGER.warning("Optional fg-tester call failed for %s/xgb: %s", symbol.symbol, error)

        return TrainingResult(
            symbol=symbol.symbol.strip(),
            model_type=self.MODEL_TYPE,
            benchmark_passed=True,
            evaluator_passed=True,
            metrics=metric_values,
            model=model,
            model_id=model_id,
        )
