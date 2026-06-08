"""Training orchestration for ForexGrand models."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Type, List
from pathlib import Path
import asyncio
import pandas as pd
import tensorflow as tf
from keras import callbacks, metrics, optimizers, Layer
import glob # Import glob
import os # Import os

from src.models_architecture.train_models.complex_model import ComplexNSTrainModel
from src.models_architecture.train_models.cnn_bi_lstm import CNNBiLSTMModel
from src.models_architecture.train_models.conservative_model import ConservativeNSTrainModel
from src.models_architecture.train_models.lstm_model import LSTMModel
from src.settings import Settings
from src.pipeline.evaluator import Evaluator
from src.pipeline.generate_train_data import GenerateTrainData
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase
from src.pipeline.statistics_gen import get_target_statistics
from src.pipeline.pusher import ModelPusher
from src.models_architecture.base_model import BaseModel
from src.models_architecture.no_train_model import NoTrainModel
from src.models_architecture.train_models.simple_model import SimpleNSTrainModel
from src.models_architecture.train_models.xgb_train_models.xgb_simple import XGBSimple
from src.models_architecture.train_models.xgb_train_models.xgb_tiny             import XGBTiny
from src.models_architecture.train_models.xgb_train_models.xgb_simple_shallow   import XGBSimpleShallow
from src.models_architecture.train_models.xgb_train_models.xgb_simple_slow      import XGBSimpleSlow
from src.models_architecture.train_models.xgb_train_models.xgb_balanced         import XGBBalanced
from src.models_architecture.train_models.xgb_train_models.xgb_l1_regularised   import XGBL1Regularised
from src.models_architecture.train_models.xgb_train_models.xgb_l2_regularised   import XGBL2Regularised
from src.models_architecture.train_models.xgb_train_models.xgb_gamma_pruned     import XGBGammaPruned
from src.models_architecture.train_models.xgb_train_models.xgb_column_sampled   import XGBColumnSampled
from src.models_architecture.train_models.xgb_train_models.xgb_deep_trees       import XGBDeepTrees
from src.models_architecture.train_models.xgb_train_models.xgb_high_capacity    import XGBHighCapacity
from src.models_architecture.train_models.xgb_train_models.xgb_elastic_net      import XGBElasticNet
from src.models_architecture.train_models.xgb_train_models.xgb_high_child_weight import XGBHighChildWeight
from src.models_architecture.train_models.xgb_train_models.xgb_max_complex      import XGBMaxComplex
 
from src.pipeline.performance_test import test_model_live_performance
from src.schemas import SymbolIn, TARGET_MODEL_TYPES, ModelBuildTrainArguments, EpochMetricsLogger, TrainingResult
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

CLASS_IDS = [0, 1]
class Trainer:
    """Runs training for every (symbol, model_type) combination."""

    MODEL_REGISTRY = {
        "no-train": NoTrainModel,
        "simple-ns": SimpleNSTrainModel,
        "conservative-ns": ConservativeNSTrainModel,
        "complex-ns": ComplexNSTrainModel,
        "lstm": LSTMModel,
        "cnn-bi-lstm": CNNBiLSTMModel,
        'xgb-simple':XGBSimple,
        'xgb-tiny'           : XGBTiny,
        'xgb-simple-shallow' : XGBSimpleShallow,
        'xgb-simple-slow'    : XGBSimpleSlow,
        # ── Tier 2 · Moderate ────────────────────────────
        'xgb-balanced'       : XGBBalanced,
        'xgb-l1-regularised' : XGBL1Regularised,
        'xgb-l2-regularised' : XGBL2Regularised,
        'xgb-gamma-pruned'   : XGBGammaPruned,
        # ── Tier 3 · Advanced ────────────────────────────
        'xgb-column-sampled' : XGBColumnSampled,
        'xgb-deep-trees'     : XGBDeepTrees,
        'xgb-high-capacity'  : XGBHighCapacity,
        # ── Tier 4 · Complex ─────────────────────────────
        'xgb-elastic-net'    : XGBElasticNet,
        'xgb-high-child-weight': XGBHighChildWeight,
        'xgb-max-complex'    : XGBMaxComplex,
    }

    def __init__(
        self,
        *,
        symbols: List[SymbolIn],
        model_types: list[str],
        preprocessor_class: Type[PreprocessBase],
        sequence_length: int,
        target_model_type: TARGET_MODEL_TYPES,
        hot_reload_data: bool = False,
        run_performance_test: bool = False,
        upload_models: bool = False,
        target_percentile: int = 95,
        use_dataframe_format: bool = False,
    ) -> None:
        self.symbols: List[SymbolIn] = symbols
        self.model_types = [item.strip().lower() for item in model_types]
        self.preprocessor_class = preprocessor_class
        self.target_model_type = target_model_type
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
        self.sequence_length = sequence_length
        self.hot_reload_data = hot_reload_data
        self.run_performance_test = run_performance_test
        self.upload_models = upload_models
        self.use_dataframe_format = use_dataframe_format
        self.train_ds = {}
        self.eval_ds  = {}
        self.target_percentile = target_percentile


    def run(self) -> list[TrainingResult]:
        """Execute training for all requested model types."""
        results: list[TrainingResult] = []
        for symbol in self.symbols:
            preprocessor = self.preprocessor_class(sequence_length=self.sequence_length)
            target_data = self.data_gen.load_target_data(
                bucket_name=self.config.train_bucket_name, 
                symbol_pair=symbol.symbol.strip(),
                instrument_group=symbol.group.strip(),
                sequence_length=self.sequence_length,
                stride=self.config.generated_data_strides,
                target_model=self.target_model_type,
                )
            statistics = get_target_statistics(target_data, percentiles=[self.target_percentile])
            mean_target_min_value = int((statistics['quantiles']['target_highest'][self.target_percentile] + statistics['quantiles']['target_lowest'][self.target_percentile])/2)
            LOGGER.info(f"Min Target Point: {mean_target_min_value}")
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
            data_start = self.data_gen.train_properties.data_start
            data_end = self.data_gen.train_properties.data_end
            
            symbol_string = symbol.symbol.strip()
            
            train_ds = self.get_training_data(train_path, preprocessor=preprocessor)
            eval_ds  = self.get_training_data(eval_path, preprocessor=preprocessor)
            for model_type in self.model_types:
                LOGGER.info("+++++++==================================================================++++++++++++")
                LOGGER.info("+++++++==================================================================++++++++++++")
                LOGGER.info("+++++++==================================================================++++++++++++")
                LOGGER.info("+++++++==================================================================++++++++++++")
                LOGGER.info(f"========RUNNING TEST FOR {symbol}:{model_type}==========")
                
                results.append(self._run_single(preprocessor=preprocessor,
                                                symbol=symbol.symbol.strip(),
                                                group=symbol.group.strip(),
                                                model_type=model_type,
                                                train_ds=train_ds,
                                                eval_ds=eval_ds,
                                                data_start=data_start,
                                                data_end=data_end,
                                                min_target_point=mean_target_min_value,
                                                ),)
        
        # ── Results Summary ──────────────────────────────────────────────
        passed = [r for r in results if r is not None and r.evaluator_passed]
        failed = [r for r in results if r is not None and not r.evaluator_passed]

        # Sort passed models by precision_buy, precision_sell, recall_buy, recall_sell (desc)
        passed_sorted = sorted(
            passed,
            key=lambda r: (
                r.metrics.get("precision_buy",  0.0),
                r.metrics.get("precision_sell", 0.0),
                r.metrics.get("recall_buy",     0.0),
                r.metrics.get("recall_sell",    0.0),
            ),
            reverse=True,
        )

        LOGGER.info("=" * 80)
        LOGGER.info("TRAINING SUMMARY — PASSED MODELS (ranked by precision/recall)")
        LOGGER.info("=" * 80)
        for rank, r in enumerate(passed_sorted, start=1):
            LOGGER.info(
                "[#%02d] %s:%s | prec_buy=%.4f  prec_sell=%.4f  rec_buy=%.4f  rec_sell=%.4f",
                rank,
                r.symbol,
                r.model_type,
                r.metrics.get("precision_buy",  0.0),
                r.metrics.get("precision_sell", 0.0),
                r.metrics.get("recall_buy",     0.0),
                r.metrics.get("recall_sell",    0.0),
            )

        LOGGER.info("-" * 80)
        LOGGER.info("FAILED MODELS (%d total)", len(failed))
        LOGGER.info("-" * 80)
        for r in failed:
            LOGGER.info(
                "[FAIL] %s:%s | prec_buy=%.4f  prec_sell=%.4f  rec_buy=%.4f  rec_sell=%.4f",
                r.symbol,
                r.model_type,
                r.metrics.get("precision_buy",  0.0),
                r.metrics.get("precision_sell", 0.0),
                r.metrics.get("recall_buy",     0.0),
                r.metrics.get("recall_sell",    0.0),
            )
        LOGGER.info("=" * 80)
                
        return results

    def _run_single(self, preprocessor, symbol,group, model_type: str, train_ds: tf.data, eval_ds: tf.data,
                    data_start: str, data_end: str, min_target_point:int) -> TrainingResult:
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type '{model_type}'.")

        sequence_length = int(self.sequence_length)
        model_class = self.MODEL_REGISTRY[model_type]
        model: BaseModel = model_class(sequence_length=sequence_length, preprocessor=preprocessor)
        model_obj = None
        metric_values = {}
        _reason_map = {}
        if model_type=="no-train":
            fn_args = ModelBuildTrainArguments(
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            callbacks=[],
            steps_per_epoch=self.config.steps_per_epoch,
            )
            model_obj = model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
            metric_values = model.evaluate(eval_ds)
            evaluator_passed, _reason_map = self.evaluator.evaluate(metric_values)
            if not evaluator_passed:
                LOGGER.warning("Evaluator rejected model for %s/%s", symbol, model_type)
                LOGGER.warning(f"Failure Reasons: {_reason_map}")
                return TrainingResult(
                    symbol=symbol,
                    model_type=model_type,
                    benchmark_passed=True,
                    evaluator_passed=False,
                    metrics=metric_values,
                    model=model,
                )
        elif re.match(r"^xgb", model_type):
            try:
                one_d = train_ds.element_spec[0]
                fn_args = {
                  'symbol':symbol,
                  'group':group,
                  'train_ds_keys':one_d.keys(),
                  "min_target_point":min_target_point,
                  "data_start":data_start,
                  "data_end":data_end,
                }
                model_obj = model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
                metric_values = model.evaluate(eval_ds)
                evaluator_passed, _reason_map = self.evaluator.evaluate(metric_values)            
                if not evaluator_passed:
                    LOGGER.warning("Evaluator rejected model for %s/%s", symbol, model_type)
                    LOGGER.warning(f"Failure Reasons: {_reason_map}")
                    return TrainingResult(
                        symbol=symbol,
                        model_type=model_type,
                        benchmark_passed=True,
                        evaluator_passed=False,
                        metrics=metric_values,
                        model=model,
                    )
              
            except Exception as e:
                LOGGER.warning(F"Encountered: {str(e)}")
                return TrainingResult(
                        symbol="None",
                        model_type=model_type,
                        benchmark_passed=False,
                        evaluator_passed=False,
                        metrics=metric_values,
                        model=model,
                    )
        else:
            fn_args = ModelBuildTrainArguments(
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            callbacks=[],
            steps_per_epoch=self.config.steps_per_epoch,
            )
            try:
                raw_model = model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
                model_obj = model.model

                eval_values = raw_model.evaluate(eval_ds, return_dict=True, verbose=0)
                LOGGER.info(f"Evaluation results for {symbol} {model_type}: {eval_values}")
                metric_values = {
                    "accuracy": float(eval_values.get("accuracy", 0.0)),
                    "precision_buy": float(eval_values.get("precision_buy", 0.0)),
                    "precision_sell": float(eval_values.get("precision_sell", 0.0)),
                    "recall_buy": float(eval_values.get("recall_buy", 0.0)),
                    "recall_sell": float(eval_values.get("recall_sell", 0.0)),
                    "val_loss": float(eval_values.get("loss", 0.0)),
                    "train_loss": float(model.history.history["loss"][-1]),
                }

                evaluator_passed, _reason_map = self.evaluator.evaluate(metric_values)
                if not evaluator_passed:
                    LOGGER.warning("Evaluator rejected model for %s/%s", symbol, model_type)
                    LOGGER.warning(f"Failure Reasons: {_reason_map}")
                    return TrainingResult(
                        symbol=symbol,
                        model_type=model_type,
                        benchmark_passed=True,
                        evaluator_passed=False,
                        metrics=metric_values,
                        model=model,
                      )
            except Exception as e:
                LOGGER.warning(f"Error training model: {str(e)}")
                return None
                
        LOGGER.info("PASSED EVALUATION TEST")
        LOGGER.info(f"PASS RESULT: {_reason_map}")

        
        data_range = {
            "start": data_start,#pd.Timestamp(frame["timestamp"].min()).strftime("%Y-%m-%d"),
            "end": data_end #pd.Timestamp(frame["timestamp"].max()).strftime("%Y-%m-%d"),
        }
        model_id = None
        try:
          if self.upload_models:
            model_id = self.pusher.push(
                model=model,
                symbol=symbol,
                model_type=model_type,
                metrics=metric_values,
                sequence_length=sequence_length,
                data_range=data_range,
                benchmark_passed=True,
                features_keys=list(train_ds.element_spec[0].keys())
            )
        except Exception as e:
            LOGGER.warning(f"Errror occured uploading models, Error: {str(e)}")

        try:
            if self.run_performance_test:
                test_model_live_performance(model, symbol, group, 
                                          sequence_length, self.config, model_id,
                                          stride=self.config.test_generator_stride, 
                                          eval_metrics=_reason_map,
                                          min_target_points=min_target_point,
                                          feature_keys=train_ds.element_spec[0],
                                          )
        except Exception as error:
            LOGGER.warning("Optional fg-tester call failed for %s/%s: %s", symbol, model_type, error)
            # raise error
        return TrainingResult(
            symbol=symbol,
            model_type=model_type,
            benchmark_passed=True,
            evaluator_passed=True,
            metrics=metric_values,
            model=model,
            model_id= model_id,
        )

    def deserialize(self, data, preprocessor: PreprocessBase):
        return tf.io.parse_example(data, features=preprocessor.features_metadata())
        # return tf.io.parse_example(data, features={
        #     'time':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.int64),
        #     'open':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'high':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'close':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'low':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'spread':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'real_volume':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'tick_volume':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
        #     'target_value':tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        #     'target_highest':tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        #     'target_lowest':tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        # })

    def preprocess(self, data, preprocess_layer: Layer):
        # data = preprocess_layer(data, training=True)
        target = data.pop('target')
        return data, tf.one_hot(target, depth=len(CLASS_IDS))
    
    def get_training_data(self, file_pattern: str, preprocessor: Layer, repeat=False):
        import glob
        
        if self.use_dataframe_format:
            # Load from parquet dataframe format
            split_name = os.path.basename(file_pattern)
            parquet_file = Path(file_pattern) / f"{split_name}_data.parquet"
            
            if not parquet_file.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
            print("parquet_file: ",parquet_file)
            # Load dataframe
            df = pd.read_parquet(parquet_file)
            
            # Convert dataframe to TensorFlow dataset
            # Each column in the dataframe should contain the array data
            dataset_dict = {}
            dtype_mappings = {
                'float32':tf.float32,
                'float64':tf.float32,
                'int32':tf.int64,
                'int64':tf.int64,
            }
            for col in df.columns:
                dtype = dtype_mappings[str(df[col].dtype)]
                # Assuming each row contains the full array for that column
                dataset_dict[col] = tf.convert_to_tensor(df[col].values, dtype=dtype)
            
            data = tf.data.Dataset.from_tensor_slices(dataset_dict)
            
            if self.config.shuffle_data:
                data = data.shuffle(self.config.shuffle_buffer_size, reshuffle_each_iteration=True)
            
            data = data.map(
                lambda x: self.preprocess(x, preprocess_layer=preprocessor),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            data = data.cache()
            data = data.batch(self.config.batch_size, drop_remainder=True)
            data = data.prefetch(tf.data.AUTOTUNE)
        else:
            # Load from TFRecord format (existing logic)
            split_name = os.path.basename(file_pattern)
            files = sorted(glob.glob(str(file_pattern) + f"/{split_name}_*.gz"))
            
            data = tf.data.TFRecordDataset(
                files,
                compression_type="GZIP",
                buffer_size=100 * 1024 * 1024,
                num_parallel_reads=tf.data.AUTOTUNE
            )
            data = data.map(lambda x: self.deserialize(x, preprocessor), num_parallel_calls=tf.data.AUTOTUNE)
            if self.config.shuffle_data:
                data = data.shuffle(self.config.shuffle_buffer_size, reshuffle_each_iteration=True)
            data = data.map(
                lambda x: self.preprocess(x, preprocess_layer=preprocessor),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            data = data.cache()
            data = data.batch(self.config.batch_size, drop_remainder=True)
            data = data.prefetch(tf.data.AUTOTUNE)
        
        return data
