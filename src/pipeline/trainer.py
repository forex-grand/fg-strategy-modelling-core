"""Training orchestration for ForexGrand models."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Type, List
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
    ) -> None:
        self.symbols: List[SymbolIn] = symbols
        self.model_types = [item.strip().lower() for item in model_types]
        self.preprocessor_class = preprocessor_class
        self.target_model_type = target_model_type
        self.config = Settings()
        self.data_gen = GenerateTrainData(
            train_base_bucket=self.config.train_bucket_name, eval_base_bucket=self.config.eval_bucket_name)
        self.evaluator = Evaluator(self.config)
        self.pusher = ModelPusher(self.config)
        self.sequence_length = sequence_length
        self.hot_reload_data = hot_reload_data
        self.run_performance_test = run_performance_test
        self.train_ds = {}
        self.eval_ds  = {}

    def run(self) -> list[TrainingResult]:
        """Execute training for all requested model types."""
        results: list[TrainingResult] = []
        for symbol in self.symbols:
            train_path, eval_path = self.data_gen.load_data(
                symbol_pair=symbol.symbol.strip().upper(),
                instrument_group=symbol.group.strip().lower(),
                sequence_length=self.sequence_length,
                stride=self.config.generated_data_strides,
                hot_reload=self.hot_reload_data,
                target_model=self.target_model_type,
            )
            data_start = self.data_gen.train_properties.data_start
            data_end = self.data_gen.train_properties.data_end
            
            preprocessor = self.preprocessor_class(sequence_length=self.sequence_length)
            symbol_string = symbol.symbol.strip().upper()
            
            train_ds = self.get_training_data(train_path, preprocessor=preprocessor)
            eval_ds  = self.get_training_data(eval_path, preprocessor=preprocessor)
            for model_type in self.model_types:
                print("+++++++==================================================================++++++++++++")
                print("+++++++==================================================================++++++++++++")
                print("+++++++==================================================================++++++++++++")
                print("+++++++==================================================================++++++++++++")
                print(f"========RUNNING TEST FOR {symbol}:{model_type}==========")
                
                results.append(self._run_single(preprocessor=preprocessor,
                                                symbol=symbol.symbol.strip().upper(),
                                                group=symbol.group.strip().lower(),
                                                model_type=model_type,
                                                train_ds=train_ds,
                                                eval_ds=eval_ds,
                                                data_start=data_start,
                                                data_end=data_end),)
                
        return results

    def _run_single(self, preprocessor, symbol,group, model_type: str, train_ds: tf.data, eval_ds: tf.data,
                    data_start: str, data_end: str) -> TrainingResult:
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
        
        elif re.match(r"^xgb", model_type):
            model_obj = model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args={})
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
                    model_gcs_path=None,
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
            raw_model = model.build_train_model(train_ds=train_ds, eval_ds=eval_ds, fn_args=fn_args)
            model_obj = model.model

            # The original error came from raw_model.evaluate, so we will not call evaluate if eval_ds is empty
            if tf.data.experimental.cardinality(eval_ds).numpy() > 0: 
                eval_values = raw_model.evaluate(eval_ds, return_dict=True, verbose=0)
                print(f"Evaluation results for {symbol} {model_type}: {eval_values}")
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
                        model_gcs_path=None,
                        metrics=metric_values,
                        model=model,
                      )
        LOGGER.warning("PASSED EVALUATION TEST")
        LOGGER.warning(f"PASS RESULT: {_reason_map}")

        data_range = {
            "start": data_start,#pd.Timestamp(frame["timestamp"].min()).strftime("%Y-%m-%d"),
            "end": data_end #pd.Timestamp(frame["timestamp"].max()).strftime("%Y-%m-%d"),
        }
        model_id = self.pusher.push(
            model=model,
            symbol=symbol,
            model_type=model_type,
            metrics=metric_values,
            sequence_length=sequence_length,
            data_range=data_range,
            benchmark_passed=True,
        )

        try:
            if self.run_performance_test:
                test_model_live_performance(model, symbol, group, 
                                          sequence_length, self.config, model_id,
                                          stride=self.config.test_generator_stride)
        except Exception as error:
            LOGGER.warning("Optional fg-tester call failed for %s/%s: %s", symbol, model_type, error)

        return TrainingResult(
            symbol=symbol,
            model_type=model_type,
            benchmark_passed=True,
            evaluator_passed=True,
            model_gcs_path=model.generate_model_path(self.sequence_length, "no_train_model"),
            metrics=metric_values,
            model=model,
            model_id= model_id,
        )

    def deserialize(self, data):
        return tf.io.parse_example(data, features={
            'time':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.int64),
            'open':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'high':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'close':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'low':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'spread':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'real_volume':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'tick_volume':tf.io.FixedLenFeature(shape=[self.sequence_length], dtype=tf.float32),
            'target_value':tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
            'target_highest':tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
            'target_lowest':tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        })

    def preprocess(self, data, preprocess_layer: Layer):
        data = preprocess_layer(data, training=True)
        target = data.pop('target')
        return data, tf.one_hot(target, depth=3)
    
    def get_training_data(self, file_pattern: str, preprocessor: Layer, repeat=False):
        import glob
        # Dynamically determine the file prefix from the last part of the file_pattern (directory name)
        split_name = os.path.basename(file_pattern)
        files = sorted(glob.glob(str(file_pattern) + f"/{split_name}_*.gz"))
        
        data = tf.data.TFRecordDataset(
            files,
            compression_type="GZIP",
            buffer_size=100 * 1024 * 1024,
            num_parallel_reads=tf.data.AUTOTUNE
        )
        data = data.map(self.deserialize, num_parallel_calls=tf.data.AUTOTUNE)
        if self.config.shuffle_data:
            data = data.shuffle(self.config.shuffle_buffer_size, reshuffle_each_iteration=True)
        data = data.batch(self.config.batch_size, drop_remainder=True)
        data = data.map(
            lambda x: self.preprocess(x, preprocess_layer=preprocessor),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        data = data.prefetch(tf.data.AUTOTUNE)
        return data
