"""Base preprocessing layer embedded in the model graph."""

from __future__ import annotations

import abc
import tempfile
from pathlib import Path
from typing import Any
import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(name="Preprocessing_layer")
class PreprocessBase(keras.layers.Layer, metaclass=abc.ABCMeta):
    """
    Abstract preprocessing layer.

    Implementations must return:
    - Training: {"features": tensor, "target": tensor}
    - Inference: {"features": tensor}
    """

    def __init__(self, sequence_length: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        # self._aux_model_cache: dict[str, Any] = {}

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
        })
        return config
    
    def call(self, inputs, training: bool = False):
        """Run preprocessing inside the Keras graph."""
        return self.preprocess(inputs, training=training)

    @keras.utils.register_keras_serializable(name="Preproces_function")
    def get_transform_layer(self,):
        return self

    @abc.abstractmethod
    @keras.utils.register_keras_serializable(name="Preproces_function")
    def preprocess(self, inputs, training: bool = False) -> dict[str, tf.Tensor]:
        """Transform raw model inputs into downstream tensors."""

    @abc.abstractmethod
    @keras.utils.register_keras_serializable(name="Preproces_function")
    def features_metadata(self,):
        pass

    # def load_and_run_auxiliary_model(self, model_gcs_path: str, inputs: tf.Tensor) -> tf.Tensor:
    #     """
    #     Download an auxiliary model from GCS, run inference, and return output tensor.
    #     """
    #     if model_gcs_path not in self._aux_model_cache:
    #         local_model_dir = self._download_model_from_gcs(model_gcs_path)
    #         self._aux_model_cache[model_gcs_path] = tf.saved_model.load(local_model_dir)

    #     loaded_model = self._aux_model_cache[model_gcs_path]
    #     infer = loaded_model.signatures.get("serving_default")
    #     if infer is None:
    #         raise ValueError(f"Auxiliary model '{model_gcs_path}' has no serving_default signature.")

    #     try:
    #         outputs = infer(features=inputs)
    #     except TypeError:
    #         outputs = infer(inputs)

    #     if isinstance(outputs, dict):
    #         return next(iter(outputs.values()))
    #     return outputs

    # def _download_model_from_gcs(self, model_gcs_path: str) -> str:
    #     """Recursively copy a GCS model directory to a local temp directory."""
    #     source = model_gcs_path.rstrip("/")
    #     if not tf.io.gfile.exists(source):
    #         raise FileNotFoundError(f"Auxiliary model path does not exist: {source}")

    #     temp_root = Path(tempfile.mkdtemp(prefix="fg_aux_model_"))
    #     target = temp_root / "model"
    #     self._copy_tree(source, str(target))
    #     return str(target)

    def _copy_tree(self, source: str, target: str) -> None:
        tf.io.gfile.makedirs(target)
        for item in tf.io.gfile.listdir(source):
            src_item = f"{source}/{item}"
            dst_item = str(Path(target) / item)
            if tf.io.gfile.isdir(src_item):
                self._copy_tree(src_item, dst_item)
            else:
                tf.io.gfile.copy(src_item, dst_item, overwrite=True)

    
    def aggregate_to_timeframe(
        self,
        x: dict[str, tf.Tensor], #Tensor data (rows/batch_no, sequence_length, feature_length)
        target_tf_minutes: int, #New TF to compute to.
        current_tf_minutes: int = 1, #Current Data Timeframe Minutes
        )->dict[str, tf.Tensor]:
        ##check if target tf is greater than current tf
        if target_tf_minutes<current_tf_minutes and current_tf_minutes>0:
            raise ValueError("Target Timeframe should be higher than data current timeframe.")
        
        ##check if x matches expected dimension
        x_dict = {"time","open","high","close","low","spread","real_volume","tick_volume"}.difference(list(x.keys()))
        if len(x_dict)!=0:
            raise ValueError(f"Expected columns not found in data, {x_dict} not found.")
        
        close_shape = x['close'].shape
        if len(close_shape)!=2:
            raise ValueError(f"Each data shape should maintain a shape of (rows, sequence_length).")

        ##check if sequence length produces at least 2 bars.
        bars_per_conversion_bar = target_tf_minutes//current_tf_minutes
        if close_shape[1]//bars_per_conversion_bar<2:
            raise ValueError(f"Sequence Length not enough to perform aggregation or aggregation bars < 2. \
                             sequence length: {close_shape[1]}, bars_expected: {close_shape[1]//bars_per_conversion_bar}")

        times       = x['time']
        opens       = x['open']
        closes      = x['close']
        highs       = x['high']
        lows        = x['low']
        real_vol    = x['real_volume']
        spread      = x['spread']
        tick_vols   = x['tick_volume']

        ##check if times is in seconds
        if times[0][0]//10**9!=1:
            raise ValueError(f"Timestamp field is not in seconds, numbers in time: {times[0][0]//10**9}")
        
        ##Convert the times into the target timeframe compartments or ids.
        target_tf_seconds = target_tf_minutes * 60
        bar_start_indexes = times // target_tf_seconds

        keys = ["time", "open", "close", "high", "low", "spread", "real_volume", "tick_volume"]            
        def compute_bars_from_row(row_tuple):
            start_indices, _opens, _highs, _closes, _lows, _real_volumes, _spreads, _tick_volumes = row_tuple

            unique_vals, unq_indices, _ = tf.unique_with_counts(start_indices)
            range_indices = tf.cast(tf.range(tf.shape(start_indices)[0]), tf.int32)
            groups = tf.dynamic_partition(range_indices, unq_indices, tf.shape(unique_vals)[0])
            groups = groups[1:]

            num_groups = tf.shape(unique_vals)[0] - 1

            # Output TensorArrays
            ta_time     = tf.TensorArray(tf.int64, size=num_groups, dynamic_size=False)
            ta_open     = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)
            ta_close    = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)
            ta_high     = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)
            ta_low      = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)
            ta_spread   = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)
            ta_real_vol = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)
            ta_tick_vol = tf.TensorArray(tf.float32, size=num_groups, dynamic_size=False)

            def loop_body(i, ta_time, ta_open, ta_close, ta_high, ta_low, ta_spread, ta_real_vol, ta_tick_vol):
                group = groups[i]
                time     = tf.cast(start_indices[group[0]] * target_tf_seconds, tf.int64)
                open_    = _opens[group[0]]
                close_   = _closes[group[-1]]
                high_    = tf.reduce_max(tf.gather(_highs, group))
                low_     = tf.reduce_min(tf.gather(_lows, group))
                spread_  = tf.reduce_mean(tf.gather(_spreads, group))
                real_vol = tf.reduce_sum(tf.gather(_real_volumes, group))
                tick_vol = tf.reduce_sum(tf.gather(_tick_volumes, group))

                ta_time     = ta_time.write(i, time)
                ta_open     = ta_open.write(i, open_)
                ta_close    = ta_close.write(i, close_)
                ta_high     = ta_high.write(i, high_)
                ta_low      = ta_low.write(i, low_)
                ta_spread   = ta_spread.write(i, spread_)
                ta_real_vol = ta_real_vol.write(i, real_vol)
                ta_tick_vol = ta_tick_vol.write(i, tick_vol)

                return i + 1, ta_time, ta_open, ta_close, ta_high, ta_low, ta_spread, ta_real_vol, ta_tick_vol

            _, ta_time, ta_open, ta_close, ta_high, ta_low, ta_spread, ta_real_vol, ta_tick_vol = tf.while_loop(
                cond=lambda i, *_: i < num_groups,
                body=loop_body,
                loop_vars=(
                    tf.constant(0),
                    ta_time, ta_open, ta_close, ta_high, ta_low,
                    ta_spread, ta_real_vol, ta_tick_vol,
                ),
            )

            bar_tuples = (
                ta_time.stack(),
                ta_open.stack(),
                ta_close.stack(),
                ta_high.stack(),
                ta_low.stack(),
                ta_spread.stack(),
                ta_real_vol.stack(),
                ta_tick_vol.stack(),
            )
            # bar_tuples is a tuple of 8 tensors, each shape (num_bars,)
            return bar_tuples
        results = [compute_bars_from_row((bar_start_indexes[i],opens[i],highs[i],closes[i], lows[i], real_vol[i], spread[i], tick_vols[i])) for i in range(bar_start_indexes.shape[0])]

        full_data = {
          'time':  tf.stack([b[0] for b in results]),
          'open':  tf.stack([b[1] for b in results]),
          'close': tf.stack([b[2] for b in results]),
          'high':  tf.stack([b[3] for b in results]),
          'low':   tf.stack([b[4] for b in results]),
          'spread':tf.stack([b[5] for b in results]),
          'real_vol':tf.stack([b[6] for b in results]),
          'tick_vol':tf.stack([b[7] for b in results]),
          } 
      
        return full_data
    def extract_end_times(
        self,
        x:tf.Tensor,
    )->tf.Tensor:
        return tf.squeeze(x[:,-1])