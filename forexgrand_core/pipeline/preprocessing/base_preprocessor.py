"""Base preprocessing layer embedded in the model graph."""

from __future__ import annotations

import abc
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

    def __init__(self, sequence_length: int, min_target_points=200, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.min_target_points = min_target_points
        self.auxmodel_dictionary: dict[str, Any] = {}

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "min_target_points":self.min_target_points,
        })
        return config
    
    def call(self, inputs, training: bool = False):
        """Run preprocessing inside the Keras graph."""
        return self.preprocess(inputs, training=training)

    @keras.utils.register_keras_serializable(name="Preproces_function")
    def get_transform_layer(self,):
        return self

    def attach_intermediary_model(
        self,
        model_id: str,
        name: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Download and attach an auxiliary model for use inside preprocessors.

        The loaded object is stored in ``self.auxmodel_dictionary`` and can be
        accessed by subclasses during ``preprocess``.
        """
        if not model_id:
            raise ValueError("model_id is required to attach an intermediary model.")

        from forexgrand_core.aux_model_manager import AuxilaryModelManager

        model_name = name or model_id
        manager = AuxilaryModelManager(model_id=model_id, output_path=output_path)
        self.auxmodel_dictionary[model_name] = manager.model
        return manager.model

    @abc.abstractmethod
    @keras.utils.register_keras_serializable(name="Preproces_function")
    def preprocess(self, inputs, training: bool = False) -> dict[str, tf.Tensor]:
        """Transform raw model inputs into downstream tensors."""

    @abc.abstractmethod
    @keras.utils.register_keras_serializable(name="Preproces_function")
    def features_metadata(self,):
        pass

    def _copy_tree(self, source: str, target: str) -> None:
        tf.io.gfile.makedirs(target)
        for item in tf.io.gfile.listdir(source):
            src_item = f"{source}/{item}"
            dst_item = str(Path(target) / item)
            if tf.io.gfile.isdir(src_item):
                self._copy_tree(src_item, dst_item)
            else:
                tf.io.gfile.copy(src_item, dst_item, overwrite=True)

    @keras.utils.register_keras_serializable(name="aggregate_function")    
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
        tf.debugging.assert_equal(
            x['close'].ndim, 2,
            message="Each data shape should maintain a shape of (rows, sequence_length)."
        )
        
        ##check if sequence length produces at least 2 bars.
        bars_per_conversion_bar = target_tf_minutes//current_tf_minutes
        min_num_partitions = (close_shape[1]//bars_per_conversion_bar)
        tf.debugging.assert_greater_equal(min_num_partitions, 2, message=f"Sequence Length not enough to perform aggregation or aggregation bars < 2. sequence length: {close_shape[1]}, bars_expected: {close_shape[1]//bars_per_conversion_bar}")     

        times       = x['time']
        opens       = x['open']
        closes      = x['close']
        highs       = x['high']
        lows        = x['low']
        real_vol    = x['real_volume']
        spread      = x['spread']
        tick_vols   = x['tick_volume']

        ##check if times is in seconds
        units_val = tf.math.floordiv(times[0][0],10**9)
        tf.debugging.assert_equal(
            units_val,
            tf.constant(1, tf.int64),
            message=f"Timestamp field is not in seconds"
        )

        ##Convert the times into the target timeframe compartments or ids.
        target_tf_seconds = target_tf_minutes * 60
        bar_start_indexes = times // target_tf_seconds
        keys = ["time", "open", "close", "high", "low", "spread", "real_volume", "tick_volume"]


        def compute_bars_from_row(row_tuple):
            start_indices, _opens, _highs, _closes, _lows, _real_volumes, _spreads, _tick_volumes = row_tuple
            unique_vals, unq_indices, _ = tf.unique_with_counts(start_indices)
            num_partitions = tf.shape(unique_vals)[0]
            indices = tf.range(tf.shape(unq_indices)[0])
            indices_groups = tf.ragged.stack_dynamic_partitions(indices, partitions=unq_indices, num_partitions=num_partitions)
            required_indices_groups = indices_groups[-min_num_partitions:]
            
            ta_time = tf.gather(start_indices, required_indices_groups)[:, -1:].flat_values*target_tf_seconds
            ta_open = tf.gather(_opens, required_indices_groups)[:, :1].flat_values
            ta_close = tf.gather(_closes, required_indices_groups)[:, -1:].flat_values
            ta_high  = tf.reduce_max(tf.gather(_highs, required_indices_groups), axis=-1)
            ta_low   = tf.reduce_max(tf.gather(_lows, required_indices_groups), axis=-1)
            ta_spread   = tf.reduce_mean(tf.gather(_spreads, required_indices_groups), axis=-1)
            ta_real_vol = tf.reduce_sum(tf.gather(_real_volumes, required_indices_groups), axis=-1)
            ta_tick_vol = tf.reduce_sum(tf.gather(_tick_volumes, required_indices_groups), axis=-1)
            bar_tuples = (
                ta_time,
                ta_open,
                ta_close,
                ta_high,
                ta_low,
                ta_spread,
                ta_real_vol,
                ta_tick_vol,
            )

            # bar_tuples is a tuple of 8 tensors, each shape (num_bars,)
            return bar_tuples

        results = [compute_bars_from_row((bar_start_indexes[i],opens[i],highs[i],closes[i], lows[i], real_vol[i], spread[i], tick_vols[i])) for i in range(bar_start_indexes.shape[0])]

        min_counts = min_num_partitions-2

        full_data = {
          'time':  tf.stack([b[0][-min_counts:] for b in results]),
          'open':  tf.stack([b[1][-min_counts:] for b in results]),
          'close': tf.stack([b[2][-min_counts:] for b in results]),
          'high':  tf.stack([b[3][-min_counts:] for b in results]),
          'low':   tf.stack([b[4][-min_counts:] for b in results]),
          'spread':tf.stack([b[5][-min_counts:] for b in results]),
          'real_vol':tf.stack([b[6][-min_counts:] for b in results]),
          'tick_vol':tf.stack([b[7][-min_counts:] for b in results]),
          }
        for k in full_data:
          full_data[k] = tf.ensure_shape(full_data[k], [close_shape[0], min_counts])
        return full_data

        results = [compute_bars_from_row((bar_start_indexes[i],opens[i],highs[i],closes[i], lows[i], real_vol[i], spread[i], tick_vols[i])) for i in range(bar_start_indexes.shape[0])]

        min_counts = min_num_partitions-2

        full_data = {
          'time':  tf.stack([b[0][-min_counts:] for b in results]),
          'open':  tf.stack([b[1][-min_counts:] for b in results]),
          'close': tf.stack([b[2][-min_counts:] for b in results]),
          'high':  tf.stack([b[3][-min_counts:] for b in results]),
          'low':   tf.stack([b[4][-min_counts:] for b in results]),
          'spread':tf.stack([b[5][-min_counts:] for b in results]),
          'real_vol':tf.stack([b[6][-min_counts:] for b in results]),
          'tick_vol':tf.stack([b[7][-min_counts:] for b in results]),
          }
        for k in full_data:
          full_data[k] = tf.ensure_shape(full_data[k], [close_shape[0], min_counts])
        return full_data


    def extract_end_times(
        self,
        x:tf.Tensor,
    )->tf.Tensor:
        return tf.squeeze(x[:,-1])
