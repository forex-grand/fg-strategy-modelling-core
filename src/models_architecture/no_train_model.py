from src.models_architecture.base_model import BaseModel
import numpy as np
import tensorflow as tf
import keras
import os


@keras.utils.register_keras_serializable()
class LambdaLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        direction_values = NoTrainModel._extract_direction_values(inputs, None)
        return NoTrainModel._to_int_tensor(direction_values)


class NoTrainModel(BaseModel):
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length

    @staticmethod
    def _extract_direction_values(batch_x, batch_y):
        if batch_y is not None:
            if isinstance(batch_y, dict):
                if "direction" in batch_y:
                    return batch_y["direction"]
                if "target" in batch_y:
                    return batch_y["target"]
            return batch_y

        if isinstance(batch_x, dict):
            if "direction" in batch_x:
                return batch_x["direction"]
            if "target" in batch_x:
                return batch_x["target"]

        if isinstance(batch_x, tuple):
            for item in batch_x:
                if isinstance(item, dict) and "direction" in item:
                    return item["direction"]

        raise ValueError("No direction values were found in the provided batch.")

    @staticmethod
    def _to_int_tensor(values) -> tf.Tensor:
        tensor = tf.cast(values, tf.int32)
        return tf.reshape(tensor, [-1])

    def evaluate(self, eval_ds):
        cardinality = eval_ds.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "-1"))
        num_batches = steps_per_epoch if steps_per_epoch > 0 else (cardinality if cardinality > 0 else 100)

        num_eval_batches = cardinality if cardinality > 0 else num_batches
        if cardinality == -2:
            num_eval_batches = -1

        accuracy = keras.metrics.CategoricalAccuracy()
        precision_buy = keras.metrics.Precision(class_id=0)
        precision_sell = keras.metrics.Precision(class_id=1)
        recall_buy = keras.metrics.Recall(class_id=0)
        recall_sell = keras.metrics.Recall(class_id=1)

        for batch_x, batch_y in eval_ds.take(num_eval_batches):
            direction_values = self._extract_direction_values(batch_x, batch_y)
            y_true = self._to_int_tensor(direction_values)
            y_pred = y_true

            y_true_np = np.asarray(y_true.numpy() if hasattr(y_true, "numpy") else y_true)
            if y_true_np.size == 0:
                continue

            unique_labels = np.unique(y_true_np)
            if unique_labels.size == 0:
                continue

            class_id_map = {int(label): idx for idx, label in enumerate(unique_labels)}
            remapped_true = np.vectorize(class_id_map.get)(y_true_np).astype(np.int32)
            remapped_pred = remapped_true.copy()

            depth = max(len(class_id_map), 2)
            y_true_one_hot = tf.one_hot(tf.constant(remapped_true), depth=depth)
            y_pred_one_hot = tf.one_hot(tf.constant(remapped_pred), depth=depth)

            if depth > 1:
                precision_buy.update_state(y_true_one_hot, y_pred_one_hot)
                precision_sell.update_state(y_true_one_hot, y_pred_one_hot)
                recall_buy.update_state(y_true_one_hot, y_pred_one_hot)
                recall_sell.update_state(y_true_one_hot, y_pred_one_hot)
            else:
                precision_buy.update_state(y_true_one_hot, y_pred_one_hot)
                recall_buy.update_state(y_true_one_hot, y_pred_one_hot)

            accuracy.update_state(y_true_one_hot, y_pred_one_hot)

        metrics = {
            "accuracy": accuracy.result(),
            "precision_buy": precision_buy.result(),
            "precision_sell": precision_sell.result(),
            "recall_buy": recall_buy.result(),
            "recall_sell": recall_sell.result(),
            "val_loss": 0.0,
            "train_loss": 0.0,
        }
        return metrics

    def build_train_model(self, train_ds, eval_ds, fn_args):
        inputs = self._build_input_signature()
        inputs_dict = {inp.name: inp for inp in inputs}
        x = self.preprocessor(inputs_dict)
        y = LambdaLayer()(x)
        model = keras.Model(inputs, y)
        self.model = model
        self.model.compile()
        return self.model
    
    def get_serving_signature(self):
        input_signature = {
            "time": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.int64, name="time"),
            "open": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="open"),
            "high": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="high"),
            "close": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="close"),
            "low": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="low"),
            "spread": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="spread"),
            "real_volume": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="real_volume"),
            "tick_volume": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="tick_volume"),
        }

        @tf.function(input_signature=[input_signature])
        def serve(examples):
            return {"output":self.model(examples)}
        return serve