from __future__ import annotations

from forexgrand_core.models_architecture.train_models.base_train_model import TrainModel
import tensorflow as tf
import keras

try:
    import keras_tuner as kt
except ImportError:
    kt = None

class ConservativeNSTrainModel(TrainModel):
    """
    Conservative No-Sequence Model.
    """
    def __init__(self, preprocessor, sequence_length):
        super().__init__(preprocessor, sequence_length)
    
    def build_model(
        self,
        input_spec: dict[str, tf.TensorSpec],
        num_classes: int,
        hp: kt.HyperParameters | None = None,
    ):
        inputs = {key:keras.Input(shape=(spec.shape[-1] if spec.shape.rank>1 else 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        x = keras.layers.concatenate(list(inputs.values()))
        if hp is None:
            layer_units = [64, 64, 32]
            dropout_rate = 0.3
        else:
            num_layers = hp.Int("dense_layers", min_value=2, max_value=5, default=3)
            layer_units = [
                hp.Int(f"dense_units_{idx}", min_value=32, max_value=256, step=32, default=64)
                for idx in range(num_layers)
            ]
            dropout_rate = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.05, default=0.3)

        for units in layer_units:
            x = keras.layers.Dense(units)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        output = keras.layers.Dense(
            num_classes,
            activation="softmax",
            bias_initializer=self._output_bias_initializer(),
        )(x)
        return keras.Model(inputs, output)

    
