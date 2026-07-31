from __future__ import annotations

from forexgrand_core.models_architecture.train_models.base_train_model import TrainModel
import tensorflow as tf
import keras

try:
    import keras_tuner as kt
except ImportError:
    kt = None

class CNNBiLSTMModel(TrainModel):
    """
    CNN + Bi-LSTM Sequence Model.
    """
    def __init__(self, preprocessor, sequence_length):
        super().__init__(preprocessor, sequence_length)
        
    def build_model(
        self,
        input_spec: dict[str, tf.TensorSpec],
        num_classes: int,
        hp: kt.HyperParameters | None = None,
    ):
        ##verify all features have same length
        spec1_length = None
        features_length = 0
        for spec in input_spec.values():
            features_length += 1
            if not spec1_length:
                spec1_length = spec.shape[-1]
            else:
                if spec.shape[-1] != spec1_length:
                    raise ValueError("The features must be of same length.")

        if features_length<2:
            raise ValueError("There must be more than one feature to use lstm model.")
        
        inputs = inputs = {key:keras.Input(shape=(spec.shape[-1] if spec.shape.rank>1 else 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        x = keras.layers.concatenate(list(inputs.values()), axis=-1)  # (batch, all_features)
        x = keras.layers.Reshape(target_shape=(spec1_length,features_length))(x)
        if hp is None:
            conv_layers = 2
            conv_filters = 64
            kernel_size = 3
            lstm_units = 32
            dense_units = 64
            dropout_rate = 0.3
        else:
            conv_layers = hp.Int("conv_layers", min_value=1, max_value=3, default=2)
            conv_filters = hp.Int("conv_filters", min_value=32, max_value=128, step=32, default=64)
            kernel_size = hp.Choice("kernel_size", values=[2, 3, 5], default=3)
            lstm_units = hp.Int("lstm_units", min_value=16, max_value=128, step=16, default=32)
            dense_units = hp.Int("dense_units", min_value=32, max_value=256, step=32, default=64)
            dropout_rate = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.05, default=0.3)

        conv = x
        for _ in range(conv_layers):
            conv = keras.layers.Conv1D(
                conv_filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
            )(conv)
        # Residual connection (project x to match conv channels)
        skip = keras.layers.Dense(conv_filters)(x)
        x = keras.layers.Add()([conv, skip])
        x = keras.layers.LayerNormalization()(x)
        # Temporal reasoning
        x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units, return_sequences=False))(x)
        x = keras.layers.Dense(dense_units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        output = keras.layers.Dense(
            num_classes,
            activation="softmax",
            bias_initializer=self._output_bias_initializer(),
        )(x)
        return keras.Model(inputs, output)
