from src.models_architecture.train_models.base_train_model import TrainModel
import tensorflow as tf
import keras

class CNNBiLSTMModel(TrainModel):
    """
    CNN + Bi-LSTM Sequence Model.
    """
    def __init__(self, preprocessor, sequence_length):
        super().__init__(preprocessor, sequence_length)
        
    def build_model(self, input_spec:dict[str,tf.TensorSpec]):
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
        conv = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
        conv = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(conv)
        # Residual connection (project x to match conv channels)
        skip = keras.layers.Dense(64)(x)
        x = keras.layers.Add()([conv, skip])
        x = keras.layers.LayerNormalization()(x)
        # Temporal reasoning
        x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False))(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.Dense(3, activation="softmax")(x)
        return keras.Model(inputs, output)