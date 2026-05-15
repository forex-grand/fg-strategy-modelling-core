from src.models_architecture.train_models.base_train_model import TrainModel
import tensorflow as tf
import keras

class ComplexNSTrainModel(TrainModel):
    """
    Complex No-Sequence Model.
    """
    def __init__(self, preprocessor, sequence_length):
        super().__init__(preprocessor, sequence_length)
    
    def build_model(self, input_spec:dict[str,tf.TensorSpec]):
        inputs = inputs = {key:keras.Input(shape=(spec.shape[-1] if spec.shape.rank>1 else 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        x = keras.layers.concatenate(list(inputs.values()))

        # Branch 1: deep path
        b1 = keras.layers.Dense(128, activation="relu")(x)
        b1 = keras.layers.BatchNormalization()(b1)
        b1 = keras.layers.Dropout(0.3)(b1)
        b1 = keras.layers.Dense(64, activation="relu")(b1)
        b1 = keras.layers.BatchNormalization()(b1)

        # Branch 2: wide shallow path
        b2 = keras.layers.Dense(64, activation="relu")(x)
        b2 = keras.layers.BatchNormalization()(b2)

        # Merge + residual from input projection
        skip = keras.layers.Dense(64)(x)
        merged = keras.layers.Add()([b1, b2, skip])
        merged = keras.layers.LayerNormalization()(merged)
        merged = keras.layers.Dropout(0.2)(merged)

        # Final head
        out = keras.layers.Dense(32, activation="gelu")(merged)
        output = keras.layers.Dense(3, activation="softmax")(out)
        return keras.Model(inputs, output)