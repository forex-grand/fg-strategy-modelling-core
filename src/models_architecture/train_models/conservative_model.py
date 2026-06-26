from src.models_architecture.train_models.base_train_model import TrainModel
import tensorflow as tf
import keras

class ConservativeNSTrainModel(TrainModel):
    """
    Conservative No-Sequence Model.
    """
    def __init__(self, preprocessor, sequence_length):
        super().__init__(preprocessor, sequence_length)
    
    def build_model(self, input_spec:dict[str,tf.TensorSpec], num_classes:int):
        inputs = {key:keras.Input(shape=(spec.shape[-1] if spec.shape.rank>1 else 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        x = keras.layers.concatenate(list(inputs.values()))
        for units in [64, 64, 32]:
            x = keras.layers.Dense(units)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.Dense(num_classes, activation="softmax")(x)
        return keras.Model(inputs, output)

    