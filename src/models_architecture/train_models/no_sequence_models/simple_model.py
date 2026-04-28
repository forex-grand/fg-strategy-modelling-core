from src.models_architecture.train_models.base_train_model import TrainModel
import tensorflow as tf
import keras

class SimpleNSTrainModel(TrainModel):
    """
    Simple No-Sequence Model.
    """
    def __init__(self, preprocessor, sequence_length):
        super().__init__(preprocessor, sequence_length)
    
    def _validate_shape(self, spec:tf.TensorSpec):
        # if len(spec.shape)>2:
        #     raise ValueError("Can't use a sequence features for a no sequence model.")
        return True
    
    def build_model(self, input_spec:dict[str,tf.TensorSpec]):
        for _, spec in input_spec.items():
            self._validate_shape(spec=spec)

        inputs = {key:keras.Input(shape=(spec.shape[-1] or 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}

        x = keras.layers.concatenate(list(inputs.values()))
        x = keras.layers.Dense(32)(x)
        x = keras.layers.Dense(32)(x)
        output = keras.layers.Dense(3, activation="softmax")(x)
        return keras.Model(inputs, output)