from src.models_architecture.base_model import BaseModel
import tensorflow as tf
import keras


@keras.utils.register_keras_serializable()
class LambdaLayer(keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs['target']
    
class NoTrainModel(BaseModel):
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length

    def build_train_model(self, train_ds, eval_ds, fn_args):
        inputs = self._build_input_signature()
        inputs_dict = {inp.name: inp for inp in inputs}
        x = self.preprocessor(inputs_dict)
        y = LambdaLayer()(x)
        model = keras.Model(inputs, y)
        self.model = model
        self.model.compile()
        return self.model

    def _build_inference(self) -> keras.Model:
        inputs = self._build_input_signature(self.sequence_length)
        named_inputs = {tensor.name.split(":")[0]: tensor for tensor in inputs}
        outputs = self.preprocessor(named_inputs)

        model = keras.Model(inputs=inputs, outputs=outputs, name="no_train_inference")
        self.model = model
        return model

    def save(self, path: str):
        return self.save_inference_model(
            sequence_length=self.sequence_length,
            metadata={"requested_path": path},
        )

    @staticmethod
    def load(path: str) -> keras.Model:
        return keras.models.load_model(path)
