from src.models_architecture.base_model import BaseModel
import tensorflow as tf
import keras
import os


@keras.utils.register_keras_serializable()
class LambdaLayer(keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs['direction']
  
class NoTrainModel(BaseModel):
    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length

    def evaluate(self, eval_ds):
        ##Define evaluation metrics
        precision_buy = keras.metrics.Precision(class_id=0)
        precision_sell= keras.metrics.Precision(class_id=1)
        precision_hold= keras.metrics.Precision(class_id=2)
        
        recall_buy = keras.metrics.Recall(class_id=0)
        recall_sell= keras.metrics.Recall(class_id=1)
        recall_hold= keras.metrics.Recall(class_id=2)
        accuracy   = keras.metrics.CategoricalAccuracy()

        cardinality = eval_ds.cardinality()
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH","-1"))
        num_batches = steps_per_epoch if steps_per_epoch>0 else (cardinality if cardinality>0 else 100)
        
        num_eval_batches = cardinality if cardinality>0 else num_batches
        if cardinality==-2:
            num_eval_batches = -1
        for batch_x, batch_y in eval_ds.take(num_eval_batches):
            y_true = tf.cast(tf.squeeze((batch_y)), tf.int32)
            y_pred = tf.one_hot(batch_x['direction'], depth=3)
            precision_buy.update_state(y_true, y_pred)
            precision_sell.update_state(y_true, y_pred)
            precision_hold.update_state(y_true, y_pred)

            recall_buy.update_state(y_true, y_pred)
            recall_sell.update_state(y_true, y_pred)
            recall_hold.update_state(y_true, y_pred)

            accuracy.update_state(y_true, y_pred)

        metrics = {
            'accuracy':accuracy.result(),
            'precision_buy':precision_buy.result(),
            'precision_sell':precision_sell.result(),
            'recall_buy':recall_buy.result(),
            'recall_sell':recall_sell.result(),
            'val_loss':0.0,
            'train_loss':0.0,
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
