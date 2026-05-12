import os
from src.models_architecture.base_model import BaseModel
import tensorflow as tf
import keras
from xgboost import XGBClassifier
from abc import abstractmethod

class XGBTrainModel(BaseModel):
    """
      The model on the base model object is the preprocessing model object that outputs an array of numpy.
    """

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length

    def build_model(self, input_spec:dict[str,tf.TensorSpec]):
        inputs = {key:keras.Input(shape=(spec.shape[-1] or 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        y = keras.layers.concatenate(list(inputs.values()))
        return keras.Model(inputs, y)

    @abstractmethod
    def build_xgb_model(self,):
        pass

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

        for batch_x, batch_y in eval_ds.take(num_eval_batches):
            X = np.stack([value[...,0] for value in batch_x.values()], axis=1)
            y_true = np.argmax(batch_y, axis=1)
            y_pred = self.xgb_model.predict(X)
            y_pred = tf.one_hot(y_pred, depth=3)

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
        y = self.build_model(train_ds.element_spec[0])(x)
        model_ = keras.Model(inputs, y)
        self.model = model_
        
        ###initialize and train xgb classifier
        xgb_model = self.build_xgb_model()
        
        ###training loop
        cardinality = train_ds.cardinality()      
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH","-1"))
        num_batches = steps_per_epoch if steps_per_epoch>0 else (cardinality if cardinality>0 else 100)
        for batch_x, batch_y in train_ds.take(num_batches):
            X = np.stack([value[...,0] for value in batch_x.values()], axis=1)
            y = np.argmax(batch_y, axis=1)
            xgb_model.fit(X, y)
        self.xgb_model = xgb_model

        return self.xgb_model