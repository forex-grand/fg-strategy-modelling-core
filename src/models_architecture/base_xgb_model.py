import os
from re import VERBOSE
from src.models_architecture.base_model import BaseModel
import tensorflow as tf
import keras
from xgboost import XGBClassifier
from abc import abstractmethod
import numpy as np
from xgboost import DMatrix
from datetime import datetime

_COMMON = dict(
    objective="multi:softmax",
    num_class=3,
    use_label_encoder=False,
    random_state=42,
    tree_method="hist",
    eval_metric="mlogloss",
    verbosity=0,
)


class XGBTrainModel(BaseModel):
    """
      The model on the base model object is the preprocessing model object that outputs an array of numpy.
    """

    def __init__(self, preprocessor, sequence_length: int):
        super().__init__(sequence_length=sequence_length, preprocessor=preprocessor)
        self.sequence_length = sequence_length
        self.train_loss = 0.0
        self.eval_loss  = 0.0

    def build_model(self, input_spec:dict[str,tf.TensorSpec]):
        inputs = {key:keras.Input(shape=(spec.shape[-1] or 1,), name=key, dtype=spec.dtype)
                  for key,spec in input_spec.items()}
        y = keras.layers.concatenate(list(inputs.values()))
        return keras.Model(inputs, y)

    @abstractmethod
    def build_xgb_model(self,):
        """ Must be implemented by any xgb training version."""
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

        for batch_x, batch_y in eval_ds.take(num_eval_batches).cache():
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
            'val_loss':self.eval_loss,
            'train_loss':self.train_loss,
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
        X = []
        y = []
        ts = datetime.now()
        for batch_x, batch_y in train_ds.take(num_batches).cache():
            Xd = np.stack([value[...,0] for value in batch_x.values()], axis=1)
            yd = np.argmax(batch_y, axis=1)
            X.append(Xd)
            y.append(yd)
        print("Training data collected: ",(datetime.now().timestamp() - ts.timestamp()),"s")
        Xe = []
        ye = []
        for batch_x, batch_y in eval_ds.take(num_batches):
            Xd = np.stack([value[...,0] for value in batch_x.values()], axis=1)
            yd = np.argmax(batch_y, axis=1)
            Xe.append(Xd)
            ye.append(yd)

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        Xe = np.concatenate(Xe, axis=0)
        ye = np.concatenate(ye, axis=0)
        print(f"Train data length: {X.shape[0]}, Eval data len: {Xe.shape[0]}")
        xgb_model.fit(X, y, eval_set=[(X, y),(Xe, ye),], verbose=int(os.getenv("XGB_VERBOSE","0")))
        results = xgb_model.evals_result()
        self.train_loss = results["validation_0"]["mlogloss"][-1]
        self.eval_loss  = results["validation_1"]["mlogloss"][-1]
        self.xgb_model = xgb_model
        return self.xgb_model