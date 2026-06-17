import os
import json
import logging
import tempfile
from datetime import datetime, UTC
from pathlib import Path
import tensorflow as tf
import keras
import uuid
import zipfile
from src.settings import Settings
from src.storage.utils import getStorageClient
from xgboost import XGBClassifier
import joblib
import re
from typing import Any
from botocore.exceptions import ClientError
from openfe import transform
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
cpu_counts = os.cpu_count()

class AuxilaryModelManager:
  def __init__(self, model_id, output_path: str=None) -> None:
      self.config = Settings()
      self.storage_client = getStorageClient(self.config.s3_storage_option)(self.config)
      self.storage_bucket = self.config.models_bucket
      self.data_directory = Path(output_path) / 'models' if output_path is not None else Path(self.config.data_directory).expanduser().resolve() / 'models'
      self._client = self.storage_client.client
      self.model  = self.fetch_model_from_storage(model_id=model_id)

  def download_file(self, bucket: str, key: str, dest: Path) -> None:
      dest = Path(dest)
      dest.parent.mkdir(parents=True, exist_ok=True)
      logger.info("Downloading s3://%s/%s → %s", bucket, key, dest)
      self._client.download_file(bucket, key, str(dest))

  def get_model_directory(self, model_id):
      return self.data_directory / model_id
  
  def _local_files_exist(self, parquet_path: Path) -> bool:
      return parquet_path.exists()
  
  def _model_key(self, model_id: str) -> str:
      return f"prediction-models/{model_id}/model.zip"
  
  def _xgb_key(self, model_id: str) -> str:
      return f"prediction-models/{model_id}/xgboost.json"

  def _feature_transformer_key(self, model_id: str):
      return  f"prediction-models/{model_id}/transformer.pkl"

  def _metadata_key(self, model_id: str) -> str:
      return f"prediction-models/{model_id}/properties.json"
 
  def object_exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

  def fetch_model_from_storage(self, model_id: str) -> dict[str, Any]:
      """
      Download model + metadata from R2.
      Returns {"model": tf.keras.Model, "metadata": dict}.
      Raises FileNotFoundError if either object is missing.
      """
      bucket = self.storage_bucket
      model_key = self._model_key(model_id)
      meta_key = self._metadata_key(model_id)

      for key in (model_key, meta_key):
          if not self.object_exists(bucket, key):
              raise FileNotFoundError(f"Object not found in bucket: {key}")

      model_dict = {}
      
      tmp_path = self.get_model_directory(model_id=model_id)
      model_zip_path = tmp_path / "model.zip"
      extract_path = tmp_path / "model"   # folder for extracted model
      meta_path = tmp_path / "properties.json"

      # Download files
      if not self._local_files_exist(model_zip_path):
        self.download_file(bucket, model_key, model_zip_path)
      if not self._local_files_exist(meta_path):
        self.download_file(bucket, meta_key, str(meta_path))
      
      properties = json.loads(meta_path.read_text())
      model_type = properties.get("model_type", "unknown")

      if "xgb" in model_type.lower():
          xgb_path = tmp_path / "xgboost.json"
          xgb_key = self._xgb_key(model_id)
          if not self._local_files_exist(xgb_path):
            self.download_file(bucket, xgb_key, str(xgb_path))

          xgb_model = XGBClassifier()
          xgb_model.load_model(str(xgb_path))
          model_dict["xgboost"] = xgb_model

          ftransform_path = tmp_path / "ftransformer.pkl"
          ftransform_key  = self._feature_transformer_key(model_id)
          ftransform_exist = self._local_files_exist(ftransform_path)
          
          has_feature_transformer = properties.get('has_feature_transformer', False)
          if has_feature_transformer:
            if not ftransform_exist:
              if self.object_exists(bucket, ftransform_key):
                  self.download_file(bucket, ftransform_key, str(ftransform_path))
            ftransformer = joblib.load(str(ftransform_path))
            model_dict['feature_transformer'] = ftransformer
              
      # Extract zip into extract_path
      with zipfile.ZipFile(model_zip_path, "r") as zipf:
          zipf.extractall(extract_path)

      # Load TensorFlow SavedModel from extracted folder
      model_obj = tf.saved_model.load(str(extract_path))
      model = model_obj.signatures['serving_default']
      model_dict["model"] = model

      with open(meta_path) as f:
          metadata: dict = json.load(f)
          model_dict["metadata"] = metadata

      logger.info("Loaded model '%s' from storage.", model_id)
      return model_dict
  
  def _ohlc_to_feature_dict(self, row_dict) -> dict:
      length = len(row_dict['time'])
      sequence_length = self.model['metadata']['sequence_length']
      time_  = row_dict['time'].numpy()[:,-sequence_length:]
      open_  = row_dict['open'].numpy()[:,-sequence_length:]
      high_  = row_dict['high'].numpy()[:,-sequence_length:]
      close_ = row_dict['close'].numpy()[:,-sequence_length:]
      low_   = row_dict['low'].numpy()[:,-sequence_length:]
      spread_      = row_dict['spread'].numpy()[:,-sequence_length:]
      tick_volume_ = row_dict['tick_volume'].numpy()[:,-sequence_length:]
      real_volume_ = row_dict['real_volume'].numpy()[:,-sequence_length:]
      
      feature_dict = [tf.train.Example(features=tf.train.Features(feature={
          "time": tf.train.Feature(int64_list=tf.train.Int64List(value=time_[i])),
          "open": tf.train.Feature(float_list=tf.train.FloatList(value=open_[i])),
          "high": tf.train.Feature(float_list=tf.train.FloatList(value=high_[i])),
          "close": tf.train.Feature(float_list=tf.train.FloatList(value=close_[i])),
          "low": tf.train.Feature(float_list=tf.train.FloatList(value=low_[i])),
          "spread": tf.train.Feature(float_list=tf.train.FloatList(value=spread_[i])),
          "tick_volume": tf.train.Feature(float_list=tf.train.FloatList(value=tick_volume_[i])),
          "real_volume": tf.train.Feature(float_list=tf.train.FloatList(value=real_volume_[i])),
      })).SerializeToString() for i in range(length)]
      return feature_dict

  def prepare_data(self, data):
      ts = datetime.now()
      serialized = self._ohlc_to_feature_dict(data)
      return serialized


  # ---------------------------------------------------------------------------
  # Inference
  # ---------------------------------------------------------------------------

  def run_nn_inference(self, model: tf.keras.Model, data: np.ndarray) -> list[list[float]]:
      """
      Run model.predict on *data* (shape (N, 8)).
      Returns a list of per-row prediction lists.
      """
      preds: np.ndarray = model(data, verbose=0)
      # Ensure 2-D: (N, output_units)
      if preds.ndim == 1:
          preds = preds.reshape(-1, 1)
      return preds.tolist()

  def run_xgboost_inference(self, model_dict: dict, data: np.ndarray) -> list[float]:
      try:
          preprocessed = model_dict['model'](examples=data)['output'].numpy().tolist()
          props = model_dict['metadata']
          if props['has_feature_transformer']:
            data = pd.DataFrame(data=preprocessed, columns=props['feature_keys'])
            preprocessed, _ = transform(data, data, new_features_list=model_dict['feature_transformer'],n_jobs=cpu_counts)

          preds = model_dict['xgboost'].predict(preprocessed)
          if preds.ndim==2:
              preds = np.argmax(preds, axis=1)
          return preds.tolist()
      except Exception as e:
          raise ValueError(f"Value Error: {str(e)}")

  
  def predict(self, data):
      model = self.model
      if not model:
          self.model = self.fetch_model_from_storage(self.model_id)
          if not self.model:
              raise ValueError(f"Error encountered loading model: {self.model_id}")
          model = self.model
      data = self.prepare_data(data)
      model_type = model["metadata"].get("model_type", "none")
      if "nn" in model_type.lower():
          preds = self.run_nn_inference(model["model"], data)
      else:
          preds = self.run_xgboost_inference(model, data)
      
      return preds

  