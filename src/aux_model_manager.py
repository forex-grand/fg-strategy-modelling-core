import os
import json
import logging
import tempfile
import threading
from datetime import datetime, UTC
from pathlib import Path
import tensorflow as tf
import keras
import uuid
import zipfile
try:
    from openfe import OpenFE, transform
except ModuleNotFoundError:
    OpenFE = None
    transform = None

from src.settings import Settings
from src.storage.utils import getStorageClient
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None
try:
    import joblib
except ModuleNotFoundError:
    joblib = None
import re
from typing import Any
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)
cpu_counts = os.cpu_count()
_OPENFE_TRANSFORM_LOCK = threading.Lock()



def _is_transformer_bundle(feature_transformer):
    return isinstance(feature_transformer, dict) and "features" in feature_transformer

def _get_openfe_features(feature_transformer):
    if _is_transformer_bundle(feature_transformer):
        return feature_transformer["features"]
    return feature_transformer

def _get_openfe_imputer(feature_transformer):
    if _is_transformer_bundle(feature_transformer):
        return feature_transformer.get("imputer")
    return None

def _openfe_transform(X_train, X_eval, feature_transformer, n_jobs):
    if transform is None:
        raise ModuleNotFoundError("openfe is required to transform OpenFE features.")
    features = _get_openfe_features(feature_transformer)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in exp",
            category=RuntimeWarning,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return transform(X_train, X_eval, features, n_jobs=n_jobs)

def _sanitize_openfe_frame(frame):
    return frame.replace([np.inf, -np.inf], np.nan)

def transform_fe(X, feature_transformer):
  cpu_counts = os.cpu_count()
  X_transformed, _ = _openfe_transform(X, X, feature_transformer, n_jobs=cpu_counts)
  X_transformed = _sanitize_openfe_frame(X_transformed)
  imputer = _get_openfe_imputer(feature_transformer)
  if imputer is not None:
    X_transformed = pd.DataFrame(imputer.transform(X_transformed), columns=X_transformed.columns)
  return X_transformed

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
          if XGBClassifier is None:
              raise ModuleNotFoundError("xgboost is required to load auxiliary XGBoost models.")
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
            if joblib is None:
              raise ModuleNotFoundError("joblib is required to load OpenFE transformer bundles.")
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
  
  def _ohlc_to_feature_dict(self, row_dict) -> dict[str, tf.Tensor]:
      sequence_length = int(self.model['metadata']['sequence_length'])
      feature_dict = {}
      for key in ("time", "open", "high", "close", "low", "spread", "tick_volume", "real_volume"):
          value = row_dict[key]
          if isinstance(value, tf.Tensor):
              tensor = value
          else:
              tensor = tf.convert_to_tensor(value)

          if tensor.shape.rank == 1:
              tensor = tf.expand_dims(tensor, axis=0)

          if tensor.shape.rank >= 2 and tensor.shape[-1] != sequence_length:
              tensor = tensor[:, -sequence_length:]

          feature_dict[key] = tensor
      return feature_dict

  def prepare_data(self, data):
      ts = datetime.now()
      prepared = self._ohlc_to_feature_dict(data)
      return prepared


  # ---------------------------------------------------------------------------
  # Inference
  # ---------------------------------------------------------------------------

  def run_nn_inference(self, model: tf.keras.Model, data: dict[str, tf.Tensor]) -> list[list[float]]:
      """
      Run saved-model inference on a feature dictionary.
      Returns a list of per-row prediction lists.
      """
      preds = model(**data)['output']
      preds_np = preds.numpy() if hasattr(preds, 'numpy') else np.asarray(preds)
      if preds_np.ndim == 1:
          preds_np = preds_np.reshape(-1,)

      return preds_np.tolist()

  def run_xgboost_inference(self, model_dict: dict, data: dict[str, tf.Tensor]) -> list[float]:
      try:
          preprocessed = model_dict['model'](**data)['output']
          preprocessed_np = preprocessed.numpy() if hasattr(preprocessed, 'numpy') else np.asarray(preprocessed)
          props = model_dict['metadata']
          if props['has_feature_transformer']:
            data = pd.DataFrame(data=preprocessed_np, columns=props['feature_keys'])
            with _OPENFE_TRANSFORM_LOCK:
              preprocessed_np = transform_fe(data, model_dict['feature_transformer'])

          preds = model_dict['xgboost'].predict(preprocessed_np)
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
      if "nn" in model_type.lower() or model_type.lower()=='no-train':
          preds = self.run_nn_inference(model["model"], data)
      else:
          preds = self.run_xgboost_inference(model, data)
      
      return preds

