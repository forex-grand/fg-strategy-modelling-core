import os
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import keras
import zipfile
from collections import Counter
try:
    from openfe import OpenFE, transform
except ModuleNotFoundError:
    OpenFE = None
    transform = None

from forexgrand_core.settings import Settings
from forexgrand_core.storage.utils import getStorageClient
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None
try:
    import joblib
except ModuleNotFoundError:
    joblib = None
from typing import Any
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)
cpu_counts = os.cpu_count()
_OPENFE_TRANSFORM_LOCK = threading.Lock()

# Class index convention (must match EnsemblePusher and training/prediction code):
#   0 = buy, 1 = sell, 2 = hold
BUY, SELL, HOLD = 0, 1, 2


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
  def __init__(self, model_id, output_path: str = None) -> None:
      self.config = Settings()
      self.storage_client = getStorageClient(self.config.s3_storage_option)(self.config)
      self.storage_bucket = self.config.models_bucket
      self.data_directory = Path(output_path) / 'models' if output_path is not None else Path(self.config.data_directory).expanduser().resolve() / 'models'
      self._client = self.storage_client.client
      self.model_id = model_id
      self.model = self.fetch_model_from_storage(model_id=model_id)
      self.model_obj = None

  # ------------------------------------------------------------------
  # Storage helpers
  # ------------------------------------------------------------------
  def download_file(self, bucket: str, key: str, dest: Path) -> None:
      dest = Path(dest)
      dest.parent.mkdir(parents=True, exist_ok=True)
      logger.info("Downloading s3://%s/%s → %s", bucket, key, dest)
      self._client.download_file(bucket, key, str(dest))

  def get_model_directory(self, model_id):
      return self.data_directory / model_id

  def _local_files_exist(self, path: Path) -> bool:
      return path.exists()

  def _model_key(self, model_id: str) -> str:
      return f"prediction-models/{model_id}/model.zip"

  def _xgb_key(self, model_id: str) -> str:
      return f"prediction-models/{model_id}/xgboost.json"

  def _feature_transformer_key(self, model_id: str):
      return f"prediction-models/{model_id}/transformer.pkl"

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

  # ------------------------------------------------------------------
  # Loading: dispatches to single-model or ensemble loader based on
  # the `model_kind` field in properties.json
  # ------------------------------------------------------------------
  def fetch_model_from_storage(self, model_id: str) -> dict[str, Any]:
      """
      Download model + metadata from storage and load it.

      Returns a dict shaped either as:
        {"kind": "single", "metadata": ..., "model": ..., ["xgboost": ...,
         "feature_transformer": ...]}
      or:
        {"kind": "ensemble", "metadata": ..., "components": {model_id: {...}}}

      Raises FileNotFoundError if the container/metadata objects are missing.
      """
      bucket = self.storage_bucket
      model_key = self._model_key(model_id)
      meta_key = self._metadata_key(model_id)

      for key in (model_key, meta_key):
          if not self.object_exists(bucket, key):
              raise FileNotFoundError(f"Object not found in bucket: {key}")

      tmp_path = self.get_model_directory(model_id=model_id)
      model_zip_path = tmp_path / "model.zip"
      meta_path = tmp_path / "properties.json"

      if not self._local_files_exist(model_zip_path):
          self.download_file(bucket, model_key, model_zip_path)
      if not self._local_files_exist(meta_path):
          self.download_file(bucket, meta_key, str(meta_path))

      properties = json.loads(meta_path.read_text())
      model_kind = properties.get("model_kind", "single")

      if model_kind == "ensemble":
          result = self._load_ensemble(tmp_path, model_zip_path, properties)
      else:
          result = self._load_single(model_id, tmp_path, model_zip_path, meta_path, properties)

      filter_model_id = properties.get("filter_model_id")
      if filter_model_id:
          result["filter_model_id"] = filter_model_id
          result["filter_class"] = properties.get("filter_class")
          # Filter models are themselves regular pushed models (single or
          # ensemble); reuse the same loader recursively so they get the
          # exact same load path, including their own filter if any.
          result["filter_model"] = self.fetch_model_from_storage(filter_model_id)

      return result

  def _load_single(self, model_id, tmp_path, model_zip_path, meta_path, properties) -> dict[str, Any]:
      model_dict: dict[str, Any] = {"kind": "single", "metadata": properties}
      model_type = properties.get("model_type", "unknown")
      extract_path = tmp_path / "model"

      if "xgb" in model_type.lower():
          if XGBClassifier is None:
              raise ModuleNotFoundError("xgboost is required to load auxiliary XGBoost models.")
          xgb_path = tmp_path / "xgboost.json"
          if not self._local_files_exist(xgb_path):
              self.download_file(self.storage_bucket, self._xgb_key(model_id), str(xgb_path))
          xgb_model = XGBClassifier()
          xgb_model.load_model(str(xgb_path))
          model_dict["xgboost"] = xgb_model

          if properties.get('has_feature_transformer', False):
              if joblib is None:
                  raise ModuleNotFoundError("joblib is required to load OpenFE transformer bundles.")
              ftransform_path = tmp_path / "ftransformer.pkl"
              ftransform_key = self._feature_transformer_key(model_id)
              if not self._local_files_exist(ftransform_path):
                  if self.object_exists(self.storage_bucket, ftransform_key):
                      self.download_file(self.storage_bucket, ftransform_key, str(ftransform_path))
              model_dict['feature_transformer'] = joblib.load(str(ftransform_path))

      if not self._local_files_exist(extract_path):
          with zipfile.ZipFile(model_zip_path, "r") as zipf:
              zipf.extractall(extract_path)

      model_obj = tf.saved_model.load(str(extract_path))
      model_dict["_model_root"] = model_obj
      model_dict["model"] = model_obj.signatures['serving_default']
      
      logger.info("Loaded model '%s' from storage.", model_id)
      return model_dict

  def _load_ensemble(self, tmp_path, container_zip_path, properties) -> dict[str, Any]:
      extract_root = tmp_path / "ensemble"
      if not self._local_files_exist(extract_root):
          with zipfile.ZipFile(container_zip_path, "r") as zf:
              zf.extractall(extract_root)

      components: dict[str, Any] = {}
      for comp_meta in properties.get("component_models", []):
          comp_id = comp_meta["model_id"]
          comp_dir = extract_root / "models" / comp_id
          comp_props_path = comp_dir / "properties.json"
          comp_properties = (
              json.loads(comp_props_path.read_text()) if comp_props_path.exists() else comp_meta
          )

          comp_dict: dict[str, Any] = {
              "metadata": comp_properties,
              "priority": comp_meta["priority"],
          }

          if "xgb" in comp_meta.get("model_type", "").lower():
              if XGBClassifier is None:
                  raise ModuleNotFoundError("xgboost is required to load auxiliary XGBoost models.")
              xgb_model = XGBClassifier()
              xgb_model.load_model(str(comp_dir / "xgboost.json"))
              comp_dict["xgboost"] = xgb_model

              if comp_meta.get("has_feature_transformer", False):
                  if joblib is None:
                      raise ModuleNotFoundError("joblib is required to load OpenFE transformer bundles.")
                  comp_dict["feature_transformer"] = joblib.load(str(comp_dir / "transformer.pkl"))

          comp_extract_path = comp_dir / "model"
          if not self._local_files_exist(comp_extract_path):
              with zipfile.ZipFile(comp_dir / "model.zip", "r") as zf:
                  zf.extractall(comp_extract_path)

          model_obj = tf.saved_model.load(str(comp_extract_path))
          comp_dict["model"] = model_obj.signatures["serving_default"]

          components[comp_id] = comp_dict

      logger.info(
          "Loaded ensemble '%s' (%d components) from storage.",
          properties.get("symbol"), len(components),
      )

      return {"kind": "ensemble", "metadata": properties, "components": components}

  # ------------------------------------------------------------------
  # Feature prep
  # ------------------------------------------------------------------
  def _ohlc_to_feature_dict(self, row_dict, sequence_length: int = None) -> dict[str, tf.Tensor]:
      if sequence_length is None:
          sequence_length = int(self.model.get("metadata", {}).get("sequence_length", 0))
      if sequence_length <= 0:
          raise ValueError("sequence_length is required to prepare OHLC feature data.")

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

  def prepare_data(self, data, sequence_length: int):
      return self._ohlc_to_feature_dict(data, sequence_length)

  # ---------------------------------------------------------------------------
  # Inference (unchanged from original single-model implementation)
  # ---------------------------------------------------------------------------

  def run_nn_inference(self, model: tf.keras.Model, data: dict[str, tf.Tensor], model_dict: dict = None) -> list[list[float]]:
      """
      Run saved-model inference on a feature dictionary.
      Returns a list of per-row prediction lists.
      """
      preds = model(**data)['output']
      preds_np = preds.numpy() if hasattr(preds, 'numpy') else np.asarray(preds)
      if preds_np.ndim == 1:
          preds_np = preds_np.reshape(-1,)

      predictions = preds_np.tolist()

      if model_dict.get("filter_model") is not None:
          predictions = self._apply_filter(model_dict, data, predictions)      

      return predictions
  def run_xgboost_inference(self, model_dict: dict, data: dict[str, tf.Tensor]) -> list[float]:
      try:
          preprocessed = model_dict['model'](**data)['output']
          preprocessed_np = preprocessed.numpy() if hasattr(preprocessed, 'numpy') else np.asarray(preprocessed)
          props = model_dict['metadata']
          if props['has_feature_transformer']:
            data = pd.DataFrame(data=preprocessed_np, columns=props['feature_keys'])
            with _OPENFE_TRANSFORM_LOCK:
              preprocessed_np = transform_fe(data, model_dict['feature_transformer'])

          model = model_dict['xgboost']

          best_iteration_range = None
          try:
            best_iteration_range = (0, model.best_iteration + 1)
          except:
            pass
          preds = model.predict(preprocessed_np, iteration_range=best_iteration_range)
          if preds.ndim == 2:
              preds = np.argmax(preds, axis=1)
          predictions = preds.tolist()    

          if model_dict.get("filter_model") is not None:
            predictions = self._apply_filter(model_dict, data, predictions)

          return predictions
      except Exception as e:
          raise ValueError(f"Value Error: {str(e)}")

  # ------------------------------------------------------------------
  # Predict: dispatches to single-model or ensemble voting logic
  # ------------------------------------------------------------------
  def predict(self, data):
      if not self.model:
          self.model = self.fetch_model_from_storage(self.model_id)
          if not self.model:
              raise ValueError(f"Error encountered loading model: {self.model_id}")

      model = self.model

      if model["kind"] == "ensemble":
          predictions = self._predict_ensemble(model, data)
      else:
          predictions = self._predict_single(model, data)

      if model.get("filter_model") is not None:
          predictions = self._apply_filter(model, data, predictions)

      return predictions

  def _apply_filter(self, model, data, predictions):
      """
      Runs this model's filter model on `data` first, then overrides any
      row where the filter's prediction doesn't match filter_class to HOLD,
      leaving matching rows' original predictions untouched.
      """
      filter_model = model["filter_model"]
      filter_class = model.get("filter_class")

      if filter_model["kind"] == "ensemble":
          filter_preds = self._predict_ensemble(filter_model, data)
      else:
          filter_preds = self._predict_single(filter_model, data)

      filter_preds = filter_preds if isinstance(filter_preds, list) else [filter_preds]
      predictions = predictions if isinstance(predictions, list) else [predictions]

      if len(filter_preds) != len(predictions):
          raise ValueError(
              f"Filter model produced {len(filter_preds)} predictions but the "
              f"main model produced {len(predictions)}; row counts must match."
          )

      return [
          pred if int(fp) == filter_class else HOLD
          for fp, pred in zip(filter_preds, predictions)
      ]

  def _predict_single(self, model, data):
      seq_len = int(model["metadata"].get("sequence_length", 0))
      prepared = self.prepare_data(data, seq_len)
      model_type = model["metadata"].get("model_type", "none")
      if "xgb" in model_type.lower():
          return self.run_xgboost_inference(model, prepared)
    
      
      return self.run_nn_inference(model["model"], prepared, model)

  def _predict_ensemble(self, model, data):
      """
      Runs every component model over `data`, then resolves a vote per row.

      Returns a list of resolved class predictions, one per input row
      (length 1 in -> length 1 out, same shape contract as single-model
      predict()).
      """
      ensemble_meta = model["metadata"]
      confirmation_type = ensemble_meta.get("confirmation_type", "majority")

      # {comp_id: [pred_row0, pred_row1, ...]}
      component_predictions: dict[str, list[int]] = {}
      component_priority: dict[str, int] = {}

      for comp_id, comp in model["components"].items():
          seq_len = int(comp["metadata"].get("sequence_length", 0))
          prepared = self.prepare_data(data, seq_len)

          if "xgb" in comp["metadata"].get("model_type", "").lower():
              preds = self.run_xgboost_inference(comp, prepared)
          else:
              preds = self.run_nn_inference(comp["model"], prepared, comp)

          preds = preds if isinstance(preds, list) else [preds]
          component_predictions[comp_id] = [int(p) for p in preds]
          component_priority[comp_id] = comp["priority"]

      row_counts = {len(v) for v in component_predictions.values()}
      if len(row_counts) > 1:
          raise ValueError(
              f"Ensemble components produced mismatched row counts: "
              f"{ {cid: len(v) for cid, v in component_predictions.items()} }"
          )
      n_rows = row_counts.pop() if row_counts else 0

      results = []
      for row_idx in range(n_rows):
          row_votes = {
              comp_id: {
                  "prediction": component_predictions[comp_id][row_idx],
                  "priority": component_priority[comp_id],
              }
              for comp_id in component_predictions
          }
          results.append(self._resolve_ensemble_vote(row_votes, confirmation_type))

      return results

  @staticmethod
  def _resolve_ensemble_vote(votes: dict, confirmation_type: str):
      predictions = [v["prediction"] for v in votes.values()]

      if not predictions:
          return HOLD

      if confirmation_type == "unanimous":
          return predictions[0] if len(set(predictions)) == 1 else HOLD

      counts = Counter(predictions)
      top_count = max(counts.values())
      tied = [p for p, c in counts.items() if c == top_count]

      if len(tied) == 1:
          return tied[0]

      if confirmation_type == "priority_tiebreak":
          for _, v in sorted(votes.items(), key=lambda kv: kv[1]["priority"]):
              if v["prediction"] in tied:
                  return v["prediction"]

      return HOLD  # majority with unresolved tie and no priority rule -> safe default
