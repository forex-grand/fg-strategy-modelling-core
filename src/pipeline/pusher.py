"""Push validated models to GCS with strict artifact conventions."""

from __future__ import annotations

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
from src.models_architecture.base_model import BaseModel
from src.storage.utils import getStorageClient
try:
    import joblib
except ModuleNotFoundError:
    joblib = None
import re

LOGGER = logging.getLogger(__name__)

class ModelPusher:
    """Persists models and metadata under required GCS path conventions."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.storage_client = getStorageClient(config.s3_storage_option)(self.config)
        self.storage_bucket = config.models_bucket

    def push(
        self,
        *,
        model: BaseModel,
        symbol: str,
        model_type: str,
        metrics: dict[str, float],
        sequence_length: int,
        data_range: dict[str, str],
        benchmark_passed: bool,
        features_keys: list = [],
        filter_model_id: str | None = None,
        filter_class: int | None = None,
    ) -> str:
        """Save tf.saved_model and metadata, then upload to target GCS path.

        filter_model_id / filter_class: optional pair identifying another
        already-pushed model that acts as a data-point filter ahead of this
        model. If either is given, both must be given. When set, any row
        where the filter model's prediction != filter_class will be
        overridden to a HOLD prediction downstream (handled by
        AuxilaryModelManager, not here).
        """
        if (filter_model_id is None) != (filter_class is None):
            raise ValueError(
                "filter_model_id and filter_class must be provided together."
            )

        trained_at = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        unique_identifier = uuid.uuid4()
        model_object_key = f"prediction-models/{unique_identifier}/model.zip"
        props_object_key = f"prediction-models/{unique_identifier}/properties.json"

        with tempfile.TemporaryDirectory(prefix="fg_push_") as temp_dir:
            local_root = Path(temp_dir)
            model_dir = local_root / "saved_model"

            # Save SavedModel to disk
            tf.saved_model.save(
                model.model,
                str(model_dir),
                signatures={"serving_default": model.get_serving_signature()}
            )

            # Zip the saved_model directory
            zip_path = local_root / "model.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file in model_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, arcname=file.relative_to(model_dir))

            metadata = {
                "symbol": symbol,
                "model_type": model_type.lower(),
                "has_feature_transformer":True if model.feature_transformer else False,
                "feature_keys":features_keys,
                "sequence_length": int(sequence_length),
                "trained_at": trained_at,
                "metrics": {
                    "precision_buy": float(metrics.get("precision_buy", 0.0)),
                    "recall_buy": float(metrics.get("recall_buy", 0.0)),
                    "precision_sell": float(metrics.get("precision_sell", 0.0)),
                    "recall_sell": float(metrics.get("recall_sell", 0.0)),
                    "val_loss": float(metrics.get("val_loss", 0.0)),
                    "train_loss": float(metrics.get("train_loss", 0.0)),
                },
                "data_range": data_range,
                "benchmark_passed": bool(benchmark_passed),
                "filter_model_id": filter_model_id,
                "filter_class": filter_class,
            }
            metadata_path = local_root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            
            if re.match(r"xgb",model_type.lower()):
                local_xgb_path = local_root / "xgboost.json"
                if not model.xgb_model:
                    raise ValueError("XGBoost Model object is none, can't save.")
                model.xgb_model.save_model(local_xgb_path)
                xgboost_object_key = f"prediction-models/{unique_identifier}/xgboost.json"
                self.storage_client.upload_file(
                  file_directory=str(local_xgb_path),
                  bucket=self.storage_bucket,
                  object_key=xgboost_object_key,
                  )

            if model.feature_transformer is not None:
              if joblib is None:
                raise ModuleNotFoundError("joblib is required to upload feature transformer bundles.")
              local_transformer_path = local_root / "transformer.pkl"
              joblib.dump(model.feature_transformer, local_transformer_path)
              xgboost_object_key = f"prediction-models/{unique_identifier}/transformer.pkl"
              self.storage_client.upload_file(
              file_directory=str(local_transformer_path),
              bucket=self.storage_bucket,
              object_key=xgboost_object_key,
              )

            self.storage_client.upload_file(
                file_directory=str(zip_path),
                bucket=self.storage_bucket,
                object_key=model_object_key,
            )
            self.storage_client.upload_file(
                file_directory=str(metadata_path),
                bucket=self.storage_bucket,
                object_key=props_object_key,
            )

        LOGGER.info("Model pushed to %s", model_object_key)
        return unique_identifier