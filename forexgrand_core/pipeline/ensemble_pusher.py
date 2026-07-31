"""Bundle several already-pushed models into a single ensemble artifact.

Reuses the exact same storage-key conventions as ModelPusher
(`prediction-models/{id}/model.zip`, `.../properties.json`, `.../xgboost.json`,
`.../transformer.pkl`) so AuxilaryModelManager can load either a single model
or an ensemble from the same `model_id` entrypoint.
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
import zipfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from forexgrand_core.settings import Settings
from forexgrand_core.storage.utils import getStorageClient

LOGGER = logging.getLogger(__name__)

# Class index convention (must match single-model training/prediction code):
#   0 = buy, 1 = sell, 2 = hold
BUY, SELL, HOLD = 0, 1, 2

_VALID_SIGNAL_TYPES = {"both", "buy_only", "sell_only"}
_VALID_CONFIRMATION_TYPES = {"majority", "unanimous", "priority_tiebreak"}


class EnsemblePusher:
    """Downloads already-pushed models and bundles them into one ensemble artifact."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.storage_client = getStorageClient(config.s3_storage_option)(self.config)
        self.storage_bucket = config.models_bucket
        self._client = self.storage_client.client

    # ------------------------------------------------------------------
    # Storage key helpers (mirror AuxilaryModelManager's convention)
    # ------------------------------------------------------------------
    def _model_key(self, model_id: str) -> str:
        return f"prediction-models/{model_id}/model.zip"

    def _xgb_key(self, model_id: str) -> str:
        return f"prediction-models/{model_id}/xgboost.json"

    def _transformer_key(self, model_id: str) -> str:
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

    def download_file(self, bucket: str, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
        self._client.download_file(bucket, key, str(dest))

    # ------------------------------------------------------------------
    # Fetching each component model's artifacts
    # ------------------------------------------------------------------
    def _fetch_component(self, model_id: str, workdir: Path) -> dict[str, Any]:
        bucket = self.storage_bucket
        comp_dir = workdir / model_id
        model_zip_path = comp_dir / "model.zip"
        meta_path = comp_dir / "properties.json"

        for key in (self._model_key(model_id), self._metadata_key(model_id)):
            if not self.object_exists(bucket, key):
                raise FileNotFoundError(f"Object not found in bucket: {key}")

        self.download_file(bucket, self._model_key(model_id), model_zip_path)
        self.download_file(bucket, self._metadata_key(model_id), meta_path)

        properties = json.loads(meta_path.read_text())
        model_type = properties.get("model_type", "unknown")

        xgb_path = None
        transformer_path = None

        if "xgb" in model_type.lower():
            xgb_path = comp_dir / "xgboost.json"
            self.download_file(bucket, self._xgb_key(model_id), xgb_path)

        if properties.get("has_feature_transformer", False):
            transformer_key = self._transformer_key(model_id)
            if self.object_exists(bucket, transformer_key):
                transformer_path = comp_dir / "transformer.pkl"
                self.download_file(bucket, transformer_key, transformer_path)

        return {
            "model_id": model_id,
            "properties": properties,
            "paths": {
                "model_zip": model_zip_path,
                "meta": meta_path,
                "xgboost": xgb_path,
                "transformer": transformer_path,
            },
        }

    # ------------------------------------------------------------------
    # Metadata building
    # ------------------------------------------------------------------
    @staticmethod
    def _average(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _build_ensemble_metadata(
        self,
        components: list[dict[str, Any]],
        signal_type: str,
        confirmation_type: str,
        priority_order: list[str],
        enforce_symbol_match: bool = True,
        filter_model_id: str | None = None,
        filter_class: int | None = None,
    ) -> dict[str, Any]:
        seq_lengths = [int(c["properties"].get("sequence_length", 0)) for c in components]
        max_seq_length = max(seq_lengths) if seq_lengths else 0

        metrics_lists = {
            "precision_buy": [],
            "recall_buy": [],
            "precision_sell": [],
            "recall_sell": [],
        }
        for c in components:
            m = c["properties"].get("metrics", {})
            for key in metrics_lists:
                metrics_lists[key].append(float(m.get(key, 0.0)))

        avg_metrics = {key: self._average(vals) for key, vals in metrics_lists.items()}

        if signal_type == "buy_only":
            avg_metrics["precision_sell"] = 0.0
            avg_metrics["recall_sell"] = 0.0
        elif signal_type == "sell_only":
            avg_metrics["precision_buy"] = 0.0
            avg_metrics["recall_buy"] = 0.0

        priority_lookup = {mid: idx + 1 for idx, mid in enumerate(priority_order)}

        component_entries = [
            {
                "model_id": c["model_id"],
                "model_type": c["properties"].get("model_type", "unknown"),
                "priority": priority_lookup[c["model_id"]],
                "sequence_length": int(c["properties"].get("sequence_length", 0)),
                "feature_keys": c["properties"].get("feature_keys", []),
                "has_feature_transformer": bool(c["properties"].get("has_feature_transformer", False)),
                "symbol": c["properties"].get("symbol"),
            }
            for c in components
        ]
        component_entries.sort(key=lambda m: m["priority"])

        symbols = {c["properties"].get("symbol") for c in components}
        if len(symbols) > 1:
            if enforce_symbol_match:
                raise ValueError(
                    f"Ensemble components span multiple symbols: {sorted(symbols)}. "
                    "Pass enforce_symbol_match=False to allow this."
                )
            LOGGER.warning("Ensemble components span multiple symbols: %s", symbols)

        return {
            "model_kind": "ensemble",
            "symbol": next(iter(symbols)) if len(symbols) == 1 else sorted(symbols),
            "signal_type": signal_type,
            "confirmation_type": confirmation_type,
            "sequence_length": max_seq_length,
            "trained_at": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
            "component_models": component_entries,
            "metrics": avg_metrics,
            "filter_model_id": filter_model_id,
            "filter_class": filter_class,
        }

    def _package_container(self, components: list[dict[str, Any]], output_zip_path: Path) -> None:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for c in components:
                model_id = c["model_id"]
                paths = c["paths"]
                zf.write(paths["model_zip"], arcname=f"models/{model_id}/model.zip")
                zf.write(paths["meta"], arcname=f"models/{model_id}/properties.json")
                if paths["xgboost"] is not None:
                    zf.write(paths["xgboost"], arcname=f"models/{model_id}/xgboost.json")
                if paths["transformer"] is not None:
                    zf.write(paths["transformer"], arcname=f"models/{model_id}/transformer.pkl")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push(
        self,
        *,
        model_ids: list[str],
        signal_type: str = "both",
        confirmation_type: str = "majority",
        priority_order: list[str] | None = None,
        enforce_symbol_match: bool = True,
        filter_model_id: str | None = None,
        filter_class: int | None = None,
    ) -> str:
        """Bundle several already-pushed models into one ensemble artifact.

        Args:
            model_ids: ids of previously-pushed models (single-model artifacts)
                to include in the ensemble.
            signal_type: "both" | "buy_only" | "sell_only". Recorded in
                metadata for downstream consumers; not applied by the
                ensemble's own voting logic.
            confirmation_type: "majority" | "unanimous" | "priority_tiebreak".
                Controls how per-model votes are resolved into one signal.
            priority_order: ids in most-trusted-first order, used to break
                ties when confirmation_type == "priority_tiebreak". Defaults
                to the order of model_ids if not given.
            enforce_symbol_match: if True (default), raises when component
                models span different symbols. Set False to allow
                cross-symbol ensembles (logs a warning instead).
            filter_model_id / filter_class: optional pair identifying another
                already-pushed model that acts as a data-point filter ahead
                of this ensemble. If either is given, both must be given.
                When set, AuxilaryModelManager runs the filter model first
                and overrides any row where its prediction != filter_class
                to a HOLD prediction.

        Returns:
            The new ensemble's model_id (uuid string), usable anywhere a
            single model_id was previously used.
        """
        if not model_ids:
            raise ValueError("model_ids must contain at least one model id.")
        if signal_type not in _VALID_SIGNAL_TYPES:
            raise ValueError(f"signal_type must be one of {_VALID_SIGNAL_TYPES}")
        if confirmation_type not in _VALID_CONFIRMATION_TYPES:
            raise ValueError(f"confirmation_type must be one of {_VALID_CONFIRMATION_TYPES}")
        if (filter_model_id is None) != (filter_class is None):
            raise ValueError(
                "filter_model_id and filter_class must be provided together."
            )

        priorities = priority_order or list(model_ids)
        if set(priorities) != set(model_ids):
            raise ValueError("priority_order must contain exactly the same ids as model_ids.")

        ensemble_id = str(uuid.uuid4())
        model_object_key = f"prediction-models/{ensemble_id}/model.zip"
        props_object_key = f"prediction-models/{ensemble_id}/properties.json"

        with tempfile.TemporaryDirectory(prefix="fg_ensemble_push_") as temp_dir:
            workdir = Path(temp_dir)
            components = [self._fetch_component(mid, workdir) for mid in model_ids]

            metadata = self._build_ensemble_metadata(
                components=components,
                signal_type=signal_type,
                confirmation_type=confirmation_type,
                priority_order=priorities,
                enforce_symbol_match=enforce_symbol_match,
                filter_model_id=filter_model_id,
                filter_class=filter_class,
            )
            metadata_path = workdir / "properties.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            container_zip_path = workdir / "model.zip"
            self._package_container(components, container_zip_path)

            self.storage_client.upload_file(
                file_directory=str(container_zip_path),
                bucket=self.storage_bucket,
                object_key=model_object_key,
            )
            self.storage_client.upload_file(
                file_directory=str(metadata_path),
                bucket=self.storage_bucket,
                object_key=props_object_key,
            )

        LOGGER.info("Ensemble model pushed to %s", model_object_key)
        return ensemble_id