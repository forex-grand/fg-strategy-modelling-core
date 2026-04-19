from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.settings import Settings
from src.storage.clients import (
    AWSStorageClient,
    CloudflareR2StorageClient,
    GoogleS3StorageClient,
    MinIOStorageClient,
)
from src.storage.utils import StorageOptionEnumeration


class DataManager:
    REQUIRED_COLUMNS = {
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "real_volume",
        "spread",
    }

    STORAGE_CLIENTS = {
        StorageOptionEnumeration.AWS.value: AWSStorageClient,
        StorageOptionEnumeration.CLOUDFLARE_R2.value: CloudflareR2StorageClient,
        StorageOptionEnumeration.MINIO.value: MinIOStorageClient,
        StorageOptionEnumeration.GOOGLE_S3.value: GoogleS3StorageClient,
    }

    def __init__(
        self,
        storage_client: Optional[
            AWSStorageClient
            | CloudflareR2StorageClient
            | GoogleS3StorageClient
            | MinIOStorageClient
        ] = None,
        *,
        settings: Optional[Settings] = None,
        data_source: Optional[str] = None,
        base_bucket_name: Optional[str] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.data_source = (data_source or self.settings.data_source or "").strip().lower()
        if not self.data_source:
            raise ValueError("A data source is required to initialize the data manager.")

        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()
        self.storage_client = storage_client or self._build_storage_client()
        self.base_bucket_name = (
            (base_bucket_name or self.storage_client.bucket_name or "").strip()
            or self.settings.s3_bucket_name
        )
        self.storage_client.bucket_name = self.base_bucket_name

    def load_data(self, symbol_pair: str, instrument_group: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        pair_name = symbol_pair.strip().upper()
        group_name = instrument_group.strip().lower()
        if not pair_name:
            raise ValueError("symbol_pair is required.")
        if not group_name:
            raise ValueError("instrument_group is required.")

        parquet_path = self._build_local_directory(group_name, pair_name) / f"{pair_name}_M1.parquet"
        properties_path = parquet_path.parent / "properties.json"

        if self.settings.force_reload or not self._local_files_exist(parquet_path, properties_path):
            dataframe, properties = self._download_and_cache(group_name, pair_name, parquet_path, properties_path)
        else:
            dataframe, properties = self._load_local_files(parquet_path, properties_path)

        self._validate_properties(properties, pair_name)
        self._validate_dataframe(dataframe, pair_name)
        return dataframe, properties

    def _build_storage_client(self):
        storage_option = (self.settings.s3_storage_option or "").strip().lower()
        client_class = self.STORAGE_CLIENTS.get(storage_option)
        if client_class is None:
            supported = ", ".join(option.value for option in StorageOptionEnumeration)
            raise ValueError(
                f"Unsupported storage option '{storage_option}'. Supported options: {supported}."
            )
        return client_class(settings=self.settings)

    def _download_and_cache(
        self,
        instrument_group: str,
        pair_name: str,
        parquet_path: Path,
        properties_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        bucket_directory = self._build_bucket_directory(instrument_group, pair_name)
        parquet_buffer, properties = self.storage_client.download_data_files(
            bucket_directory,
            parquet_filename=f"{pair_name}_M1.parquet",
            json_filename="properties.json",
        )

        dataframe = self._read_parquet_buffer(parquet_buffer)
        self._validate_properties(properties, pair_name)
        self._validate_dataframe(dataframe, pair_name)
        self._save_local_files(dataframe, properties, parquet_path, properties_path)
        return dataframe, properties

    def _load_local_files(
        self,
        parquet_path: Path,
        properties_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        dataframe = pd.read_parquet(parquet_path)
        with properties_path.open("r", encoding="utf-8") as file_handle:
            properties = json.load(file_handle)
        return dataframe, properties

    def _save_local_files(
        self,
        dataframe: pd.DataFrame,
        properties: dict[str, Any],
        parquet_path: Path,
        properties_path: Path,
    ) -> None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(parquet_path, index=False)
        with properties_path.open("w", encoding="utf-8") as file_handle:
            json.dump(properties, file_handle, indent=2)

    def _build_bucket_directory(self, instrument_group: str, pair_name: str) -> str:
        return f"{self.data_source}/{instrument_group}/{pair_name}"

    def _build_local_directory(self, instrument_group: str, pair_name: str) -> Path:
        return self.data_directory / self.data_source / instrument_group / pair_name

    @staticmethod
    def _local_files_exist(parquet_path: Path, properties_path: Path) -> bool:
        return parquet_path.exists() and properties_path.exists()

    @staticmethod
    def _read_parquet_buffer(parquet_buffer: BytesIO) -> pd.DataFrame:
        parquet_buffer.seek(0)
        return pd.read_parquet(parquet_buffer)

    @classmethod
    def _validate_dataframe(cls, dataframe: pd.DataFrame, pair_name: str) -> None:
        normalized = dataframe.reset_index() if "time" not in dataframe.columns and dataframe.index.name == "time" else dataframe
        missing_columns = cls.REQUIRED_COLUMNS - set(normalized.columns)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Parquet data for '{pair_name}' is missing required columns: {missing_list}."
            )

    @staticmethod
    def _validate_properties(properties: dict[str, Any], pair_name: str) -> None:
        contract_size = properties.get("contract_size")
        if contract_size in (None, ""):
            raise ValueError(
                f"properties.json for '{pair_name}' does not contain a contract_size value."
            )
