"""Data manager for loading and caching forex market data.

This module provides the DataManager class that handles loading M1 (minute-level) forex data
from configured S3-compatible storage, with local caching for performance.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from forexgrand_core.settings import Settings
from forexgrand_core.storage.utils import getStorageClient, StorageOptionEnumeration
from forexgrand_core.storage.base_client import BaseStorageClient
from forexgrand_core.schemas import SymbolProperties


class DataManager:
    """Loads and caches M1 forex data from S3-compatible storage.
    
    This class manages data fetching from remote storage and local caching with validation.
    It supports multiple S3-compatible backends (MinIO, AWS S3, GCS, Cloudflare R2).
    
    Attributes:
        REQUIRED_COLUMNS: Set of required columns in forex data files.
        settings: Configuration settings loaded from environment.
        data_source: Data source type (e.g., 'mt5').
        data_directory: Path to local data cache directory.
        storage_client: Storage client for remote data access.
        base_bucket_name: Name of the bucket for data storage.
    
    Raises:
        ValueError: If data source is not configured or storage option is unsupported.
    
    Example:
        >>> manager = DataManager(base_bucket_name="forexgrand-train")
        >>> df, props = manager.load_data("EURUSD", "forex")
        >>> print(df.head())
    """
    
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

    def __init__(
        self,
        base_bucket_name: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        self.settings = Settings()
        self.data_source = (source or self.settings.data_source or "").strip()
        if not self.data_source:
            raise ValueError("No data source found in environment variables.")
        
        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()
        self.storage_client:BaseStorageClient = self._build_storage_client()(
          self.settings,
        )
        self.base_bucket_name = (
            (base_bucket_name or "").strip()
            or self.settings.s3_bucket_name
        )
        self.storage_client.bucket_name = self.base_bucket_name

    def load_data(self, symbol_pair: str, instrument_group: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load M1 forex data for a symbol pair.
        
        Loads market data from local cache if available and valid, otherwise downloads from
        remote storage and updates the cache. Automatically validates all data.
        
        Args:
            symbol_pair: Currency pair symbol (e.g., 'EURUSD'). Case-insensitive.
            instrument_group: Instrument group classification (e.g., 'forex', 'metals', 'crypto').
                Case-insensitive.
        
        Returns:
            A tuple of:
                - pd.DataFrame: Market data with OHLCV and spread columns
                - dict: Symbol properties including contract size and other metadata
        
        Raises:
            ValueError: If symbol_pair or instrument_group are empty/invalid, or if
                data files are missing required columns or metadata.
            OSError: If data files cannot be read or written.
        
        Example:
            >>> manager = DataManager()
            >>> df, props = manager.load_data("EURUSD", "forex")
            >>> print(f"Loaded {len(df)} rows for {props['symbol']}")
        """
        pair_name = symbol_pair.strip()
        group_name = instrument_group.strip()
        if not pair_name:
            raise ValueError("symbol_pair is required.")
        if not group_name:
            raise ValueError("instrument_group is required.")

        parquet_path = self._build_local_directory(group_name, pair_name) / f"{pair_name}_M1.parquet"
        properties_path = parquet_path.parent / "properties.json"

        if self.settings.force_reload or not self._local_files_exist(parquet_path, properties_path):
            dataframe, properties = self._download_and_cache(group_name, pair_name, parquet_path, properties_path)
        else:
            try:
                dataframe, properties = self._load_local_files(parquet_path, properties_path)
                self._validate_properties(properties, pair_name)
                self._validate_dataframe(dataframe, pair_name)
            except (OSError, ValueError, json.JSONDecodeError):
                dataframe, properties = self._download_and_cache(
                    group_name,
                    pair_name,
                    parquet_path,
                    properties_path,
                )

        self._validate_properties(properties, pair_name)
        self._validate_dataframe(dataframe, pair_name)
        return dataframe, SymbolProperties.model_validate(properties)

    def _build_storage_client(self):
        """Create and configure storage client based on settings.
        
        Returns:
            BaseStorageClient: Configured storage client instance.
            
        Raises:
            ValueError: If S3_STORAGE_OPTION is not supported.
        """
        storage_option = (self.settings.s3_storage_option or "").strip() or "minio"
        client_class = getStorageClient(self.settings.s3_storage_option)
        if client_class is None:
            supported = ", ".join(option.value for option in StorageOptionEnumeration)
            raise ValueError(
                f"Unsupported storage option '{storage_option}'. Supported options: {supported}."
            )
        return client_class

    def _download_and_cache(
        self,
        instrument_group: str,
        pair_name: str,
        parquet_path: Path,
        properties_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        bucket_directory = self._build_bucket_directory(instrument_group, pair_name)
        resolved_bucket_name = (
            (self.base_bucket_name or self.storage_client.bucket_name or "").strip()
            or self.settings.s3_bucket_name
        )
        parquet_buffer, properties = self.storage_client.download_data_files(
            bucket_directory,
            bucket_name=resolved_bucket_name,
            parquet_filename=f"{pair_name}_M1.parquet",
            json_filename="properties.json",
        )

        dataframe = self._read_parquet_buffer(parquet_buffer)
        
        self._validate_properties(properties, pair_name)
        self._validate_dataframe(dataframe, pair_name)
        dataframe = self._save_local_files(dataframe, properties, parquet_path, properties_path)
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
        normalized_dataframe = self._normalize_dataframe(dataframe)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_dataframe.to_parquet(parquet_path, index=False)
        with properties_path.open("w", encoding="utf-8") as file_handle:
            json.dump(properties, file_handle, indent=2)
        return normalized_dataframe

    def _build_bucket_directory(self, instrument_group: str, pair_name: str) -> str:
        return f"{self.data_source}/{instrument_group}/{pair_name}"

    def _build_local_directory(self, instrument_group: str, pair_name: str) -> Path:
        return self.data_directory / self.base_bucket_name / self.data_source / instrument_group / pair_name

    @staticmethod
    def _local_files_exist(parquet_path: Path, properties_path: Path) -> bool:
        return parquet_path.exists() and properties_path.exists()

    @staticmethod
    def _read_parquet_buffer(parquet_buffer: BytesIO) -> pd.DataFrame:
        parquet_buffer.seek(0)
        return pd.read_parquet(parquet_buffer)

    @classmethod
    def _normalize_dataframe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        if "time" not in dataframe.columns and dataframe.index.name == "time":
            return dataframe.reset_index()
        return dataframe.copy()

    @classmethod
    def _validate_dataframe(cls, dataframe: pd.DataFrame, pair_name: str) -> None:
        normalized = cls._normalize_dataframe(dataframe)
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
