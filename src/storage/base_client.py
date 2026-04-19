from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Optional

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError
from src.settings import Settings


class BaseStorageClient:
    """Shared boto3-backed S3 client for compatible object stores."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.bucket_name = self.settings.s3_bucket_name
        self.client = self._build_client(endpoint_url=endpoint_url)

    def _build_client(self, *, endpoint_url: Optional[str]) -> BaseClient:
        session = boto3.session.Session()
        client_kwargs: dict[str, Any] = {
            "service_name": "s3",
            "aws_access_key_id": self.settings.s3_access_key,
            "aws_secret_access_key": self.settings.s3_secret_key,
            "aws_session_token": self.settings.s3_session_token,
            "region_name": self.settings.s3_region_name,
            "config": Config(signature_version="s3v4"),
        }

        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        return session.client(**client_kwargs)

    def check_data_object(self, object_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_key)
        except ClientError as error:
            if self._is_missing_object_error(error):
                raise FileNotFoundError(
                    f"Object '{object_key}' was not found in bucket '{self.bucket_name}'."
                ) from error
            raise

        return True

    def download_data_files(
        self,
        bucket_directory: str,
        *,
        parquet_filename: str = "data.parquet",
        json_filename: str = "metadata.json",
    ) -> tuple[BytesIO, dict[str, Any]]:
        parquet_key = self._build_object_key(bucket_directory, parquet_filename)
        json_key = self._build_object_key(bucket_directory, json_filename)

        parquet_buffer = self._download_binary_object(parquet_key)
        json_payload = self._download_json_object(json_key)
        return parquet_buffer, json_payload

    def _download_binary_object(self, object_key: str) -> BytesIO:
        self.check_data_object(object_key)

        response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
        payload = response["Body"].read()
        buffer = BytesIO(payload)
        buffer.seek(0)
        return buffer

    def _download_json_object(self, object_key: str) -> dict[str, Any]:
        self.check_data_object(object_key)

        response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
        payload = response["Body"].read().decode("utf-8")
        return json.loads(payload)

    @staticmethod
    def _build_object_key(bucket_directory: str, filename: str) -> str:
        normalized_directory = bucket_directory.strip("/")
        return f"{normalized_directory}/{filename}" if normalized_directory else filename

    @staticmethod
    def _is_missing_object_error(error: ClientError) -> bool:
        error_code = str(error.response.get("Error", {}).get("Code", ""))
        return error_code in {"404", "NoSuchKey", "NotFound"}
