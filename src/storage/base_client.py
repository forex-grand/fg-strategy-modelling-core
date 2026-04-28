from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlparse

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
        self.client:boto3.client = self._build_client(endpoint_url=endpoint_url)

    def _build_client(self, *, endpoint_url: Optional[str]) -> BaseClient:
        client_kwargs: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.settings.s3_region_name,
            "aws_access_key_id": self.settings.s3_access_key,
            "aws_secret_access_key": self.settings.s3_secret_key,
            "aws_session_token": self.settings.s3_session_token,
            "config": Config(signature_version="s3v4"),
        }
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
            client_kwargs["config"] = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            )

        # Remove None values so boto3 can fall back to default credential/provider chain where applicable.
        clean_kwargs = {
            key: value for key, value in client_kwargs.items() if value is not None
        }
        return boto3.client(**clean_kwargs)

    def check_data_object(self, object_key: str) -> bool:
        return self.check_data_object_in_bucket(object_key, bucket_name=None)

    def check_data_object_in_bucket(
        self,
        object_key: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)
        try:
            self.client.head_object(Bucket=resolved_bucket_name, Key=object_key)
        except ClientError as error:
            if self._is_missing_object_error(error):
                raise FileNotFoundError(
                    f"Object '{object_key}' was not found in bucket '{resolved_bucket_name}'."
                ) from error
            if self._is_bad_request_error(error):
                raise ValueError(
                    self._build_bad_request_message(object_key, resolved_bucket_name, error)
                ) from error
            raise

        return True

    def download_data_files(
        self,
        bucket_directory: str,
        *,
        bucket_name: Optional[str] = None,
        parquet_filename: str = "data.parquet",
        json_filename: str = "metadata.json",
    ) -> tuple[BytesIO, dict[str, Any]]:
        parquet_key = self._build_object_key(bucket_directory, parquet_filename)
        json_key = self._build_object_key(bucket_directory, json_filename)
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)

        parquet_buffer = self._download_binary_object(parquet_key, bucket_name=resolved_bucket_name)
        json_payload = self._download_json_object(json_key, bucket_name=resolved_bucket_name)
        return parquet_buffer, json_payload

    def _download_binary_object(
        self,
        object_key: str,
        *,
        bucket_name: Optional[str] = None,
    ) -> BytesIO:
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)
        self.check_data_object_in_bucket(object_key, bucket_name=resolved_bucket_name)

        response = self.client.get_object(Bucket=resolved_bucket_name, Key=object_key)
        payload = response["Body"].read()
        buffer = BytesIO(payload)
        buffer.seek(0)
        return buffer

    def _download_json_object(
        self,
        object_key: str,
        *,
        bucket_name: Optional[str] = None,
    ) -> dict[str, Any]:
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)
        self.check_data_object_in_bucket(object_key, bucket_name=resolved_bucket_name)

        response = self.client.get_object(Bucket=resolved_bucket_name, Key=object_key)
        payload = response["Body"].read().decode("utf-8")
        return json.loads(payload)
    
    def upload_file(self, file_directory: str, bucket: str, object_key):
        self.client.upload_file(file_directory, bucket, object_key)
        print(f"Upload file: {file_directory} Done")

    @staticmethod
    def _build_object_key(bucket_directory: str, filename: str) -> str:
        normalized_directory = bucket_directory.strip("/")
        return f"{normalized_directory}/{filename}" if normalized_directory else filename

    @staticmethod
    def _is_missing_object_error(error: ClientError) -> bool:
        error_code = str(error.response.get("Error", {}).get("Code", ""))
        return error_code in {"404", "NoSuchKey", "NotFound"}

    @staticmethod
    def _is_bad_request_error(error: ClientError) -> bool:
        error_code = str(error.response.get("Error", {}).get("Code", ""))
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return error_code == "400" or status_code == 400

    def _build_bad_request_message(
        self,
        object_key: str,
        bucket_name: str,
        error: ClientError,
    ) -> str:
        endpoint = self.client.meta.endpoint_url
        parsed_endpoint = urlparse(endpoint) if endpoint else None
        message = (
            f"Received HTTP 400 from the S3 service while checking object '{object_key}' "
            f"in bucket '{bucket_name}'. Verify S3 endpoint: {self.settings.s3_endpoint}, bucket name: {bucket_name}, credentials, "
            "and S3-compatible client settings."
        )

        if parsed_endpoint and parsed_endpoint.hostname in {"localhost", "127.0.0.1"} and parsed_endpoint.port == 9001:
            message += (
                " The configured endpoint appears to use MinIO console port 9001; "
                "the S3 API usually listens on port 9000."
            )

        return message

    def _resolve_bucket_name(self, bucket_name: Optional[str]) -> str:
        resolved_bucket_name = (bucket_name or self.bucket_name or self.settings.s3_bucket_name or "").strip()
        if not resolved_bucket_name:
            raise ValueError("An S3 bucket name is required.")
        return resolved_bucket_name
