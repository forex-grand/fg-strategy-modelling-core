import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    data_source: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_SOURCE", "mt5")
    )
    force_reload: bool = field(
        default_factory=lambda: os.getenv("FORCE_RELOAD", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    s3_storage_option: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_STORAGE_OPTION", "minio")
    )
    s3_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_ENDPOINT", "http://localhost:9001")
    )
    s3_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_ACCESS_KEY", "minio")
    )
    s3_secret_key: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_SECRET_KEY", "minioadmin")
    )
    s3_region_name: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_REGION_NAME", "us-east-1")
    )
    s3_session_token: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_SESSION_TOKEN")
    )
    s3_bucket_name: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_BUCKET_NAME", "forexgrand")
    )
    data_directory: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_DIRECTORY", "../../data")
    )
