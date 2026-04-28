from __future__ import annotations

from typing import Optional

from src.settings import Settings
from src.storage.base_client import BaseStorageClient
from boto3 import client
from botocore.config import Config

class CloudflareR2StorageClient(BaseStorageClient):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        resolved_settings = settings or Settings()
        super().__init__(
            settings=resolved_settings,
            endpoint_url=resolved_settings.s3_endpoint,
        )
        self.client = client(
            "s3",
            endpoint_url=resolved_settings.s3_endpoint,
            aws_access_key_id=resolved_settings.s3_access_key,
            aws_secret_access_key=resolved_settings.s3_secret_key,
            region_name=resolved_settings.s3_region_name,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
