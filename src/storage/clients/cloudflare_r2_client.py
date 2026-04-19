from __future__ import annotations

from typing import Optional

from src.settings import Settings
from src.storage.base_client import BaseStorageClient


class CloudflareR2StorageClient(BaseStorageClient):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        resolved_settings = settings or Settings()
        super().__init__(
            settings=resolved_settings,
            endpoint_url=resolved_settings.s3_endpoint,
        )
