from __future__ import annotations

from typing import Optional

from forexgrand_core.settings import Settings
from forexgrand_core.storage.base_client import BaseStorageClient


class GoogleS3StorageClient(BaseStorageClient):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        resolved_settings = settings or Settings()
        super().__init__(
            settings=resolved_settings,
            endpoint_url=resolved_settings.s3_endpoint,
        )
