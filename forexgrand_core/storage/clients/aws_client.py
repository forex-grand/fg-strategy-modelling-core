from __future__ import annotations

from typing import Optional

from forexgrand_core.settings import Settings
from forexgrand_core.storage.base_client import BaseStorageClient


class AWSStorageClient(BaseStorageClient):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__(settings=settings, endpoint_url=None)
