from enum import Enum
from typing import Type

from src.storage.base_client import BaseStorageClient
from src.storage.clients import (
    AWSStorageClient,
    CloudflareR2StorageClient,
    GoogleS3StorageClient,
    MinIOStorageClient,
)


class StorageOptionEnumeration(str, Enum):
    AWS = "aws"
    CLOUDFLARE = "cloudflare"
    MINIO = "minio"
    GOOGLE_S3 = "google_s3"


storage_option_enumeration = StorageOptionEnumeration


def getStorageClient(storage_option: str) -> Type[BaseStorageClient] | None:
    normalized_option = (storage_option or "").strip().lower()
    mapping: dict[str, Type[BaseStorageClient]] = {
        StorageOptionEnumeration.AWS.value: AWSStorageClient,
        StorageOptionEnumeration.CLOUDFLARE.value: CloudflareR2StorageClient,
        StorageOptionEnumeration.MINIO.value: MinIOStorageClient,
        StorageOptionEnumeration.GOOGLE_S3.value: GoogleS3StorageClient,
    }
    return mapping.get(normalized_option)
