from enum import Enum


class StorageOptionEnumeration(str, Enum):
    AWS = "aws"
    CLOUDFLARE_R2 = "cloudflare_r2"
    MINIO = "minio"
    GOOGLE_S3 = "google_s3"


storage_option_enumeration = StorageOptionEnumeration
