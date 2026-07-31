"""Runtime configuration helpers for ForexGrand Core."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from forexgrand_core.settings import Settings


@dataclass(frozen=True)
class R2Config:
    """Cloudflare R2 configuration values used by the package."""

    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    train_bucket_name: str
    eval_bucket_name: str
    test_bucket_name: str
    model_upload_bucket: str
    region_name: str = "auto"
    data_source: str = "mt5"
    force_reload: bool = False
    data_directory: Optional[str] = None


def build_r2_endpoint(account_id: str) -> str:
    """Build the standard Cloudflare R2 S3 API endpoint for an account ID."""
    clean_account_id = account_id.strip()
    if not clean_account_id:
        raise ValueError("account_id cannot be empty.")
    return f"https://{clean_account_id}.r2.cloudflarestorage.com"


def configure_r2(
    *,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
    account_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    train_bucket_name: Optional[str] = None,
    eval_bucket_name: Optional[str] = None,
    test_bucket_name: Optional[str] = None,
    model_upload_bucket: Optional[str] = None,
    region_name: str = "auto",
    data_source: str = "mt5",
    force_reload: bool = False,
    data_directory: Optional[str] = None,
) -> Settings:
    """Configure the package to use Cloudflare R2 storage.

    The helper writes the environment variables consumed by ``Settings`` and the
    storage clients, then returns a fresh ``Settings`` instance.

    Provide either ``account_id`` or ``endpoint``. If bucket-specific names are
    omitted, ``bucket_name`` is reused for training, evaluation, test, and model
    upload storage.
    """
    resolved_endpoint = endpoint.strip() if endpoint else None
    if not resolved_endpoint:
        if not account_id:
            raise ValueError("configure_r2 requires either account_id or endpoint.")
        resolved_endpoint = build_r2_endpoint(account_id)

    values = {
        "DATA_SOURCE": data_source,
        "FORCE_RELOAD": str(force_reload).lower(),
        "S3_STORAGE_OPTION": "cloudflare",
        "S3_ENDPOINT": resolved_endpoint,
        "S3_ACCESS_KEY": access_key_id,
        "S3_SECRET_KEY": secret_access_key,
        "S3_REGION_NAME": region_name,
        "S3_BUCKET_NAME": bucket_name,
        "TRAIN_BUCKET_NAME": train_bucket_name or bucket_name,
        "EVAL_BUCKET_NAME": eval_bucket_name or bucket_name,
        "TEST_BUCKET_NAME": test_bucket_name or bucket_name,
        "MODEL_UPLOAD_BUCKET": model_upload_bucket or bucket_name,
    }
    if data_directory is not None:
        values["DATA_DIRECTORY"] = data_directory

    missing = [key for key, value in values.items() if value is None or str(value).strip() == ""]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required R2 configuration values: {missing_list}")

    os.environ.update(values)
    return Settings()


__all__ = ["R2Config", "build_r2_endpoint", "configure_r2"]
