"""Environment variable validation and validation utilities.

This module provides utilities for validating environment configuration and
raising helpful errors when required environment variables are missing or invalid.
"""

import os
from typing import Optional, Set


class EnvironmentValidator:
    """Validates environment variables for the package."""
    
    REQUIRED_FOR_STORAGE = {
        "cloudflare": {"S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY"},
    }
    
    @staticmethod
    def validate_storage_config(storage_option: str) -> None:
        """Validate storage configuration based on storage option.
        
        Args:
            storage_option: Storage backend type. Currently only cloudflare is supported.
            
        Raises:
            EnvironmentError: If required environment variables are missing for the
                selected storage option.
        """
        storage_option = (storage_option or "").strip().lower() or "minio"
        
        if storage_option not in EnvironmentValidator.REQUIRED_FOR_STORAGE:
            supported = ", ".join(EnvironmentValidator.REQUIRED_FOR_STORAGE.keys())
            raise EnvironmentError(
                f"Invalid storage option '{storage_option}'. "
                f"Supported options: {supported}.\n"
                f"Use configure_r2(...) or set S3_STORAGE_OPTION=cloudflare."
            )
        
        required = EnvironmentValidator.REQUIRED_FOR_STORAGE[storage_option]
        missing = set()
        
        for var in required:
            if not os.getenv(var):
                missing.add(var)
        
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise EnvironmentError(
                f"Storage option '{storage_option}' requires these environment variables: "
                f"{missing_list}\n"
                f"Use forexgrand_core.configure_r2(...) or set them before using storage."
            )
    
    @staticmethod
    def validate_data_source() -> None:
        """Validate data source configuration.
        
        Raises:
            EnvironmentError: If DATA_SOURCE is not configured.
        """
        data_source = os.getenv("DATA_SOURCE", "").strip()
        if not data_source:
            raise EnvironmentError(
                "DATA_SOURCE environment variable is not set or empty.\n"
                "Set it to your data source type (e.g., 'mt5')."
            )
    
    @staticmethod
    def validate_bucket_names() -> None:
        """Validate bucket name configuration.
        
        Raises:
            EnvironmentError: If required bucket names are not configured.
        """
        required_buckets = {
            "S3_BUCKET_NAME": "Main data bucket",
            "TRAIN_BUCKET_NAME": "Training data bucket",
            "EVAL_BUCKET_NAME": "Evaluation data bucket",
        }
        
        missing = []
        for var, desc in required_buckets.items():
            if not os.getenv(var):
                missing.append(f"{var} ({desc})")
        
        if missing:
            missing_str = "\n  ".join(missing)
            raise EnvironmentError(
                f"Required bucket names not configured:\n  {missing_str}\n"
                f"These variables should be set in your environment."
            )
    
    @staticmethod
    def log_config() -> None:
        """Log current configuration for debugging.
        
        Note: Does not log sensitive values like passwords/keys.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        safe_vars = {
            "DATA_SOURCE": os.getenv("DATA_SOURCE"),
            "S3_STORAGE_OPTION": os.getenv("S3_STORAGE_OPTION"),
            "S3_ENDPOINT": os.getenv("S3_ENDPOINT"),
            "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME"),
            "TRAIN_BUCKET_NAME": os.getenv("TRAIN_BUCKET_NAME"),
            "EVAL_BUCKET_NAME": os.getenv("EVAL_BUCKET_NAME"),
            "BATCH_SIZE": os.getenv("BATCH_SIZE"),
            "EPOCHS": os.getenv("EPOCHS"),
            "LEARNING_RATE": os.getenv("LEARNING_RATE"),
        }
        
        logger.debug("Configuration loaded:")
        for key, value in safe_vars.items():
            logger.debug(f"  {key}: {value}")


def validate_environment_on_import() -> None:
    """Validate critical environment variables on package import.
    
    This function is called when the package is imported to ensure
    all critical configuration is in place.
    
    Raises:
        EnvironmentError: If critical environment variables are missing or invalid.
    """
    try:
        # Always validate data source
        EnvironmentValidator.validate_data_source()
        
        # Validate storage configuration
        storage_option = os.getenv("S3_STORAGE_OPTION", "cloudflare")
        EnvironmentValidator.validate_storage_config(storage_option)
        
        # Validate bucket names
        EnvironmentValidator.validate_bucket_names()
        
        # Log configuration
        EnvironmentValidator.log_config()
        
    except EnvironmentError as e:
        # Re-raise with additional context
        raise EnvironmentError(
            f"Environment validation failed:\n{str(e)}\n\n"
            f"Please refer to README.md for configuration instructions."
        ) from e
