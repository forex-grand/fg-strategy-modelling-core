"""Configuration settings for ForexGrand strategy modelling core.

This module provides the Settings dataclass that loads configuration from environment variables.
All settings have sensible defaults for local development.

Environment Variables:
    DATA_SOURCE: Data source type (default: 'mt5'). Options: 'mt5', 'other'
    FORCE_RELOAD: Force reload data from storage (default: 'false')
    S3_STORAGE_OPTION: Storage backend (default: 'minio'). Options: 'minio', 'aws', 'gcs', 'cloudflare'
    S3_ENDPOINT: S3/MinIO endpoint URL (default: 'http://localhost:9000')
    S3_ACCESS_KEY: S3 access key (default: 'minio')
    S3_SECRET_KEY: S3 secret key (default: 'minioadmin')
    S3_REGION_NAME: AWS region (default: 'us-east-1')
    S3_SESSION_TOKEN: AWS session token for temporary credentials (optional)
    S3_BUCKET_NAME: Main bucket name (default: 'forexgrand')
    DATA_DIRECTORY: Local data cache directory (default: '../../data')
    TRAIN_BUCKET_NAME: Training data bucket (default: 'forexgrand-train')
    EVAL_BUCKET_NAME: Evaluation data bucket (default: 'forexgrand-eval')
    TEST_BUCKET_NAME: Test data bucket (default: 'forexgrand-test')
    TF_RECORD_COMPRESSION: TFRecord compression type (default: 'GZIP'). Options: 'GZIP', 'ZSTD'
    SEQUENCE_STRIDE: Data generation stride (default: 100)
    BATCH_SIZE: Training batch size (default: 64)
    SHUFFLE_TRAIN_DATA: Shuffle training data (default: True)
    SHUFFLE_BUFFER_SIZE: Shuffle buffer size (default: 10000)
    LEARNING_RATE: Model learning rate (default: 1e-3)
    EPOCHS: Number of training epochs (default: 50)
    STEPS_PER_EPOCH: Steps per epoch (default: 1000)
    EVAL_MIN_PRECISION: Minimum precision threshold (default: 0.55)
    EVAL_MIN_RECALL: Minimum recall threshold (default: 0.4)
    MAX_OVERFIT_GAP: Maximum overfitting gap (default: 0.2)
    MODEL_UPLOAD_BUCKET: Model upload bucket (default: 'forexgrand-models')
    PERFORMANCE_BASE_URL: Performance testing service URL (default: 'http://localhost:8002')
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """Configuration settings loaded from environment variables.
    
    This class provides typed access to all configuration settings used throughout
    the ForexGrand strategy modelling core. Settings are loaded from environment
    variables with sensible defaults for local development.
    
    Attributes:
        data_source: Data source type (e.g., 'mt5')
        force_reload: Whether to force reload data from remote storage
        s3_storage_option: Which S3-compatible storage backend to use
        s3_endpoint: URL endpoint for S3/MinIO storage
        s3_access_key: Access key for S3 authentication
        s3_secret_key: Secret key for S3 authentication
        s3_region_name: AWS region name
        s3_session_token: Temporary session token for AWS access
        s3_bucket_name: Main S3 bucket name
        data_directory: Local directory for caching downloaded data
        train_bucket_name: Bucket for training data
        eval_bucket_name: Bucket for evaluation data
        test_bucket_name: Bucket for test data
        tf_record_compression_type: Compression type for TFRecord files
        generated_data_strides: Stride for data generation sequences
        batch_size: Batch size for model training
        shuffle_data: Whether to shuffle training data
        shuffle_buffer_size: Size of shuffle buffer
        learning_rate: Learning rate for model optimization
        epochs: Number of training epochs
        steps_per_epoch: Steps per training epoch
        eval_min_precision: Minimum precision benchmark for evaluation
        eval_min_recall: Minimum recall benchmark for evaluation
        eval_max_overfit_gap: Maximum allowed overfitting gap
        models_bucket: Bucket for storing trained models
        performance_base_url: URL for performance testing service
    """
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
        default_factory=lambda: os.getenv("S3_ENDPOINT", "http://localhost:9000")
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
    train_bucket_name: Optional[str] = field(
        default_factory=lambda: os.getenv("TRAIN_BUCKET_NAME", "forexgrand-train")
    )
    eval_bucket_name: Optional[str] = field(
        default_factory=lambda: os.getenv("EVAL_BUCKET_NAME", "forexgrand-eval")
    )
    test_bucket_name: Optional[str] = field(
        default_factory=lambda: os.getenv("TEST_BUCKET_NAME", "forexgrand-test")
    )
    tf_record_compression_type: Optional[str] = field(
        default_factory=lambda: os.getenv("TF_RECORD_COMPRESSION", "GZIP")
    )

    ##Model Training settings section
    generated_data_strides: Optional[int] = field(
        default_factory=lambda: int(os.getenv("SEQUENCE_STRIDE", 100))
    )
    batch_size: Optional[int] = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", 64))
    )
    shuffle_data: Optional[bool] = field(
        default_factory=lambda: bool(os.getenv("SHUFFLE_TRAIN_DATA", True))
    )
    shuffle_buffer_size: Optional[int] = field(
        default_factory=lambda: int(os.getenv("SHUFFLE_BUFFER_SIZE",10000))
    )
    learning_rate: Optional[float] = field(
        default_factory=lambda: float(os.getenv("LEARNING_RATE", 1e-3))
    )
    epochs: Optional[int] = field(
        default_factory=lambda: int(os.getenv("EPOCHS", 50))
    )
    steps_per_epoch: Optional[int] = field(
        default_factory=lambda: int(os.getenv('STEPS_PER_EPOCH', 1000))
    )
    ##evaluation benchmarks
    eval_min_precision: Optional[float] = field(
        default_factory=lambda: float(os.getenv('EVAL_MIN_PRECISION', 0.55))
    )
    eval_min_recall: Optional[float] = field(
        default_factory=lambda: float(os.getenv('EVAL_MIN_RECALL', 0.4))
    )
    eval_max_overfit_gap: Optional[float] = field(
        default_factory=lambda: float(os.getenv('MAX_OVERFIT_GAP', 0.2))
    )

    models_bucket: Optional[str] = field(
        default_factory=lambda: os.getenv("MODEL_UPLOAD_BUCKET","forexgrand-models")
    )
    ##URL for performance testing service integration
    performance_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("PERFORMANCE_BASE_URL", "http://localhost:8002")
    )
    test_bucket_name: Optional[str] = field(
        default_factory=lambda: os.getenv("TEST_BUCKET_NAME", "forexgrand-test")
    )
    