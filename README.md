# ForexGrand Strategy Modelling Core

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A professional Python package for developing and training machine learning models on forex market data. This package provides a complete pipeline for data loading, feature engineering, model training, and performance evaluation.

## Features

- **Multi-backend Storage**: Support for AWS S3, MinIO, Google Cloud Storage, and Cloudflare R2
- **Technical Indicators**: Pre-built indicators (MA, ATR, RSI, Bollinger Bands, etc.)
- **Scalable Architecture**: Modular design for custom preprocessing and model architectures
- **Production Ready**: Comprehensive error handling, logging, and validation
- **Type Hints**: Full type annotations for IDE support and type checking
- **Environment-driven Configuration**: All settings via environment variables
- **TensorFlow Integration**: Native TFRecord support for efficient data loading

## Installation

### From PyPI (Recommended)

```bash
pip install fg-strategy-modelling-core
```

### From Source

```bash
git clone https://github.com/forexgrand/fg-strategy-modelling-core.git
cd fg-strategy-modelling-core
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Setup Environment Variables

Create a `.env` file or export environment variables:

```bash
# Data source configuration
export DATA_SOURCE="metaquotes"
export FORCE_RELOAD="false"

# Storage configuration (MinIO/S3)
export S3_STORAGE_OPTION="minio"  # Options: minio, aws, gcs, cloudflare
export S3_ENDPOINT="http://localhost:9000"
export S3_ACCESS_KEY="minio"
export S3_SECRET_KEY="minioadmin"
export S3_BUCKET_NAME="forexgrand"

# Bucket names
export TRAIN_BUCKET_NAME="forexgrand-train"
export EVAL_BUCKET_NAME="forexgrand-eval"
export TEST_BUCKET_NAME="forexgrand-test"
export MODEL_UPLOAD_BUCKET="forexgrand-models"

# Training configuration
export BATCH_SIZE="64"
export EPOCHS="50"
export LEARNING_RATE="0.001"
export SEQUENCE_STRIDE="100"

# Evaluation benchmarks
export EVAL_MIN_PRECISION="0.55"
export EVAL_MIN_RECALL="0.4"
export MAX_OVERFIT_GAP="0.2"
```

### 2. Load and Process Data

```python
from src.data_manager import DataManager
from src.indicators import TensorFlowIndicators

# Initialize data manager
manager = DataManager(base_bucket_name="forexgrand-train")

# Load market data
df, properties = manager.load_data(
    symbol_pair="EURUSD",
    instrument_group="forex"
)

print(f"Loaded {len(df)} rows of data")
print(f"Columns: {list(df.columns)}")
```

### 3. Add Technical Indicators

```python
from src.indicators import tf_ma, tf_atr, tf_rsi

# Calculate Moving Average
df['MA_20'] = tf_ma(df, period=20, column='close')

# Calculate ATR
df['ATR_14'] = tf_atr(df, period=14)

# Calculate RSI
df['RSI_14'] = tf_rsi(df, period=14, column='close')

print(df[['close', 'MA_20', 'ATR_14', 'RSI_14']].head())
```

### 4. Train Models

```python
from src.main import run_training
from src.pipeline.preprocessing.example_preprocessor import ExamplePreprocessor

# Run training pipeline
results = run_training(
    symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
    model_types=['conservative', 'simple', 'complex'],
    preprocessor_class=ExamplePreprocessor,
    sequence_length=60,
)

# Process results
for result in results:
    print(f"{result.symbol} - {result.model_type}:")
    print(f"  Model Path: {result.model_path}")
    print(f"  Accuracy: {result.accuracy:.4f}")
```

## Configuration Reference

### Environment Variables

#### Data Source & Storage

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `DATA_SOURCE` | `mt5` | `mt5`, other | Data source type |
| `FORCE_RELOAD` | `false` | `true`, `false` | Force reload from remote storage |
| `S3_STORAGE_OPTION` | `minio` | `minio`, `aws`, `gcs`, `cloudflare` | Storage backend |
| `S3_ENDPOINT` | `http://localhost:9000` | URL | S3/MinIO endpoint |
| `S3_ACCESS_KEY` | `minio` | string | Storage access key |
| `S3_SECRET_KEY` | `minioadmin` | string | Storage secret key |
| `S3_REGION_NAME` | `us-east-1` | string | AWS region name |
| `S3_SESSION_TOKEN` | (optional) | string | AWS temporary credentials |
| `S3_BUCKET_NAME` | `forexgrand` | string | Main bucket name |
| `DATA_DIRECTORY` | `../../data` | path | Local cache directory |

#### Bucket Names

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_BUCKET_NAME` | `forexgrand-train` | Training data bucket |
| `EVAL_BUCKET_NAME` | `forexgrand-eval` | Evaluation data bucket |
| `TEST_BUCKET_NAME` | `forexgrand-test` | Test data bucket |
| `MODEL_UPLOAD_BUCKET` | `forexgrand-models` | Trained models bucket |

#### Model Training

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `BATCH_SIZE` | `64` | 1-1000 | Training batch size |
| `EPOCHS` | `50` | 1-1000 | Number of training epochs |
| `LEARNING_RATE` | `0.001` | 0.00001-0.1 | Model learning rate |
| `SEQUENCE_STRIDE` | `100` | 1-1000 | Data generation stride |
| `SHUFFLE_TRAIN_DATA` | `true` | `true`, `false` | Shuffle training data |
| `SHUFFLE_BUFFER_SIZE` | `10000` | 100-1000000 | Shuffle buffer size |
| `STEPS_PER_EPOCH` | `1000` | 1-10000 | Steps per training epoch |

#### Evaluation & Benchmarks

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `EVAL_MIN_PRECISION` | `0.55` | 0.0-1.0 | Minimum precision threshold |
| `EVAL_MIN_RECALL` | `0.4` | 0.0-1.0 | Minimum recall threshold |
| `MAX_OVERFIT_GAP` | `0.2` | 0.0-1.0 | Max allowed overfit gap |
| `TF_RECORD_COMPRESSION` | `GZIP` | `GZIP`, `ZSTD` | TFRecord compression |
| `PERFORMANCE_BASE_URL` | `http://localhost:8002` | URL | Performance service URL |

## API Reference

### DataManager

Load market data from S3-compatible storage with local caching.

```python
from src.data_manager import DataManager

manager = DataManager(base_bucket_name="forexgrand-train")
df, properties = manager.load_data("EURUSD", "forex")
```

**Methods:**
- `load_data(symbol_pair: str, instrument_group: str) -> tuple[DataFrame, dict]`
  - Loads M1 forex data. Returns DataFrame and symbol properties.
  - Automatically caches data locally unless `FORCE_RELOAD=true`

### Indicators

Technical analysis indicators for feature engineering.

```python
from src import indicators

# Simple Moving Average
ma_20 = indicators.tf_ma(df, period=20, column='close')

# Average True Range (volatility)
atr_14 = indicators.tf_atr(df, period=14)

# Relative Strength Index (momentum)
rsi_14 = indicators.tf_rsi(df, period=14, column='close')

# Bollinger Bands
bands = indicators.tf_bollinger_bands(df, period=20, deviation=2.0)

# Factory functions (for fixed parameters)
ma_factory = indicators.ma_factory(period=20)
slope_factory = indicators.slope_factory(period=14)
```

**Available Indicators:**
- `tf_ma()` - Simple Moving Average
- `tf_slope()` - Linear Regression Slope
- `tf_atr()` - Average True Range
- `tf_rsi()` - Relative Strength Index
- `tf_stdev()` - Standard Deviation
- `tf_bollinger_bands()` - Bollinger Bands
- `tf_german_klass_volatility()` - Garman-Klass Volatility
- `tf_wick_bar_range_ratio()` - Wick-to-Range Ratio
- `tf_normalize_feature()` - Quantile-based Normalization

### Training Pipeline

```python
from src.main import run_training

results = run_training(
    symbols=['EURUSD', 'GBPUSD'],
    model_types=['conservative', 'simple', 'complex'],
    preprocessor_class=CustomPreprocessor,
    sequence_length=60,
)
```

## Custom Preprocessing

Create custom preprocessing pipelines by extending `PreprocessBase`:

```python
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase

class MyPreprocessor(PreprocessBase):
    def process(self, df):
        # Add custom feature engineering
        df['custom_feature'] = ...
        return df

# Use in training
results = run_training(
    symbols=['EURUSD'],
    model_types=['simple'],
    preprocessor_class=MyPreprocessor,
)
```

## Model Types

Supported model architectures:
- `conservative` - Simple CNN model (low compute)
- `simple` - LSTM-based model
- `complex` - Ensemble architecture
- `lstm` - Pure LSTM model
- `cnn_bi_lstm` - Bidirectional CNN-LSTM hybrid

## Troubleshooting

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Storage Connection Issues

Verify environment variables:
```bash
echo $S3_ENDPOINT
echo $S3_ACCESS_KEY
echo $S3_BUCKET_NAME
```

### Data Loading Failures

Check local cache directory exists:
```bash
mkdir -p $DATA_DIRECTORY
```

Enable debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Memory Issues

Reduce batch size:
```bash
export BATCH_SIZE=32
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run tests with coverage:

```bash
pytest --cov=src tests/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions:
- 📧 Email: dev@forexgrand.com
- 🐛 Issues: [GitHub Issues](https://github.com/forexgrand/fg-strategy-modelling-core/issues)
- 📚 Documentation: [Full Docs](https://fg-strategy-modelling-core.readthedocs.io)

## Changelog

### [0.1.0] - 2024-04-29

**Initial Release**
- Core data loading and caching
- Technical indicators module
- Training pipeline
- Multi-backend storage support
- Comprehensive documentation

