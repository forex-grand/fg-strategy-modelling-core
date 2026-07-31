# Getting Started with ForexGrand Strategy Modelling Core

This guide will help you get started with the ForexGrand package in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip or conda
- S3-compatible storage (MinIO for local development, or AWS S3)

## 1. Installation

### Install from PyPI (Once Published)
```bash
pip install fg-strategy-modelling-core
```

### Install from Source (Development)
```bash
git clone https://github.com/forexgrand/fg-strategy-modelling-core.git
cd fg-strategy-modelling-core
pip install -e .
```

### With Optional Dependencies
```bash
# For AWS S3 support
pip install fg-strategy-modelling-core[aws]

# For Google Cloud Storage
pip install fg-strategy-modelling-core[gcs]

# For development/testing
pip install fg-strategy-modelling-core[dev]
```

## 2. Configuration

### Option A: Set Environment Variables

```bash
export DATA_SOURCE="mt5"
export S3_STORAGE_OPTION="minio"
export S3_ENDPOINT="http://localhost:9000"
export S3_ACCESS_KEY="minio"
export S3_SECRET_KEY="minioadmin"
export S3_BUCKET_NAME="forexgrand"
export TRAIN_BUCKET_NAME="forexgrand-train"
export EVAL_BUCKET_NAME="forexgrand-eval"
```

### Option B: Copy and Modify .env File

```bash
cp .env.example .env
# Edit .env with your settings
# Then source it: source .env
```

## 3. First Steps

### Access Market Data

```python
from forexgrand_core.data_manager import DataManager

# Initialize data manager
manager = DataManager(base_bucket_name="forexgrand-train")

# Load data for a currency pair
df, properties = manager.load_data(
    symbol_pair="EURUSD",
    instrument_group="forex"
)

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(df.head())
```

### Compute Technical Indicators

```python
from forexgrand_core.indicators import tf_ma, tf_atr, tf_rsi

# Add indicators to your DataFrame
df['MA_20'] = tf_ma(df, period=20, column='close')
df['ATR_14'] = tf_atr(df, period=14)
df['RSI_14'] = tf_rsi(df, period=14, column='close')

# View the results
print(df[['close', 'MA_20', 'ATR_14', 'RSI_14']].head(20))
```

### Train Models

```python
from forexgrand_core.main import run_training

# Train models
results = run_training(
    symbols=['EURUSD', 'GBPUSD'],
    model_types=['simple', 'conservative'],
    sequence_length=60,
)

# Analyze results
for result in results:
    print(f"{result.symbol} - {result.model_type}:")
    print(f"  Accuracy: {result.accuracy}")
    print(f"  Precision: {result.precision}")
    print(f"  Model saved: {result.model_path}")
```

## 4. Available Indicators

All indicators work with pandas DataFrames containing OHLCV data.

### Basic Indicators

```python
from forexgrand_core import indicators

# Moving Average
ma = indicators.tf_ma(df, period=20, column='close')

# Average True Range (volatility)
atr = indicators.tf_atr(df, period=14)

# Relative Strength Index (momentum)
rsi = indicators.tf_rsi(df, period=14, column='close')

# Standard Deviation
stdev = indicators.tf_stdev(df, period=20, column='close')

# Slope (linear regression)
slope = indicators.tf_slope(df, period=20, column='close')
```

### Complex Indicators

```python
# Bollinger Bands
bands = indicators.tf_bollinger_bands(df, period=20, deviation=2.0)
print(bands.head())  # Returns DataFrame with 'upper', 'middle', 'lower'

# German-Klass Volatility
gk_vol = indicators.tf_german_klass_volatility(df, period=14)

# Wick-to-Range Ratio
wick_ratio = indicators.tf_wick_bar_range_ratio(df)

# Normalize Feature
normalized = indicators.tf_normalize_feature(df, 'close', lower_quantile=0.01, upper_quantile=0.99)
```

### Factory Functions

Use factory functions to create indicators with fixed parameters:

```python
# Create MA function with fixed period
ma_20 = indicators.ma_factory(period=20, column='close')
ma_values = ma_20(df)

# Create slope function
slope_14 = indicators.slope_factory(period=14)
slope_values = slope_14(df)

# Create ATR function
atr_14 = indicators.atr_factory(period=14)
atr_values = atr_14(df)
```

## 5. Configuration Reference

### Storage Backends

#### MinIO (Local Development)
```bash
export S3_STORAGE_OPTION="minio"
export S3_ENDPOINT="http://localhost:9000"
export S3_ACCESS_KEY="minio"
export S3_SECRET_KEY="minioadmin"
```

#### AWS S3
```bash
export S3_STORAGE_OPTION="aws"
export S3_REGION_NAME="us-east-1"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

#### Google Cloud Storage
```bash
export S3_STORAGE_OPTION="gcs"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Cloudflare R2
```bash
export S3_STORAGE_OPTION="cloudflare"
export S3_ENDPOINT="https://your-account.r2.cloudflarestorage.com"
export S3_ACCESS_KEY="your-key"
export S3_SECRET_KEY="your-secret"
```

### Training Configuration

```bash
# Model training
export BATCH_SIZE="64"
export EPOCHS="50"
export LEARNING_RATE="0.001"
export SEQUENCE_STRIDE="100"

# Data handling
export SHUFFLE_TRAIN_DATA="true"
export SHUFFLE_BUFFER_SIZE="10000"

# Evaluation
export EVAL_MIN_PRECISION="0.55"
export EVAL_MIN_RECALL="0.4"
export MAX_OVERFIT_GAP="0.2"
```

## 6. Troubleshooting

### ImportError: No module named 'tensorflow'

Install TensorFlow:
```bash
pip install tensorflow>=2.16.0
```

### EnvironmentError: DATA_SOURCE not set

Set required environment variables:
```bash
export DATA_SOURCE="mt5"
export S3_STORAGE_OPTION="minio"
# ... see Configuration section above
```

### Cannot connect to storage

Check your storage credentials:
```python
from forexgrand_core.settings import Settings
s = Settings()
print(f"Endpoint: {s.s3_endpoint}")
print(f"Bucket: {s.s3_bucket_name}")
print(f"Region: {s.s3_region_name}")
```

### Module not found when importing

Ensure package is installed and in Python path:
```bash
pip install -e .  # Install in development mode
python -c "import src; print(forexgrand_core.__version__)"
```

## 7. Next Steps

- 📖 Read [README.md](README.md) for complete API reference
- 🔧 Check [PUBLISHING.md](PUBLISHING.md) for deployment information
- 🧪 Run tests: `pytest tests/` (requires test setup)
- 📚 Explore examples in the `examples/` directory
- 🤝 Contribute improvements on [GitHub](https://github.com/forexgrand/fg-strategy-modelling-core)

## 8. Common Tasks

### Load Multiple Symbols

```python
from forexgrand_core.data_manager import DataManager

manager = DataManager()

symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
dataframes = {}

for symbol in symbols:
    df, props = manager.load_data(symbol, 'forex')
    dataframes[symbol] = df
    print(f"{symbol}: {len(df)} rows loaded")
```

### Create Custom Preprocessing

```python
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase

class CustomPreprocessor(PreprocessBase):
    def process(self, df):
        # Your custom feature engineering
        df['custom_feature'] = df['high'] - df['low']
        return df

from forexgrand_core.main import run_training

results = run_training(
    symbols=['EURUSD'],
    model_types=['simple'],
    preprocessor_class=CustomPreprocessor,
)
```

### Batch Process Multiple Symbols

```python
from forexgrand_core.main import run_training

symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']
models = ['conservative', 'simple']

results = run_training(
    symbols=symbols,
    model_types=models,
    sequence_length=60,
)

# Summary
print(f"Trained {len(results)} models")
for r in results:
    print(f"{r.symbol:8s} {r.model_type:12s} accuracy: {r.accuracy:.4f}")
```

## Support & Help

- **Documentation**: See README.md and PUBLISHING.md
- **API Reference**: Check docstrings with `help(function_name)`
- **Issues**: Report bugs on GitHub
- **Email**: dev@forexgrand.com

Happy modeling! 🚀
