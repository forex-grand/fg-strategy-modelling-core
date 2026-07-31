# ForexGrand Strategy Modelling Core

ForexGrand Strategy Modelling Core is a Python package for building forex strategy modelling workflows. It includes utilities for loading market data from Cloudflare R2, generating technical indicators, preparing TFRecord datasets, training models, evaluating model quality, and packaging trained models for deployment.

The distribution name is `fg-strategy-modelling-core`; the Python import package is `forexgrand_core`.

## Features

- Cloudflare R2-backed storage access through the S3-compatible API.
- A runtime configuration helper so users do not have to manually export environment variables.
- Data loading and local caching for symbol market data.
- TensorFlow-based technical indicators for feature engineering.
- Dataset generation utilities for train, evaluation, and test workflows.
- Training pipelines for neural-network, KNN, XGBoost, and no-train target models.
- Model evaluation, performance checks, and model publishing helpers.
- Extensible preprocessing classes for custom feature pipelines.

## Installation

Install from PyPI:

```bash
pip install fg-strategy-modelling-core
```

Install from source for development:

```bash
git clone https://github.com/forexgrand/fg-strategy-modelling-core.git
cd fg-strategy-modelling-core
pip install -e ".[dev]"
```

## Configure Cloudflare R2

The package currently supports Cloudflare R2 storage. Configure it at the start of your script with `configure_r2`:

```python
from forexgrand_core import configure_r2

settings = configure_r2(
    account_id="your-cloudflare-account-id",
    access_key_id="your-r2-access-key-id",
    secret_access_key="your-r2-secret-access-key",
    bucket_name="forexgrand-data",
    train_bucket_name="forexgrand-train",
    eval_bucket_name="forexgrand-eval",
    test_bucket_name="forexgrand-test",
    model_upload_bucket="forexgrand-models",
)
```

You can pass `endpoint="https://<account-id>.r2.cloudflarestorage.com"` instead of `account_id` if you already have the full endpoint.

`configure_r2` sets these environment variables for the current Python process and returns a fresh `Settings` object:

| Variable | Purpose |
| --- | --- |
| `S3_STORAGE_OPTION` | Always set to `cloudflare` |
| `S3_ENDPOINT` | Cloudflare R2 S3 API endpoint |
| `S3_ACCESS_KEY` | R2 access key ID |
| `S3_SECRET_KEY` | R2 secret access key |
| `S3_REGION_NAME` | R2 region, defaults to `auto` |
| `S3_BUCKET_NAME` | Main data bucket |
| `TRAIN_BUCKET_NAME` | Training dataset bucket |
| `EVAL_BUCKET_NAME` | Evaluation dataset bucket |
| `TEST_BUCKET_NAME` | Test dataset bucket |
| `MODEL_UPLOAD_BUCKET` | Trained model upload bucket |

If bucket-specific names are omitted, `bucket_name` is reused for all buckets.

## Load Market Data

```python
from forexgrand_core import configure_r2
from forexgrand_core.data_manager import DataManager

configure_r2(
    account_id="your-cloudflare-account-id",
    access_key_id="your-r2-access-key-id",
    secret_access_key="your-r2-secret-access-key",
    bucket_name="forexgrand-data",
)

manager = DataManager(base_bucket_name="forexgrand-data")
df, properties = manager.load_data(
    symbol_pair="EURUSD",
    instrument_group="forex",
)
```

## Use Technical Indicators

```python
from forexgrand_core.indicators import tf_atr, tf_ma, tf_rsi

df["MA_20"] = tf_ma(df, period=20, column="close")
df["ATR_14"] = tf_atr(df, period=14)
df["RSI_14"] = tf_rsi(df, period=14, column="close")
```

Available indicators include moving average, slope, ATR, RSI, standard deviation, Bollinger Bands, Garman-Klass volatility, wick-to-range ratio, and normalization helpers.

## Train Models

```python
from forexgrand_core import configure_r2
from forexgrand_core.main import run_training
from forexgrand_core.pipeline.preprocessing.example_preprocessor import ExamplePreprocessor

configure_r2(
    account_id="your-cloudflare-account-id",
    access_key_id="your-r2-access-key-id",
    secret_access_key="your-r2-secret-access-key",
    bucket_name="forexgrand-data",
    model_upload_bucket="forexgrand-models",
)

results = run_training(
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    model_types=["conservative", "simple", "complex"],
    preprocessor_class=ExamplePreprocessor,
    sequence_length=60,
)
```

## Custom Preprocessing

Create your own preprocessing class by extending `PreprocessBase`:

```python
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase


class MyPreprocessor(PreprocessBase):
    def preprocess(self, inputs, training: bool = False):
        return inputs

    def features_metadata(self):
        return {}
```

Use the class in the training pipeline:

```python
results = run_training(
    symbols=["EURUSD"],
    model_types=["simple"],
    preprocessor_class=MyPreprocessor,
    sequence_length=60,
)
```

## Validate Configuration

Configuration is not validated on package import, so `import forexgrand_core` works before credentials are available. Validate explicitly when you want a clear setup error:

```python
from forexgrand_core.env_validator import validate_environment_on_import

validate_environment_on_import()
```

## Build And Publish To PyPI

Install build tools:

```bash
python -m pip install --upgrade build twine
```

Build the source distribution and wheel:

```bash
python -m build
```

Check the package:

```bash
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

## Development

Run tests:

```bash
pytest
```

Build locally:

```bash
python -m build
```

Check imports:

```python
import forexgrand_core
from forexgrand_core import configure_r2, Settings
```

## License

MIT
