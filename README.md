# ForexGrand Strategy Modelling Core

ForexGrand Strategy Modelling Core is a Python package for building forex strategy modelling workflows. It includes utilities for loading market data from Cloudflare R2, preparing TFRecord datasets, training models, evaluating model quality, and packaging trained models for deployment.

The distribution name is `fg-strategy-modelling-core`; the Python import package is `forexgrand_core`.

## Features

- Cloudflare R2-backed storage access through the S3-compatible API.
- A runtime configuration helper so users do not have to manually export environment variables.
- Data loading and local caching for symbol market data.
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

## Command Line Interface

Installing the package provides an `fg_core` command with data workflow subcommands:

```bash
fg_core download_data EURUSD forex --bucket forexgrand-data --source mt5
fg_core generate_train_data EURUSD forex --sequence-length 2800 --stride 100
fg_core preprocess_data prices.parquet preprocess.py --output data/processed.pkl.gz
```

`preprocess.py` must define `preprocess_fn(dataframe)`. The input can be CSV, Parquet,
or a pickle file, and the preprocessing result is saved as a gzip-compressed pickle.
Each command prints a JSON object containing its output path and summary information.
Storage credentials and other runtime settings use the same environment variables as
the Python API.

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

## Generate Training Data

```python
import tensorflow as tf
from forexgrand_core import configure_r2
from forexgrand_core.pipeline.no_train_trainer import NoTrainTrainer
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.schemas import SymbolIn, TimeBasedTarget


class Preprocess(PreprocessBase):
    def preprocess(self, data, training=False):
        return {"direction": tf.zeros(tf.shape(data["close"])[0], dtype=tf.int64)}

    def features_metadata(self):
        return {"direction": tf.io.FixedLenFeature([], tf.int64)}

configure_r2(
    account_id="your-cloudflare-account-id",
    access_key_id="your-r2-access-key-id",
    secret_access_key="your-r2-secret-access-key",
    bucket_name="forexgrand-data",
)

trainer = NoTrainTrainer(
    symbols=[SymbolIn(symbol="EURUSD", group="forex")],
    sequence_length=2800,
    preprocessor_class=Preprocess,
    target_model_type=TimeBasedTarget(stop_minutes=60, mode="prices"),
    run_performance_test=False,
    hot_reload_data=False,
    upload_models=True,
    target_percentile=99,
    use_dataframe_format=False,
)
results = trainer.run()
```

For data generation APIs, pass `source` explicitly when the data is not under
the `DATA_SOURCE` environment value:

```python
data_gen.load_single_data(..., source="metaquotes")
data_gen.load_data(..., source="metaquotes")
```

## Backtest A Strategy

Backtesting currently uses the Python API only. The CLI backtest command is
temporarily unavailable. Pass an instance of `SignalsBase` to `run_backtest`;
its `signals(batch)` method returns one direction per input window: `0` for buy,
`1` for sell, or `2` for no trade.

```python
from forexgrand_core.backtesting import run_backtest

result = run_backtest(
    strategy=my_strategy,
    bucket_name="forexgrand-test",
    source="dukascopy",
    symbol_pair="EURUSD",
    instrument_group="forex",
    sequence_length=60,
    stride=5,
    start_index=0,
    end_index=-1,
    return_in_points=True,
    sl_calculation={"mode": "fixed", "sl_points": 100, "tp_points": 150},
)
print(result.positions)
print(result.positions_total, result.buy_count, result.sell_count)
```

`sl_calculation` supports exactly these mode-specific keys (omitted keys use the
shown defaults):

- `{"mode": "fixed", "sl_points": 100, "tp_points": 100}`
- `{"mode": "range", "range": 60, "sl_ratio": 1.0, "tp_ratio": 1.0}`
- `{"mode": "atr", "sl_multiplier": 3.0, "tp_multiplier": 3.0, "atr_period": 14}`

Unknown keys or unsupported modes raise `ValueError`. Entry prices default to
the bid-based convention; use `entry_price_type="ask"` or `"mid"` when needed.
Use `start_index` and `end_index` to limit the test data; `end_index` is an
exclusive endpoint, and `-1` (the default) runs through the final bar.
Every position is closed by `tp`, `sl`, a `tiebreak`, or `eod`, and the result
contains `profit_equity`, `dd_equity`, and `unsupported_signal_count`.
Set `return_in_points=True` to divide position profits, drawdowns, and both
equity curves by the symbol point size. The CLI accepts the same options; pass
`--sl-calculation` as a JSON object and use `--return-in-points` for point-valued
output. Add `--output result.pkl.gz` to save the complete result dataclass.

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
