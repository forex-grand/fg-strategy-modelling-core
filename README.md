# fg-strategy-modelling-core
This Repository provides abstract classes for developing a new model.

## Quick Usage

`DataManager` loads the raw M1 parquet data for a symbol pair and instrument group from the configured storage bucket. `GenerateTrainData` then uses separate train and eval data managers, optionally computes indicators, converts to the requested timeframe, and saves versioned gzipped TFRecord files.

```python
from src.data_manager import DataManager
from src.generate_train_data import GenerateTrainData
from src.indicators import TensorFlowIndicators

generator = GenerateTrainData(
    data_manager=DataManager,
    data_manager_kwargs={
        "data_source": "mt5",
    },
    sequence_length=128,
    timeframe="1m",
    indicators=[
        {
            "function": TensorFlowIndicators.MA(period=20),
            "buffers": [0],
            "timeframe": "1m",
            "name": "ma",
            "kwargs": {},
        },
        {
            "function": TensorFlowIndicators.BollingerBands(period=20, deviation=2.0),
            "buffers": [1, 2],
            "timeframe": "4h",
            "name": "bb",
            "kwargs": {},
        },
    ],
)

train_data_path, eval_data_path = generator.load_data(
    symbol_pair="EURUSD",
    instrument_group="forex",
    hot_reload=False,
)
```

Notes:
- `GenerateTrainData` automatically creates one `DataManager` for `forexgrand-train` and one for `forexgrand-eval`.
- Indicator configs must include `function`, `buffers`, `timeframe`, `name`, and optional `kwargs`.
- Indicator column names are generated like `ma_1m_0` or `bb_4h_2`.
- Cached generated files are reused when the stored metadata still matches the current timeframe, features, and indicator request.
