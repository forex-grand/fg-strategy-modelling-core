import numpy as np
import pandas as pd

import forexgrand_core.pipeline.generate_train_data as generate_train_data
from forexgrand_core.pipeline.generate_train_data import GenerateTrainData, _build_sequence_data
from forexgrand_core.schemas import SymbolProperties, TimeBasedTarget


def _generator(sequence_length, stride, chunk_size, target_model=None):
    generator = GenerateTrainData.__new__(GenerateTrainData)
    generator.sequence_length = sequence_length
    generator.stride = stride
    generator.chunk_size = chunk_size
    generator.target_model = target_model
    return generator


def _market(rows):
    values = np.arange(rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "open": values,
            "close": values,
            "high": values + 1,
            "low": values - 1,
            "real_volume": values,
            "spread": np.zeros(rows, dtype=np.float32),
            "tick_volume": values,
        }
    )


def _symbol_properties():
    return SymbolProperties(
        symbol="TEST",
        source="test",
        group="test",
        contract_size=1,
        point_size=1.0,
        digits=0,
        data_start=None,
        data_end=None,
    )


def _window_starts_from_chunks(dataframe, chunks, symbol_properties):
    starts = []
    for chunk in chunks:
        sequence_data = _build_sequence_data(dataframe.iloc[chunk], symbol_properties)
        starts.extend(sequence_data["open"][:, 0].tolist())
    return starts


def test_chunk_slices_match_explicit_strided_windows_without_targets(monkeypatch):
    sequence_length = 4
    stride = 3
    dataframe = _market(20)
    generator = _generator(sequence_length, stride, chunk_size=2)
    monkeypatch.setattr(generate_train_data, "SEQUENCE_LENGTH", sequence_length)
    monkeypatch.setattr(generate_train_data, "STRIDE", stride)
    monkeypatch.setattr(generate_train_data, "TARGET_MODEL", None)

    example_count = generator._count_examples(dataframe)
    chunks = generator._build_chunk_slices(len(dataframe), example_count, None)

    assert example_count == 6
    assert _window_starts_from_chunks(dataframe, chunks, None) == [0, 3, 6, 9, 12, 15]


def test_target_chunks_preserve_exact_windows_and_targets(monkeypatch):
    sequence_length = 4
    stride = 3
    target_length = 5
    target_model = TimeBasedTarget(stop_minutes=target_length, mode="raw_difference")
    dataframe = _market(20)
    generator = _generator(sequence_length, stride, chunk_size=2, target_model=target_model)
    symbol_properties = _symbol_properties()
    monkeypatch.setattr(generate_train_data, "SEQUENCE_LENGTH", sequence_length)
    monkeypatch.setattr(generate_train_data, "STRIDE", stride)
    monkeypatch.setattr(generate_train_data, "TARGET_MODEL", target_model)

    example_count, returned_target_length = generator._count_targets(dataframe)
    chunks = generator._build_chunk_slices(
        len(dataframe), example_count, returned_target_length
    )

    starts = []
    target_values = []
    for chunk in chunks:
        sequence_data = _build_sequence_data(dataframe.iloc[chunk], symbol_properties)
        starts.extend(sequence_data["open"][:, 0].tolist())
        target_values.extend(sequence_data["target_value"][:, 0].tolist())

    assert example_count == 4
    assert starts == [0, 3, 6, 9]
    assert target_values == [-5.0, -5.0, -5.0, -5.0]