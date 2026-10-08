import numpy as np
import pandas as pd

import fg_core.pipeline.generate_train_data as generate_train_data
from fg_core.pipeline.generate_train_data import GenerateTrainData, _build_sequence_data
from fg_core.schemas import SymbolProperties, TimeBasedTarget


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


def test_generate_multiple_train_data_shape_and_alignment():
    generator = _generator(sequence_length=4, stride=2, chunk_size=10)
    generator.train_base_bucket = "forexgrand-train"
    generator.eval_base_bucket = "forexgrand-eval"

    # EURUSD: starts at 00:00, 10 minutes, but 00:04 is missing
    eur_times = [
        pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(minutes=int(i))
        for i in [0, 1, 2, 3, 5, 6, 7, 8, 9]  # 00:04 missing
    ]
    eur_df = pd.DataFrame(
        {
            "time": eur_times,
            "open": [10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "close": [10.5, 11.5, 12.5, 13.5, 15.5, 16.5, 17.5, 18.5, 19.5],
            "high": [11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "low": [9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            "real_volume": [100.0] * 9,
            "spread": [1.0] * 9,
            "tick_volume": [50.0] * 9,
        }
    )

    # USDJPY: starts at 00:02 (00:00 and 00:01 missing), ends at 00:09
    jpy_times = [
        pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(minutes=int(i))
        for i in range(2, 10)
    ]
    jpy_df = pd.DataFrame(
        {
            "time": jpy_times,
            "open": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "close": [102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "high": [103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "real_volume": [200.0] * 8,
            "spread": [2.0] * 8,
            "tick_volume": [60.0] * 8,
        }
    )

    # AUDUSD: starts at 00:01 (00:00 missing), ends at 00:09
    aud_times = [
        pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(minutes=int(i))
        for i in range(1, 10)
    ]
    aud_df = pd.DataFrame(
        {
            "time": aud_times,
            "open": [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            "close": [51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5],
            "high": [52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
            "low": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            "real_volume": [300.0] * 9,
            "spread": [0.5] * 9,
            "tick_volume": [40.0] * 9,
        }
    )

    custom_data = {
        ("forex", "USDJPY"): jpy_df,
        ("forex", "EURUSD"): eur_df,
        ("forex", "AUDUSD"): aud_df,
    }

    # Pass in non-alphabetical order: USDJPY, EURUSD, AUDUSD
    symbols = [("forex", "USDJPY"), ("forex", "EURUSD"), ("forex", "AUDUSD")]

    result = generator.generate_multiple_train_data(
        symbols=symbols,
        bucket="train",
        sequence_length=4,
        stride=2,
        custom_data=custom_data,
    )

    # 10 total minutes (00:00 to 00:09)
    # sequence_length = 4, stride = 2 -> total_sequences = ((10 - 4) // 2) + 1 = 4
    # num_symbol_pairs = 3
    # Expected shape: (4, 3, 4)
    expected_fields = ["time", "open", "close", "high", "low", "real_volume", "spread", "tick_volume"]
    for field in expected_fields:
        assert field in result, f"Field {field} missing from result"
        assert result[field].shape == (4, 3, 4), f"Field {field} has shape {result[field].shape}, expected (4, 3, 4)"

    # Alphabetical order:
    # index 0: AUDUSD
    # index 1: EURUSD
    # index 2: USDJPY
    assert generator.symbol_pairs == ["AUDUSD", "EURUSD", "USDJPY"]

    # Check backfilling for USDJPY (index 2) at 00:00 and 00:01 (first two minutes of sequence 0)
    # The first available value was at 00:02 with open = 102.0
    # Sequence 0 covers time indices 0, 1, 2, 3 (00:00, 00:01, 00:02, 00:03)
    usdjpy_seq0_open = result["open"][0, 2, :]
    assert usdjpy_seq0_open[0] == 102.0  # backfilled from 00:02
    assert usdjpy_seq0_open[1] == 102.0  # backfilled from 00:02
    assert usdjpy_seq0_open[2] == 102.0  # original 00:02
    assert usdjpy_seq0_open[3] == 103.0  # original 00:03

    # Check forward-filling for EURUSD (index 1) at 00:04 (time index 4)
    # Sequence 1 (starts at index 2*stride=2) covers time indices 2, 3, 4, 5 (00:02, 00:03, 00:04, 00:05)
    # Time index 4 was missing, should be ffilled with 00:03 value (13.0)
    eurusd_seq1_open = result["open"][1, 1, :]
    assert eurusd_seq1_open[0] == 12.0  # 00:02
    assert eurusd_seq1_open[1] == 13.0  # 00:03
    assert eurusd_seq1_open[2] == 13.0  # 00:04 forward-filled from 00:03
    assert eurusd_seq1_open[3] == 15.0  # 00:05

    # Check backfill for AUDUSD (index 0) at 00:00
    aud_seq0_open = result["open"][0, 0, :]
    assert aud_seq0_open[0] == 51.0  # backfilled from 00:01
    assert aud_seq0_open[1] == 51.0  # original 00:01


def test_generate_multiple_train_data_source_and_custom_bucket():
    import pytest
    from fg_core.generate_train_data import generate_multiple_train_data

    m1 = _market(10)
    m2 = _market(10)
    custom_data = {
        "EURUSD": m1,
        "GBPUSD": m2,
    }

    result = generate_multiple_train_data(
        symbols=[("forex", "GBPUSD"), ("forex", "EURUSD")],
        bucket="forexgrand-train",
        source="ctrader",
        sequence_length=5,
        stride=1,
        custom_data=custom_data,
    )

    # 10 rows, seq_len 5, stride 1 -> total_sequences = 10 - 5 + 1 = 6
    assert result["close"].shape == (6, 2, 5)
    assert result["time"].shape == (6, 2, 5)

    # Custom bucket like 'forexgrand-eval' passed directly
    result_eval = generate_multiple_train_data(
        symbols=[("forex", "EURUSD")],
        bucket="forexgrand-eval",
        source="ctrader",
        sequence_length=5,
        custom_data=custom_data,
    )
    assert result_eval["close"].shape == (6, 1, 5)


def test_generate_multiple_train_data_datamanager_called_with_bucket_and_source(monkeypatch):
    from fg_core.schemas import SymbolProperties

    called_args = []

    class MockDataManager:
        def __init__(self, base_bucket_name, source=None):
            called_args.append(("init", base_bucket_name, source))

        def load_data(self, symbol_pair, instrument_group, source=None):
            called_args.append(("load", symbol_pair, instrument_group, source))
            m = _market(10)
            props = SymbolProperties(
                symbol=symbol_pair,
                source=source or "ctrader",
                group=instrument_group,
                contract_size=1,
                point_size=1.0,
                digits=0,
                data_start=None,
                data_end=None,
            )
            return m, props

    generator = _generator(sequence_length=4, stride=1, chunk_size=10)
    monkeypatch.setattr(generator, "_build_data_manager", lambda bucket, source=None: MockDataManager(bucket, source))

    res = generator.generate_multiple_train_data(
        symbols=[("forex", "EURUSD"), ("forex", "GBPUSD")],
        bucket="forexgrand-train",
        source="ctrader",
        sequence_length=4,
    )

    assert ("init", "forexgrand-train", "ctrader") in called_args
    assert ("load", "EURUSD", "forex", "ctrader") in called_args
    assert ("load", "GBPUSD", "forex", "ctrader") in called_args
    assert res["close"].shape == (7, 2, 4)


def test_generate_multiple_train_data_return_dataframe_and_validations():
    import pytest
    from fg_core.generate_train_data import generate_multiple_train_data

    m1 = _market(10)
    m2 = _market(10)
    custom_data = {"EURUSD": m1, "GBPUSD": m2}

    res_dict, combined_df = generate_multiple_train_data(
        symbols=[("forex", "EURUSD"), ("forex", "GBPUSD")],
        bucket="forexgrand-train",
        source="ctrader",
        sequence_length=4,
        custom_data=custom_data,
        return_dataframe=True,
    )

    assert isinstance(combined_df, pd.DataFrame)
    assert len(combined_df) == 10
    # Combined df columns should have symbols in alphabetical order
    assert list(combined_df.columns.levels[0]) == ["EURUSD", "GBPUSD"]
    assert res_dict["close"].shape == (7, 2, 4)

    # Empty symbols error
    with pytest.raises(ValueError, match="At least one"):
        generate_multiple_train_data(symbols=[], custom_data=custom_data)

    # Sequence length too large error
    with pytest.raises(ValueError, match="bounds exceed|At least"):
        generate_multiple_train_data(
            symbols=[("forex", "EURUSD")],
            bucket="forexgrand-train",
            source="ctrader",
            sequence_length=20,
            custom_data=custom_data,
        )
