from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from src.data_manager import DataManager
from src.settings import Settings
from src.schemas import TARGET_MODEL_TYPES, TimeBasedTarget, PointsBasedTarget, SymbolProperties

_TFRECORD_OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=1)
_SHARD_COUNT = 16
_CPU_COUNT = os.cpu_count() or 4


class GenerateTrainData:
    """Generate versioned TensorFlow training and evaluation TFRecords."""

    BASE_FEATURE_COLUMNS = (
        "open", "close", "high", "low",
        "real_volume", "spread", "tick_volume",
    )

    TIMEFRAME_ALIASES = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1d",
    }

    _WRITE_CHUNK_SIZE = 4_000

    def __init__(
        self,
        train_base_bucket: str = "forexgrand-train",
        eval_base_bucket: str = "forexgrand-eval",
    ) -> None:
        self.settings = Settings()
        self.feature_columns = self.BASE_FEATURE_COLUMNS
        self.train_base_bucket = train_base_bucket.strip() or "forexgrand-train"
        self.eval_base_bucket = eval_base_bucket.strip() or "forexgrand-eval"
        if not self.train_base_bucket or not self.eval_base_bucket:
            raise ValueError("Both train bucket and eval bucket name is required.")

        self.train_data_manager: DataManager = self._build_data_manager(self.train_base_bucket)
        self.eval_data_manager: DataManager = self._build_data_manager(self.eval_base_bucket)
        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()
        self.stride = None
        self.sequence_length = None
        self.target_model: TARGET_MODEL_TYPES = None
        self.train_properties: SymbolProperties = None
        self.eval_properties: SymbolProperties = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def load_single_data(
        self,
        bucket_name: str,
        symbol_pair: str,
        instrument_group: str,
        sequence_length: int,
        stride: int,
        hot_reload: bool = False,
        target_model: TARGET_MODEL_TYPES = None,
    ) -> Path:
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_model = target_model

        pair_name = symbol_pair.strip().upper()
        group_name = instrument_group.strip().lower()
        if not pair_name:
            raise ValueError("symbol_pair is required.")
        if not group_name:
            raise ValueError("instrument_group is required.")

        bucket_manager = self._build_data_manager(bucket_name)
        _raw_frame, self.train_properties = bucket_manager.load_data(pair_name, group_name)
        _frame = self._prepare_feature_frame(_raw_frame)
        metadata = self._build_metadata(
            symbol_pair=pair_name, instrument_group=group_name,
            train_frame=_frame, eval_frame=_frame,
        )

        existing_paths = self._find_existing_version_paths(
            symbol_pair=pair_name, instrument_group=group_name, metadata=metadata,
        )
        if existing_paths is not None and not hot_reload:
            return existing_paths

        version_number = self._resolve_next_version_number(
            symbol_pair=pair_name, instrument_group=group_name,
        )
        output_dir = self._build_output_dir(
            base_bucket=bucket_name, instrument_group=group_name,
            symbol_pair=pair_name, version_number=version_number, split="train",
        )

        self.generate_train_data_examples(_frame, output_dir=output_dir)
        self._write_metadata(output_dir / "metadata.json", metadata)
        return output_dir

    def load_data(
        self,
        symbol_pair: str,
        instrument_group: str,
        sequence_length: int,
        stride: int,
        hot_reload: bool = False,
        target_model: TARGET_MODEL_TYPES = None,
    ) -> tuple[Path, Path]:
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_model = target_model

        pair_name = symbol_pair.strip().upper()
        group_name = instrument_group.strip().lower()
        if not pair_name:
            raise ValueError("symbol_pair is required.")
        if not group_name:
            raise ValueError("instrument_group is required.")

        train_raw_frame, self.train_properties = self.train_data_manager.load_data(pair_name, group_name)
        eval_raw_frame, self.eval_properties = self.eval_data_manager.load_data(pair_name, group_name)
        train_frame = self._prepare_feature_frame(train_raw_frame)
        eval_frame = self._prepare_feature_frame(eval_raw_frame)
        metadata = self._build_metadata(
            symbol_pair=pair_name, instrument_group=group_name,
            train_frame=train_frame, eval_frame=eval_frame,
        )

        existing_paths = self._find_existing_version_paths(
            symbol_pair=pair_name, instrument_group=group_name, metadata=metadata,
        )
        if existing_paths is not None and not hot_reload:
            return existing_paths

        version_number = self._resolve_next_version_number(
            symbol_pair=pair_name, instrument_group=group_name,
        )
        train_dir = self._build_output_dir(
            base_bucket=self.train_base_bucket, instrument_group=group_name,
            symbol_pair=pair_name, version_number=version_number, split="train",
        )
        eval_dir = self._build_output_dir(
            base_bucket=self.eval_base_bucket, instrument_group=group_name,
            symbol_pair=pair_name, version_number=version_number, split="eval",
        )

        # ✅ Run train + eval generation concurrently on separate threads
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self.generate_train_data_examples, train_frame, output_dir=train_dir): "train",
                pool.submit(self.generate_eval_data_examples, eval_frame, output_dir=eval_dir): "eval",
            }
            for future in as_completed(futures):
                split = futures[future]
                exc = future.exception()
                if exc:
                    raise RuntimeError(f"{split} generation failed: {exc}") from exc

        self._write_metadata(train_dir / "metadata.json", metadata)
        self._write_metadata(eval_dir / "metadata.json", metadata)
        return train_dir, eval_dir

    def generate_train_data_examples(self, dataframe: pd.DataFrame, *, output_dir: Path) -> Path:
        shard_paths = self._build_shard_paths(output_dir, split="train")
        self._write_examples_sharded(dataframe, output_paths=shard_paths, symbol_properties=self.train_properties)
        return output_dir

    def generate_eval_data_examples(self, dataframe: pd.DataFrame, *, output_dir: Path) -> Path:
        shard_paths = self._build_shard_paths(output_dir, split="eval")
        self._write_examples_sharded(dataframe, output_paths=shard_paths, symbol_properties=self.eval_properties)
        return output_dir

    # ------------------------------------------------------------------ #
    #  Core write path                                                    #
    # ------------------------------------------------------------------ #

    def _build_shard_paths(self, output_dir: Path, split: str) -> list[Path]:
        return [
            output_dir / f"{split}_{i:05d}_of_{_SHARD_COUNT:05d}.gz"
            for i in range(_SHARD_COUNT)
        ]

    def _write_examples_sharded(
        self,
        dataframe: pd.DataFrame,
        symbol_properties: SymbolProperties,
        *,
        output_paths: list[Path],
    ) -> None:
        num_shards = len(output_paths)
        output_paths[0].parent.mkdir(parents=True, exist_ok=True)

        num_examples = self._count_examples(dataframe)
        target_data: dict[str, np.ndarray] = {}
        target_counts = num_examples
        if self.target_model:
            target_data, _counts = self._build_target_data(
                dataframe=dataframe, symbol_properties=symbol_properties
            )
            target_counts = min(num_examples, _counts)

        print("Examples:", num_examples, " Targets:", target_counts)

        time_raw = (
            dataframe["time"].astype("datetime64[s]").astype("int64").to_numpy(copy=False)
        ).astype(np.int64, copy=False)

        feature_raw: dict[str, np.ndarray] = {
            col: dataframe[col].to_numpy(dtype=np.float32, copy=False)
            for col in self.feature_columns
        }
        target_lists: dict[str, list] = {
            k: v[:target_counts, 0].tolist() for k, v in target_data.items()
        }

        seq = self.sequence_length
        stride = self.stride
        feature_cols = self.feature_columns

        # ✅ Partition examples across shards; each shard owns its example indices
        shard_index_ranges = _partition_into_shards(target_counts, num_shards)

        # ✅ Serialize each shard's examples in parallel worker processes
        worker_args = [
            (
                shard_idx,
                str(output_paths[shard_idx]),
                shard_index_ranges[shard_idx],
                time_raw,
                feature_raw,
                target_lists,
                seq,
                stride,
                feature_cols,
            )
            for shard_idx in range(num_shards)
        ]

        with ProcessPoolExecutor(max_workers=_CPU_COUNT) as executor:
            futures = {executor.submit(_write_shard_worker, args): args[0] for args in worker_args}
            for future in as_completed(futures):
                shard_idx = futures[future]
                exc = future.exception()
                if exc:
                    raise RuntimeError(f"Shard {shard_idx} failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Sequence builder — kept for external callers only                  #
    # ------------------------------------------------------------------ #

    def _build_sequence_data(self, dataframe: pd.DataFrame) -> dict[str, np.ndarray | int]:
        if len(dataframe) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows are required to create sequences."
            )
        time_values = (
            dataframe["time"].astype("datetime64[s]").astype("int64").to_numpy(copy=False)
        ).astype(np.int64, copy=False)

        feature_matrix = np.stack(
            [dataframe[col].to_numpy(dtype=np.float32, copy=False) for col in self.feature_columns],
            axis=1,
        )
        windowed = np.lib.stride_tricks.sliding_window_view(
            feature_matrix, (self.sequence_length, feature_matrix.shape[1])
        )[:, 0, :, :]

        if self.stride > 1:
            windowed = windowed[:: self.stride]

        time_windowed = self._window_array(time_values, dtype=np.int64)
        sequence_data: dict[str, np.ndarray | int] = {"time": time_windowed}
        for i, col in enumerate(self.feature_columns):
            sequence_data[col] = np.ascontiguousarray(windowed[:, :, i])
        sequence_data["num_examples"] = int(time_windowed.shape[0])
        return sequence_data

    # ------------------------------------------------------------------ #
    #  Target builder                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_points_diff(self, price_arr1, price_arr2, points_size) -> np.ndarray:
        return (price_arr1 - price_arr2) // points_size

    def _build_target_data(self, dataframe: pd.DataFrame, symbol_properties: SymbolProperties):
        if isinstance(self.target_model, TimeBasedTarget):
            required_cols = ["high", "low", "close"]
            target_seq_length = self.target_model.stop_minutes
            target_sequences = {
                col: np.lib.stride_tricks.sliding_window_view(
                    dataframe[col].to_numpy(), target_seq_length
                )
                for col in required_cols
            }
            pos_prices = np.lib.stride_tricks.sliding_window_view(
                dataframe["close"].to_numpy(), target_seq_length
            )
            pos_prices = pos_prices[self.sequence_length - 1: -1, 0]
            pos_prices = pos_prices[:: self.stride]

            points = symbol_properties.point_size
            target_cols = {
                "target_highest": np.expand_dims(
                    self._calculate_points_diff(
                        np.max(target_sequences["high"][self.sequence_length :: self.stride], axis=-1),
                        pos_prices, points,
                    ).astype("float32"), axis=-1,
                ),
                "target_lowest": np.expand_dims(
                    self._calculate_points_diff(
                        pos_prices,
                        np.min(target_sequences["low"][self.sequence_length :: self.stride], axis=-1),
                        points,
                    ).astype("float32"), axis=-1,
                ),
                "target_value": np.expand_dims(
                    self._calculate_points_diff(
                        pos_prices,