from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_manager import DataManager
from src.schemas import (
    TARGET_MODEL_TYPES,
    PointsBasedTarget,
    SymbolProperties,
    TimeBasedTarget,
)
from src.settings import Settings

# ============================================================
# TFRecord settings
# ============================================================

_TFRECORD_OPTIONS = tf.io.TFRecordOptions(
    compression_type="GZIP",
    compression_level=1,
)

_SHARD_COUNT = max(1, os.cpu_count() or 1)


# ============================================================
# Multiprocess worker
# ============================================================

def _write_tfrecord_shard_worker(
    shard_path: str,
    shard_indices: np.ndarray,
    time_windows: np.ndarray,
    feature_windows: np.ndarray,
    target_arrays: dict[str, np.ndarray],
):
    writer = tf.io.TFRecordWriter(
        shard_path,
        options=_TFRECORD_OPTIONS,
    )

    try:
        for idx in shard_indices:

            feature_dict = {}

            # ----------------------------------------------------
            # time tensor
            # ----------------------------------------------------

            serialized_time = tf.io.serialize_tensor(
                tf.convert_to_tensor(
                    time_windows[idx],
                    dtype=tf.int64,
                )
            ).numpy()

            feature_dict["time"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[serialized_time]
                )
            )

            # ----------------------------------------------------
            # feature tensor
            # shape:
            # (sequence_length, num_features)
            # ----------------------------------------------------

            serialized_features = tf.io.serialize_tensor(
                tf.convert_to_tensor(
                    feature_windows[idx],
                    dtype=tf.float32,
                )
            ).numpy()

            feature_dict["features"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[serialized_features]
                )
            )

            # ----------------------------------------------------
            # targets
            # ----------------------------------------------------

            for target_name, target_values in target_arrays.items():

                serialized_target = tf.io.serialize_tensor(
                    tf.convert_to_tensor(
                        target_values[idx],
                        dtype=tf.float32,
                    )
                ).numpy()

                feature_dict[target_name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[serialized_target]
                    )
                )

            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature_dict
                )
            )

            writer.write(example.SerializeToString())

    finally:
        writer.close()


# ============================================================
# Main class
# ============================================================

class GenerateTrainData:
    """Generate versioned TensorFlow TFRecords."""

    BASE_FEATURE_COLUMNS = (
        "open",
        "close",
        "high",
        "low",
        "real_volume",
        "spread",
        "tick_volume",
    )

    TIMEFRAME_ALIASES = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(
        self,
        train_base_bucket: str = "forexgrand-train",
        eval_base_bucket: str = "forexgrand-eval",
    ) -> None:

        self.settings = Settings()

        self.feature_columns = self.BASE_FEATURE_COLUMNS

        self.train_base_bucket = train_base_bucket.strip()
        self.eval_base_bucket = eval_base_bucket.strip()

        self.train_data_manager: DataManager = (
            self._build_data_manager(
                base_bucket_name=self.train_base_bucket
            )
        )

        self.eval_data_manager: DataManager = (
            self._build_data_manager(
                base_bucket_name=self.eval_base_bucket
            )
        )

        self.data_directory = (
            Path(self.settings.data_directory)
            .expanduser()
            .resolve()
        )

        self.sequence_length = None
        self.stride = None
        self.target_model: TARGET_MODEL_TYPES = None

        self.train_properties: SymbolProperties = None
        self.eval_properties: SymbolProperties = None

    # ============================================================
    # Public API
    # ============================================================

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

        train_raw_frame, self.train_properties = (
            self.train_data_manager.load_data(
                pair_name,
                group_name,
            )
        )

        eval_raw_frame, self.eval_properties = (
            self.eval_data_manager.load_data(
                pair_name,
                group_name,
            )
        )

        train_frame = self._prepare_feature_frame(
            train_raw_frame
        )

        eval_frame = self._prepare_feature_frame(
            eval_raw_frame
        )

        metadata = self._build_metadata(
            symbol_pair=pair_name,
            instrument_group=group_name,
            train_frame=train_frame,
            eval_frame=eval_frame,
        )

        existing_paths = self._find_existing_version_paths(
            symbol_pair=pair_name,
            instrument_group=group_name,
            metadata=metadata,
        )

        if existing_paths is not None and not hot_reload:
            return existing_paths

        version_number = self._resolve_next_version_number(
            symbol_pair=pair_name,
            instrument_group=group_name,
        )

        train_dir = self._build_output_dir(
            base_bucket=self.train_base_bucket,
            instrument_group=group_name,
            symbol_pair=pair_name,
            version_number=version_number,
            split="train",
        )

        eval_dir = self._build_output_dir(
            base_bucket=self.eval_base_bucket,
            instrument_group=group_name,
            symbol_pair=pair_name,
            version_number=version_number,
            split="eval",
        )

        self.generate_train_data_examples(
            train_frame,
            output_dir=train_dir,
        )

        self.generate_eval_data_examples(
            eval_frame,
            output_dir=eval_dir,
        )

        self._write_metadata(
            train_dir / "metadata.json",
            metadata,
        )

        self._write_metadata(
            eval_dir / "metadata.json",
            metadata,
        )

        return train_dir, eval_dir

    # ============================================================
    # TFRecord generation
    # ============================================================

    def generate_train_data_examples(
        self,
        dataframe: pd.DataFrame,
        *,
        output_dir: Path,
    ) -> Path:

        shard_paths = self._build_shard_paths(
            output_dir,
            split="train",
        )

        self._write_examples_sharded(
            dataframe,
            output_paths=shard_paths,
            symbol_properties=self.train_properties,
        )

        return output_dir

    def generate_eval_data_examples(
        self,
        dataframe: pd.DataFrame,
        *,
        output_dir: Path,
    ) -> Path:

        shard_paths = self._build_shard_paths(
            output_dir,
            split="eval",
        )

        self._write_examples_sharded(
            dataframe,
            output_paths=shard_paths,
            symbol_properties=self.eval_properties,
        )

        return output_dir

    # ============================================================
    # Optimized writer
    # ============================================================

    def _write_examples_sharded(
        self,
        dataframe: pd.DataFrame,
        symbol_properties: SymbolProperties,
        *,
        output_paths: list[Path],
    ) -> None:

        output_paths[0].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        num_examples = self._count_examples(dataframe)

        target_data = {}
        target_counts = num_examples

        if self.target_model:

            target_data, target_counts = (
                self._build_target_data(
                    dataframe=dataframe,
                    symbol_properties=symbol_properties,
                )
            )

            target_counts = min(
                target_counts,
                num_examples,
            )

        print("Examples:", num_examples)
        print("Targets:", target_counts)

        seq = self.sequence_length
        stride = self.stride

        # ========================================================
        # time windows
        # ========================================================

        time_values = (
            dataframe["time"]
            .astype("datetime64[s]")
            .astype("int64")
            .to_numpy(copy=False)
            .astype(np.int64, copy=False)
        )

        time_windows = np.lib.stride_tricks.sliding_window_view(
            time_values,
            seq,
        )

        if stride > 1:
            time_windows = time_windows[::stride]

        time_windows = np.ascontiguousarray(
            time_windows[:target_counts]
        )

        # ========================================================
        # feature matrix
        # ========================================================

        feature_matrix = np.stack(
            [
                dataframe[col].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                for col in self.feature_columns
            ],
            axis=1,
        )

        # ========================================================
        # feature windows
        # shape:
        # (num_examples, sequence_length, num_features)
        # ========================================================

        feature_windows = (
            np.lib.stride_tricks.sliding_window_view(
                feature_matrix,
                (seq, feature_matrix.shape[1]),
            )[:, 0]
        )

        if stride > 1:
            feature_windows = feature_windows[::stride]

        feature_windows = np.ascontiguousarray(
            feature_windows[:target_counts],
            dtype=np.float32,
        )

        # ========================================================
        # targets
        # ========================================================

        processed_targets = {}

        for target_name, target_values in target_data.items():

            processed_targets[target_name] = (
                np.ascontiguousarray(
                    target_values[:target_counts],
                    dtype=np.float32,
                )
            )

        # ========================================================
        # shard indices
        # ========================================================

        num_shards = len(output_paths)

        shard_indices = [
            np.arange(
                i,
                target_counts,
                num_shards,
                dtype=np.int64,
            )
            for i in range(num_shards)
        ]

        # ========================================================
        # multiprocessing write
        # ========================================================

        max_workers = min(
            num_shards,
            max(1, os.cpu_count() or 1),
        )

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        ) as executor:

            futures = []

            for shard_id, indices in enumerate(shard_indices):

                if len(indices) == 0:
                    continue

                futures.append(
                    executor.submit(
                        _write_tfrecord_shard_worker,
                        str(output_paths[shard_id]),
                        indices,
                        time_windows,
                        feature_windows,
                        processed_targets,
                    )
                )

            for future in futures:
                future.result()

    # ============================================================
    # Target builder
    # ============================================================

    def _calculate_points_diff(
        self,
        price_arr1,
        price_arr2,
        points_size,
    ) -> np.ndarray:

        return (
            (price_arr1 - price_arr2)
            // points_size
        )

    def _build_target_data(
        self,
        dataframe: pd.DataFrame,
        symbol_properties: SymbolProperties,
    ):

        if isinstance(self.target_model, TimeBasedTarget):

            required_cols = [
                "high",
                "low",
                "close",
            ]

            target_seq_length = (
                self.target_model.stop_minutes
            )

            target_sequences = {
                col: np.lib.stride_tricks.sliding_window_view(
                    dataframe[col].to_numpy(),
                    target_seq_length,
                )
                for col in required_cols
            }

            pos_prices = (
                np.lib.stride_tricks.sliding_window_view(
                    dataframe["close"].to_numpy(),
                    target_seq_length,
                )
            )

            pos_prices = pos_prices[
                self.sequence_length - 1: -1,
                0,
            ]

            pos_prices = pos_prices[::self.stride]

            points = symbol_properties.point_size

            target_cols = {

                "target_highest": np.expand_dims(
                    self._calculate_points_diff(
                        np.max(
                            target_sequences["high"][
                                self.sequence_length::self.stride
                            ],
                            axis=-1,
                        ),
                        pos_prices,
                        points,
                    ).astype("float32"),
                    axis=-1,
                ),

                "target_lowest": np.expand_dims(
                    self._calculate_points_diff(
                        pos_prices,
                        np.min(
                            target_sequences["low"][
                                self.sequence_length::self.stride
                            ],
                            axis=-1,
                        ),
                        points,
                    ).astype("float32"),
                    axis=-1,
                ),

                "target_value": np.expand_dims(
                    self._calculate_points_diff(
                        pos_prices,
                        target_sequences["close"][
                            self.sequence_length::self.stride,
                            -1,
                        ],
                        points,
                    ).astype("float32"),
                    axis=-1,
                ),
            }

            return target_cols, pos_prices.shape[0]

        elif isinstance(
            self.target_model,
            PointsBasedTarget,
        ):
            raise NotImplementedError

    # ============================================================
    # Helpers
    # ============================================================

    def _build_shard_paths(
        self,
        output_dir: Path,
        split: str,
    ) -> list[Path]:

        return [
            output_dir / (
                f"{split}_{i:05d}"
                f"_of_{_SHARD_COUNT:05d}.gz"
            )
            for i in range(_SHARD_COUNT)
        ]

    def _build_data_manager(
        self,
        base_bucket_name: str,
    ) -> DataManager:

        return DataManager(
            base_bucket_name=base_bucket_name
        )

    def _prepare_feature_frame(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        normalized_frame = (
            self._prepare_dataframe(dataframe)
        )

        if len(normalized_frame) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows "
                f"are required."
            )

        return normalized_frame

    def _prepare_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        normalized = (
            dataframe.reset_index()
            if (
                "time" not in dataframe.columns
                and dataframe.index.name == "time"
            )
            else dataframe.copy()
        )

        required_columns = {
            "time",
            *self.BASE_FEATURE_COLUMNS,
        }

        missing_columns = (
            required_columns.difference(
                normalized.columns
            )
        )

        if missing_columns:

            missing_list = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                f"Missing columns: {missing_list}"
            )

        normalized["time"] = pd.to_datetime(
            normalized["time"],
            errors="coerce",
        )

        if normalized["time"].isna().any():
            raise ValueError(
                "Invalid datetime values."
            )

        normalized = (
            normalized
            .sort_values("time")
            .drop_duplicates(
                subset="time",
                keep="last",
            )
            .reset_index(drop=True)
        )

        return normalized[
            [
                "time",
                *self.BASE_FEATURE_COLUMNS,
            ]
        ]

    def _build_output_dir(
        self,
        *,
        base_bucket: str,
        instrument_group: str,
        symbol_pair: str,
        version_number: int,
        split: str,
    ) -> Path:

        return (
            self.data_directory
            / base_bucket
            / self.train_data_manager.data_source
            / instrument_group
            / symbol_pair
            / str(self.sequence_length)
            / str(version_number)
            / split
        )

    def _build_metadata(
        self,
        *,
        symbol_pair: str,
        instrument_group: str,
        train_frame: pd.DataFrame,
        eval_frame: pd.DataFrame,
    ) -> dict[str, object]:

        train_examples = self._count_examples(
            train_frame
        )

        eval_examples = self._count_examples(
            eval_frame
        )

        return {
            "data_source":
                self.train_data_manager.data_source,

            "instrument_group":
                instrument_group,

            "symbol_pair":
                symbol_pair,

            "sequence_length":
                self.sequence_length,

            "stride":
                self.stride,

            "feature_columns":
                list(self.feature_columns),

            "train_row_count":
                int(len(train_frame)),

            "train_example_count":
                train_examples,

            "eval_row_count":
                int(len(eval_frame)),

            "eval_example_count":
                eval_examples,
        }

    def _find_existing_version_paths(
        self,
        *,
        symbol_pair: str,
        instrument_group: str,
        metadata: dict[str, object],
    ) -> Optional[tuple[Path, Path]]:

        return None

    def _resolve_next_version_number(
        self,
        *,
        symbol_pair: str,
        instrument_group: str,
    ) -> int:

        return 1

    @staticmethod
    def _write_metadata(
        metadata_path: Path,
        metadata: dict[str, object],
    ) -> None:

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:

            json.dump(
                metadata,
                file_handle,
                indent=2,
            )

    def _count_examples(
        self,
        dataframe: pd.DataFrame,
    ) -> int:

        return max(
            0,
            (
                (
                    len(dataframe)
                    - self.sequence_length
                )
                // self.stride
            ) + 1,
        )