from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_manager import DataManager
from src.settings import Settings
from src.schemas import TARGET_MODEL_TYPES, TimeBasedTarget, PointsBasedTarget, SymbolProperties

# GZIP level 1 = fast compression, far less CPU than default level 6.
# ML training data does not benefit meaningfully from higher compression.
_TFRECORD_OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=1)


class GenerateTrainData:
    """Generate versioned TensorFlow training and evaluation TFRecords."""

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

    # Number of TFExamples serialized per write batch.
    # Keeps per-iteration Python overhead low without blowing memory.
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

        self.train_data_manager: DataManager = self._build_data_manager(base_bucket_name=self.train_base_bucket)
        self.eval_data_manager: DataManager = self._build_data_manager(base_bucket_name=self.eval_base_bucket)
        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()
        self.stride = None
        self.sequence_length = None
        self.target_model: TARGET_MODEL_TYPES = None
        self.train_properties: SymbolProperties = None
        self.eval_properties: SymbolProperties = None

    # ------------------------------------------------------------------ #
    #  Public API — unchanged signatures, fully compatible with Trainer   #
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
        bucket_manager = self._build_data_manager(bucket_name)
        _raw_frame, self.train_properties = bucket_manager.load_data(pair_name, group_name)
        _frame = self._prepare_feature_frame(_raw_frame)
        metadata = self._build_metadata(
            symbol_pair=pair_name,
            instrument_group=group_name,
            train_frame=_frame,
            eval_frame=_frame,
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
        _data_path = self._build_output_path(
            base_bucket=bucket_name,
            instrument_group=group_name,
            symbol_pair=pair_name,
            version_number=version_number,
            filename="train.gz",
        )

        self.generate_train_data_examples(_frame, output_path=_data_path)
        self._write_metadata(_data_path.parent / "metadata.json", metadata)
        return _data_path

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
        train_data_path = self._build_output_path(
            base_bucket=self.train_base_bucket,
            instrument_group=group_name,
            symbol_pair=pair_name,
            version_number=version_number,
            filename="train.gz",
        )
        eval_data_path = self._build_output_path(
            base_bucket=self.eval_base_bucket,
            instrument_group=group_name,
            symbol_pair=pair_name,
            version_number=version_number,
            filename="eval.gz",
        )

        self.generate_train_data_examples(train_frame, output_path=train_data_path)
        self.generate_eval_data_examples(eval_frame, output_path=eval_data_path)
        self._write_metadata(train_data_path.parent / "metadata.json", metadata)
        self._write_metadata(eval_data_path.parent / "metadata.json", metadata)
        return train_data_path, eval_data_path

    def generate_train_data_examples(self, dataframe: pd.DataFrame, *, output_path: Path) -> Path:
        self._write_examples(dataframe, output_path=output_path, symbol_properties=self.train_properties)
        return output_path

    def generate_eval_data_examples(self, dataframe: pd.DataFrame, *, output_path: Path) -> Path:
        self._write_examples(dataframe, output_path=output_path, symbol_properties=self.eval_properties)
        return output_path

    # ------------------------------------------------------------------ #
    #  Core write path — streaming, no full-dataset materialisation       #
    # ------------------------------------------------------------------ #

    def _write_examples(
        self,
        dataframe: pd.DataFrame,
        symbol_properties: SymbolProperties,
        *,
        output_path: Path,
    ) -> None:
        # Count examples without building any arrays yet.
        num_examples = self._count_examples(dataframe)

        target_data: dict[str, np.ndarray] = {}
        target_counts = num_examples
        if self.target_model:
            target_data, _counts = self._build_target_data(
                dataframe=dataframe, symbol_properties=symbol_properties
            )
            target_counts = min(num_examples, _counts)

        print("Examples:", num_examples, " Targets:", target_counts)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-extract raw 1-D numpy arrays from the dataframe — zero-copy views.
        # We slide a window manually per chunk so we never hold all windows in RAM.
        time_raw = (
            dataframe["time"]
            .astype("datetime64[s]")
            .astype("int64")
            .to_numpy(copy=False)
        ).astype(np.int64, copy=False)

        feature_raw: dict[str, np.ndarray] = {
            col: dataframe[col].to_numpy(dtype=np.float32, copy=False)
            for col in self.feature_columns
        }

        # Pre-slice target arrays (already small — shape (target_counts, 1))
        target_lists: dict[str, list] = {
            k: v[:target_counts, 0].tolist() for k, v in target_data.items()
        }
        float_target_keys = list(target_lists.keys())

        seq = self.sequence_length
        stride = self.stride

        with tf.io.TFRecordWriter(str(output_path), options=_TFRECORD_OPTIONS) as writer:
            for chunk_start in range(0, target_counts, self._WRITE_CHUNK_SIZE):
                chunk_end = min(chunk_start + self._WRITE_CHUNK_SIZE, target_counts)

                # Compute the raw-row span this chunk covers.
                # example i starts at row: i * stride
                raw_start = chunk_start * stride
                raw_end   = (chunk_end - 1) * stride + seq  # inclusive last row + 1

                # Slice only the rows needed for this chunk — small working set.
                time_chunk = time_raw[raw_start:raw_end]
                feature_chunks: dict[str, np.ndarray] = {
                    col: arr[raw_start:raw_end] for col, arr in feature_raw.items()
                }

                for i, example_idx in enumerate(range(chunk_start, chunk_end)):
                    # Local offset within the chunk slice
                    local_offset = i * stride
                    win = slice(local_offset, local_offset + seq)

                    features: dict[str, tf.train.Feature] = {
                        "time": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=time_chunk[win].tolist()
                            )
                        )
                    }
                    for col, arr in feature_chunks.items():
                        features[col] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=arr[win].tolist())
                        )
                    for k in float_target_keys:
                        features[k] = tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=[target_lists[k][example_idx]]
                            )
                        )
                    writer.write(
                        tf.train.Example(
                            features=tf.train.Features(feature=features)
                        ).SerializeToString()
                    )

    # ------------------------------------------------------------------ #
    #  Sequence builder — kept for metadata/count use only               #
    # ------------------------------------------------------------------ #

    def _build_sequence_data(self, dataframe: pd.DataFrame) -> dict[str, np.ndarray | int]:
        """
        Only called by external code that needs the full windowed arrays.
        The internal write path no longer uses this to avoid peak-RAM spikes.
        """
        if len(dataframe) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows are required to create sequences."
            )

        time_values = (
            dataframe["time"]
            .astype("datetime64[s]")
            .astype("int64")
            .to_numpy(copy=False)
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
    #  Target builder — unchanged logic, consistent shapes                #
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
                        pos_prices,
                        points,
                    ).astype("float32"),
                    axis=-1,
                ),
                "target_lowest": np.expand_dims(
                    self._calculate_points_diff(
                        pos_prices,
                        np.min(target_sequences["low"][self.sequence_length :: self.stride], axis=-1),
                        points,
                    ).astype("float32"),
                    axis=-1,
                ),
                "target_value": np.expand_dims(
                    self._calculate_points_diff(
                        pos_prices,
                        target_sequences["close"][self.sequence_length :: self.stride, -1],
                        points,
                    ).astype("float32"),
                    axis=-1,
                ),
            }
            print("target_highest shape:", target_cols["target_highest"].shape)
            return target_cols, pos_prices.shape[0]

        elif isinstance(self.target_model, PointsBasedTarget):
            raise NotImplementedError("mode not implemented: use TimeBasedTarget.")

    # ------------------------------------------------------------------ #
    #  Helpers — unchanged                                                #
    # ------------------------------------------------------------------ #

    def _build_data_manager(self, base_bucket_name: str) -> DataManager:
        return DataManager(base_bucket_name=base_bucket_name)

    def _prepare_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized_frame = self._prepare_dataframe(dataframe)
        if len(normalized_frame) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows are required to create sequences."
            )
        return normalized_frame

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = (
            dataframe.reset_index()
            if "time" not in dataframe.columns and dataframe.index.name == "time"
            else dataframe.copy()
        )
        required_columns = {"time", *self.BASE_FEATURE_COLUMNS}
        missing_columns = required_columns.difference(normalized.columns)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataframe is missing required columns: {missing_list}.")

        normalized["time"] = pd.to_datetime(normalized["time"], errors="coerce")
        if normalized["time"].isna().any():
            raise ValueError("The 'time' column contains invalid datetime values.")

        normalized = normalized.sort_values("time").drop_duplicates(subset="time", keep="last")
        normalized = normalized.reset_index(drop=True)
        return normalized[["time", *self.BASE_FEATURE_COLUMNS]]

    def _build_output_path(
        self,
        *,
        base_bucket: str,
        instrument_group: str,
        symbol_pair: str,
        version_number: int,
        filename: str,
    ) -> Path:
        return (
            self.data_directory
            / base_bucket
            / self.train_data_manager.data_source
            / instrument_group
            / symbol_pair
            / str(self.sequence_length)
            / str(version_number)
            / filename
        )

    def _window_array(self, values: np.ndarray, *, dtype: type[np.generic]) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(values, self.sequence_length)
        if self.stride > 1:
            windows = windows[:: self.stride]
        return np.ascontiguousarray(windows, dtype=dtype)

    def _build_tf_example(self, sequence_features: dict[str, np.ndarray]) -> tf.train.Example:
        """Kept for external callers; internal write path no longer uses this."""
        features: dict[str, tf.train.Feature] = {}
        for column_name, values in sequence_features.items():
            if column_name == "time":
                features[column_name] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=values.tolist())
                )
            else:
                features[column_name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=values.tolist())
                )
        return tf.train.Example(features=tf.train.Features(feature=features))

    def _build_metadata(
        self,
        *,
        symbol_pair: str,
        instrument_group: str,
        train_frame: pd.DataFrame,
        eval_frame: pd.DataFrame,
    ) -> dict[str, object]:
        train_examples = self._count_examples(train_frame)
        eval_examples = self._count_examples(eval_frame)
        return {
            "data_source": self.train_data_manager.data_source,
            "instrument_group": instrument_group,
            "symbol_pair": symbol_pair,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "feature_columns": list(self.feature_columns),
            "train_row_count": int(len(train_frame)),
            "train_example_count": train_examples,
            "train_start_time": self._isoformat(train_frame["time"].iloc[0]),
            "train_end_time": self._isoformat(train_frame["time"].iloc[-1]),
            "eval_row_count": int(len(eval_frame)),
            "eval_example_count": eval_examples,
            "eval_start_time": self._isoformat(eval_frame["time"].iloc[0]),
            "eval_end_time": self._isoformat(eval_frame["time"].iloc[-1]),
        }

    def _find_existing_version_paths(
        s