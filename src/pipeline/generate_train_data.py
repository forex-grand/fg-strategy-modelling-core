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

        self.train_data_manager:DataManager = self._build_data_manager(base_bucket_name=self.train_base_bucket)
        self.eval_data_manager:DataManager = self._build_data_manager(base_bucket_name=self.eval_base_bucket)
        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()
        self.stride = None
        self.sequence_length = None
        self.target_model:TARGET_MODEL_TYPES = None
        self.train_properties: SymbolProperties = None
        self.eval_properties: SymbolProperties  = None

    def load_single_data(
        self,
        bucket_name: str,
        symbol_pair: str,
        instrument_group: str,
        sequence_length: int,
        stride: int,
        hot_reload: bool = False,
        target_model: TARGET_MODEL_TYPES = None
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

        self.generate_train_data_examples(
            _frame,
            output_path=_data_path,
        )
        self._write_metadata(_data_path.parent / "metadata.json", metadata)

        return _data_path

    def load_data(
        self,
        symbol_pair: str,
        instrument_group: str,
        sequence_length: int,
        stride: int,
        hot_reload: bool = False,
        target_model: TARGET_MODEL_TYPES = None
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

        self.generate_train_data_examples(
            train_frame,
            output_path=train_data_path,
        )
        self.generate_eval_data_examples(
            eval_frame,
            output_path=eval_data_path,
        )
        self._write_metadata(train_data_path.parent / "metadata.json", metadata)
        self._write_metadata(eval_data_path.parent / "metadata.json", metadata)
        return train_data_path, eval_data_path

    def generate_train_data_examples(
        self,
        dataframe: pd.DataFrame,
        *,
        output_path: Path,
    ) -> Path:
        self._write_examples(dataframe, output_path=output_path, symbol_properties=self.train_properties)
        return output_path

    def generate_eval_data_examples(
        self,
        dataframe: pd.DataFrame,
        *,
        output_path: Path,
    ) -> Path:
        self._write_examples(dataframe, output_path=output_path, symbol_properties=self.eval_properties)
        return output_path

    def _build_data_manager(
        self,
        base_bucket_name: str,
    ) -> DataManager:
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

    def _write_examples(self, dataframe: pd.DataFrame, symbol_properties: SymbolProperties, *, output_path: Path) -> None:
        sequence_data = self._build_sequence_data(dataframe)
        
        target_data, _counts = {}, None
        if self.target_model:
            target_data, _counts = self._build_target_data(dataframe=dataframe, symbol_properties=symbol_properties)
        target_counts = min(sequence_data["num_examples"], _counts) if _counts else sequence_data["num_examples"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        print("Examples: ",sequence_data["num_examples"]," Targets: ",target_counts)

        with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
            for index in range(target_counts):
                example = self._build_tf_example({
                    column_name: values[index]
                    for column_name, values in sequence_data.items()
                    if column_name != "num_examples"
                } | {
                    column_name: values[index]
                    for column_name, values in target_data.items()
                })
                writer.write(example.SerializeToString())

    def _build_sequence_data(self, dataframe: pd.DataFrame) -> dict[str, np.ndarray | int]:
        if len(dataframe) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows are required to create sequences."
            )
        
        time_values = (
            dataframe["time"].astype("datetime64[s]")  # convert directly to second precision
            .astype("int64")
            .to_numpy(copy=False)
        ).astype(np.int64, copy=False)
        sequence_data: dict[str, np.ndarray | int] = {
            "time": self._window_array(time_values, dtype=np.int64),
        }
        for column in self.feature_columns:
            values = dataframe[column].to_numpy(dtype=np.float32, copy=False)
            sequence_data[column] = self._window_array(values, dtype=np.float32)

        sequence_data["num_examples"] = int(sequence_data["time"].shape[0])

        return sequence_data

    def _calculate_points_diff(self, price_arr1, price_arr2, points_size)->np.ndarray:
        return (price_arr1 - price_arr2)//points_size
    
    def _build_target_data(self, dataframe: pd.DataFrame, symbol_properties: SymbolProperties):
        if isinstance(self.target_model, TimeBasedTarget):
            target_sequences = {}
            required_cols = ['high','low','close']
            target_seq_length = self.target_model.stop_minutes
            for col in required_cols:
                vals = np.lib.stride_tricks.sliding_window_view(
                    dataframe[col].to_numpy(), target_seq_length)
                target_sequences[col] = vals[self.sequence_length::self.stride]

            pos_prices = np.lib.stride_tricks.sliding_window_view(
                    dataframe['close'].to_numpy(), target_seq_length)
            pos_prices = pos_prices[self.sequence_length-1:-1, 0]
            pos_prices = pos_prices[::self.stride]
            points = symbol_properties.point_size
            target_cols = {
                'target_highest':np.expand_dims(self._calculate_points_diff(np.max(target_sequences['high'], axis=-1),
                                                             pos_prices, points).astype("float32"), axis=-1),
                'target_lowest':np.expand_dims(self._calculate_points_diff(pos_prices, np.min(target_sequences['low'], axis=-1), points).astype("float32"), axis=-1),
                'target_value':np.expand_dims(self._calculate_points_diff(pos_prices, target_sequences['close'][:,-1], points).astype("float32"), axis=-1),
            }
            print("target_highest shape: ",target_cols['target_highest'].shape)
            return target_cols, pos_prices.shape[0]
        elif isinstance(self.target_model, PointsBasedTarget):
            raise NotImplemented("mode not implemented: use TimeBasedTarget.")

    def _window_array(self, values: np.ndarray, *, dtype: type[np.generic]) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(values, self.sequence_length)
        if self.stride > 1:
            windows = windows[:: self.stride]
        return np.ascontiguousarray(windows, dtype=dtype)


    def _build_tf_example(self, sequence_features: dict[str, np.ndarray]) -> tf.train.Example:
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
        self,
        *,
        symbol_pair: str,
        instrument_group: str,
        metadata: dict[str, object],
    ) -> Optional[tuple[Path, Path]]:
        train_root = self._build_version_root(
            base_bucket=self.train_base_bucket,
            instrument_group=instrument_group,
            symbol_pair=symbol_pair,
        )
        eval_root = self._build_version_root(
            base_bucket=self.eval_base_bucket,
            instrument_group=instrument_group,
            symbol_pair=symbol_pair,
        )
        if not train_root.exists() or not eval_root.exists():
            return None

        common_versions = sorted(
            set(self._list_version_numbers(train_root)).intersection(
                self._list_version_numbers(eval_root)
            ),
            reverse=True,
        )
        for version_number in common_versions:
            train_dir = train_root / str(version_number)
            eval_dir = eval_root / str(version_number)
            train_data_path = train_dir / "train.gz"
            eval_data_path = eval_dir / "eval.gz"
            train_metadata_path = train_dir / "metadata.json"
            eval_metadata_path = eval_dir / "metadata.json"
            if not (
                train_data_path.exists()
                and eval_data_path.exists()
                and train_metadata_path.exists()
                and eval_metadata_path.exists()
            ):
                continue

            train_metadata = self._read_metadata(train_metadata_path)
            eval_metadata = self._read_metadata(eval_metadata_path)
            if train_metadata != eval_metadata:
                continue
            if self._metadata_matches(train_metadata, metadata):
                return train_data_path, eval_data_path
        return None

    def _resolve_next_version_number(
        self,
        *,
        symbol_pair: str,
        instrument_group: str,
    ) -> int:
        train_root = self._build_version_root(
            base_bucket=self.train_base_bucket,
            instrument_group=instrument_group,
            symbol_pair=symbol_pair,
        )
        eval_root = self._build_version_root(
            base_bucket=self.eval_base_bucket,
            instrument_group=instrument_group,
            symbol_pair=symbol_pair,
        )
        version_numbers = self._list_version_numbers(train_root) + self._list_version_numbers(eval_root)
        return (max(version_numbers) + 1) if version_numbers else 1

    def _build_version_root(
        self,
        *,
        base_bucket: str,
        instrument_group: str,
        symbol_pair: str,
    ) -> Path:
        return (
            self.data_directory
            / base_bucket
            / self.train_data_manager.data_source
            / instrument_group
            / symbol_pair
            / str(self.sequence_length)
        )

    @staticmethod
    def _list_version_numbers(root: Path) -> list[int]:
        if not root.exists():
            return []
        versions: list[int] = []
        for child in root.iterdir():
            if child.is_dir() and child.name.isdigit():
                versions.append(int(child.name))
        return sorted(versions)

    @staticmethod
    def _read_metadata(metadata_path: Path) -> dict[str, object]:
        with metadata_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @staticmethod
    def _write_metadata(metadata_path: Path, metadata: dict[str, object]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle, indent=2)

    @staticmethod
    def _metadata_matches(
        existing_metadata: dict[str, object],
        current_metadata: dict[str, object],
    ) -> bool:
        keys_to_match = {
            "data_source",
            "instrument_group",
            "symbol_pair",
            "timeframe",
            "sequence_length",
            "stride",
            "feature_columns",
            "indicator_specs",
            "train_row_count",
            "train_example_count",
            "train_start_time",
            "train_end_time",
            "eval_row_count",
            "eval_example_count",
            "eval_start_time",
            "eval_end_time",
        }
        return all(existing_metadata.get(key) == current_metadata.get(key) for key in keys_to_match)

    def _count_examples(self, dataframe: pd.DataFrame) -> int:
        return max(0, ((len(dataframe) - self.sequence_length) // self.stride) + 1)

    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        normalized = timeframe.strip().lower()
        if normalized not in GenerateTrainData.TIMEFRAME_ALIASES:
            supported = ", ".join(sorted(GenerateTrainData.TIMEFRAME_ALIASES))
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported values: {supported}.")
        return normalized

    @staticmethod
    def _isoformat(value: pd.Timestamp) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        unit = timeframe[-1]
        quantity = int(timeframe[:-1])
        if unit == "m":
            return quantity
        if unit == "h":
            return quantity * 60
        if unit == "d":
            return quantity * 1440
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
