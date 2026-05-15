from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os

from torch import prelu

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
        preprocess_data: bool = False,
        preprocess_layer: Optional[keras.Model] = None
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
        if preprocess_data and not preprocess_layer:
            raise ValueError("You need to attach a preprocess layer object if preprocess data is True.")
        self.preprocess_data = preprocess_data
        self.preprocess_layer = preprocess_layer

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
        metadata = self._build_metadata(
            symbol_pair=pair_name, instrument_group=group_name,
            train_df=_raw_frame, eval_df=_raw_frame, target_type=target_model,
        )
        existing_paths = self._find_existing_version_path_single(
            symbol_pair=pair_name, instrument_group=group_name, metadata=metadata,
            bucket_name=bucket_name,
        )
        
        if existing_paths is not None and not hot_reload:
            print("Examples in data: ",self._count_examples(_raw_frame))
            return existing_paths
            
        _frame = self._prepare_feature_frame(_raw_frame)
        
        version_number = self._resolve_next_version_number(
            symbol_pair=pair_name, instrument_group=group_name,
        )
        output_dir = self._build_output_dir(
            base_bucket=bucket_name, instrument_group=group_name,
            symbol_pair=pair_name, version_number=version_number, split="train",
        )

        # self.generate_train_data_examples(_frame, output_dir=output_dir)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self.generate_train_data_examples, _frame, output_dir=output_dir): "train",
            }
            for future in as_completed(futures):
                split = futures[future]
                exc = future.exception()
                if exc:
                    raise RuntimeError(f"{split} generation failed: {exc}") from exc
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
        
        metadata = self._build_metadata(
            symbol_pair=pair_name, instrument_group=group_name,
            train_df=train_raw_frame, eval_df=eval_raw_frame, target_type=target_model,
        )

        existing_paths = self._find_existing_version_paths(
            symbol_pair=pair_name, instrument_group=group_name, metadata=metadata,
        )
        if existing_paths is not None and not hot_reload:
            print("Train Examples in data: ",self._count_examples(train_raw_frame))
            print("Eval Examples in data: ",self._count_examples(eval_raw_frame))
            return existing_paths
        
        train_frame = self._prepare_feature_frame(train_raw_frame)
        eval_frame = self._prepare_feature_frame(eval_raw_frame)

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

        features_data = self._build_sequence_data(dataframe)
        num_examples  = features_data['num_examples']
        target_data: dict[str, np.ndarray] = {}
        target_counts = num_examples
        if self.target_model:
            target_data, _counts = self._build_target_data(
                dataframe=dataframe, symbol_properties=symbol_properties
            )
            target_counts = min(num_examples, _counts)
        
        print("Examples:", num_examples, " Targets:", target_counts)

        data_features = self.build_process_data(features_data, target_data, target_counts)
        seq = self.sequence_length
        stride = self.stride
        # ✅ Partition examples across shards; each shard owns its example indices
        shard_index_ranges = _partition_into_shards(target_counts, num_shards)
        
        # ✅ Serialize each shard's examples in parallel worker processes
        worker_args = [
            (
                shard_idx,
                str(output_paths[shard_idx]),
                shard_index_ranges[shard_idx],
                data_features,
                seq,
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
    def build_process_data(self, features, targets, target_count):
        features = {key:tf.constant(value[:target_count]) for key,value in features.items() if key != "num_examples"}
        for col in targets:
            features[col] = tf.squeeze(targets[col])

        preprocessed = {}
        if not self.preprocess_data:
            preprocessed = features
        else:
            tf_data = tf.data.Dataset.from_tensor_slices(features)
            batch_size = int(os.getenv("BATCH_SIZE","128"))
            tf_data = tf_data.batch(batch_size)
            
            preprocessed = {}
            ##run first batch to store keys
            first_batch_done = False
            for batch in tf_data.take(-1):
                processed = self._preprocess_batch_data(batch)
                for key, values in processed.items():
                    if not first_batch_done:
                        preprocessed[key] = []

                    preprocessed[key] = np.concatenate([preprocessed[key],values.numpy()], axis=0)

                if not first_batch_done:
                    first_batch_done = True

        return preprocessed

    @tf.function
    def _preprocess_batch_data(self, data):
        return self.preprocess_layer(data, training=True)

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
                        target_sequences["close"][self.sequence_length :: self.stride, -1],
                        points,
                    ).astype("float32"), axis=-1,
                ),
            }
            return target_cols, pos_prices.shape[0]

        elif isinstance(self.target_model, PointsBasedTarget):
            raise NotImplementedError("mode not implemented: use TimeBasedTarget.")

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
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

    def _build_output_dir(self, *, base_bucket, instrument_group, symbol_pair, version_number, split) -> Path:
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

    def _build_metadata(self, *, symbol_pair, instrument_group, train_df, eval_df, target_type=None) -> dict:
        
        return {
            "data_source": self.train_data_manager.data_source,
            "instrument_group": instrument_group,
            "symbol_pair": symbol_pair,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "target_type": str(target_type.type) if target_type is not None else None,
            "target_params": target_type.model_dump(exclude={"type"}) if target_type is not None else None,
            "train_start_time": train_df["time"].iloc[0].isoformat(),
            "train_end_time": train_df["time"].iloc[-1].isoformat(),
            "eval_start_time": eval_df["time"].iloc[0].isoformat(),
            "eval_end_time": eval_df["time"].iloc[-1].isoformat(),
            "processed_data":self.preprocess_data,
        }

    def _find_existing_version_paths(self, *, symbol_pair, instrument_group, metadata):
        train_root = self._build_version_root(
            base_bucket=self.train_base_bucket,
            instrument_group=instrument_group, symbol_pair=symbol_pair,
        )
        eval_root = self._build_version_root(
            base_bucket=self.eval_base_bucket,
            instrument_group=instrument_group, symbol_pair=symbol_pair,
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
            train_dir = train_root / str(version_number) / "train"
            eval_dir = eval_root / str(version_number) / "eval"
            train_metadata_path = train_dir / "metadata.json"
            eval_metadata_path = eval_dir / "metadata.json"

            train_shards = list(train_dir.glob("train_*.gz"))
            eval_shards = list(eval_dir.glob("eval_*.gz"))
            if not train_shards or not eval_shards:
                continue
            if not train_metadata_path.exists() or not eval_metadata_path.exists():
                continue

            train_metadata = self._read_metadata(train_metadata_path)
            eval_metadata = self._read_metadata(eval_metadata_path)
            if train_metadata != eval_metadata:
                continue
            if self._metadata_matches(train_metadata, metadata):
                return train_dir, eval_dir

        return None

    def _resolve_next_version_number(self, *, symbol_pair, instrument_group) -> int:
        train_root = self._build_version_root(
            base_bucket=self.train_base_bucket,
            instrument_group=instrument_group, symbol_pair=symbol_pair,
        )
        eval_root = self._build_version_root(
            base_bucket=self.eval_base_bucket,
            instrument_group=instrument_group, symbol_pair=symbol_pair,
        )
        version_numbers = (
            self._list_version_numbers(train_root) + self._list_version_numbers(eval_root)
        )
        return (max(version_numbers) + 1) if version_numbers else 1

    def _find_existing_version_path_single(self, *, symbol_pair, instrument_group, metadata, bucket_name: str):
        train_root = self._build_version_root(
            base_bucket=bucket_name,
            instrument_group=instrument_group, symbol_pair=symbol_pair,
        )

        if not train_root.exists():
            return None

        common_versions = sorted(
            set(self._list_version_numbers(train_root)),
            reverse=True,
        )

        for version_number in common_versions:
            train_dir = train_root / str(version_number) / "train"
            train_metadata_path = train_dir / "metadata.json"

            train_shards = list(train_dir.glob("train_*.gz"))
            if not train_shards:
                continue
            if not train_metadata_path.exists():
                continue

            train_metadata = self._read_metadata(train_metadata_path)
            
            if self._metadata_matches(train_metadata, metadata):
                return train_dir

        return None

    def _build_version_root(self, *, base_bucket, instrument_group, symbol_pair) -> Path:
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
    def _read_metadata(metadata_path: Path) -> dict:
        with metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _write_metadata(metadata_path: Path, metadata: dict) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    @staticmethod
    def _metadata_matches(existing_metadata: dict, current_metadata: dict) -> bool:
        keys_to_match = {
            "data_source", "instrument_group", "symbol_pair", "timeframe",
            "sequence_length", "stride", "target_type", "target_params", 
            "train_start_time", "train_end_time",
            "eval_start_time", "eval_end_time",
        }
        return all(
            existing_metadata.get(key) == current_metadata.get(key) for key in keys_to_match
        )

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


# ------------------------------------------------------------------ #
#  Module-level helpers (must be top-level for pickling by workers)   #
# ------------------------------------------------------------------ #

def _partition_into_shards(total: int, num_shards: int) -> list[range]:
    """Divide [0, total) into num_shards roughly equal contiguous ranges."""
    base, remainder = divmod(total, num_shards)
    ranges = []
    start = 0
    for i in range(num_shards):
        end = start + base + (1 if i < remainder else 0)
        ranges.append(range(start, end))
        start = end
    return ranges

def _get_features_from_feature_frame(feature_raw, target_lists):
    pass

def _write_shard_worker(args: tuple) -> None:
    """
    Top-level function (required for ProcessPoolExecutor pickling).
    Builds and writes all TFExamples for a single shard.
    """
    (
        shard_idx,
        output_path,
        example_range,
        feature_data,
        seq,
    ) = args

    import tensorflow as tf  # re-import in worker process

    options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=1)

    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for example_idx in example_range:
            features_data = {key:tf.reshape(values[example_idx], shape=(-1,)).numpy() for key, values in feature_data.items()}
            features: dict[str, tf.train.Example] = {}
            
            for feature,value in features_data.items():
                if type(value[0]) in [float,np.float32, np.float64, tf.float32]:
                   features[feature] = write_float_example(value)
                elif type(value[0]) in [int, np.int32, np.int64]:
                    features[feature] = write_int_example(value)
                elif type(value[0]) in [str, np.strings]:
                    features[feature] = write_str_example(value)
                else:
                    raise Exception("Feature type cannot not be stored, feature: ",feature," type: ",type(value[0]))
            
            writer.write(
                tf.train.Example(
                    features=tf.train.Features(feature=features)
                ).SerializeToString()
            )

def write_float_example(vals):
    return tf.train.Feature(float_list=tf.train.FloatList(value=vals))

def write_int_example(vals):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=vals))

def write_str_example(vals):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=vals))