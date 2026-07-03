from __future__ import annotations
import sys
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.eager.context import executor
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import keras
import os


from src.data_manager import DataManager
from src.settings import Settings
from src.schemas import TARGET_MODEL_TYPES, TimeBasedTarget, PointsBasedTarget, SymbolProperties
from src.aux_model_manager import AuxilaryModelManager as AX

_TFRECORD_OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=1)
_SHARD_COUNT = 16
_CPU_COUNT = os.cpu_count() or 4

BASE_FEATURE_COLUMNS = (
        "open", "close", "high", "low",
        "real_volume", "spread", "tick_volume",
    )

TIMEFRAME_ALIASES = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1d",
    }

PREPROCESS_INSTANCE = None
SEQUENCE_LENGTH = 0
STRIDE = 1
TARGET_MODEL = None
PREPROCESS_DATA = False
FILTER_BY_MODEL = False
FILTER_LABEL_ID = 0
FILTER_AUX_MODEL = None
DEBUG_MODE = bool(int(os.getenv("DEBUG_MODE", "0")))

class GenerateTrainData:
    """Generate versioned TensorFlow training and evaluation TFRecords."""

    _WRITE_CHUNK_SIZE = 4_000

    def __init__(
        self,
        train_base_bucket: str = "forexgrand-train",
        eval_base_bucket: str = "forexgrand-eval",
        preprocess_data: bool = False,
        preprocess_layer: Optional[keras.Model] = None,
        chunk_size: int = 1000,
        use_dataframe_format: bool = False,
        filter_by_model: bool = False,
        filter_model_id: Optional[str] = None,
        target_label: int = 0,
        write_parallelism: Optional[int] = None,
    ) -> None:
        self.settings = Settings()
        self.feature_columns = BASE_FEATURE_COLUMNS
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
        self.preprocess_model_path = None
        if preprocess_data and not preprocess_layer:
            raise ValueError("You need to attach a preprocess layer object if preprocess data is True.")
        
        self.preprocess_data = preprocess_data
        self.preprocess_layer = preprocess_layer
        self.chunk_size = chunk_size
        self.use_dataframe_format = use_dataframe_format
        self.filter_by_model = filter_by_model
        self.filter_model_id = filter_model_id
        self.filter_target_label_id = target_label
        self.filter_aux_model = None if not filter_by_model else AX(model_id=filter_model_id)
        self.write_parallelism = write_parallelism

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
        use_dataframe_format: bool | None = None,
        split: str = "train",
    ) -> Path:
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_model = target_model

        use_df_format = use_dataframe_format if use_dataframe_format is not None else self.use_dataframe_format

        pair_name = symbol_pair.strip()
        group_name = instrument_group.strip()
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
            bucket_name=bucket_name,  split=split,
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
            symbol_pair=pair_name, version_number=version_number, split=split,
        )
        ##save preprocess layer to disk
        self.preprocess_model_path = output_dir / "preprocess_layer.keras"
        if self.preprocess_data:
            os.makedirs(output_dir, exist_ok=True)
            keras.saving.save_model(self.preprocess_layer, self.preprocess_model_path)
        
        self.generate_train_data_examples(_frame, output_dir=output_dir, use_dataframe_format=use_df_format)

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
        use_dataframe_format: bool | None = None,
    ) -> tuple[Path, Path]:
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_model = target_model
        

        use_df_format = use_dataframe_format if use_dataframe_format is not None else self.use_dataframe_format

        pair_name = symbol_pair.strip()
        group_name = instrument_group.strip()
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

        ##save preprocess layer to disk
        self.preprocess_model_path = train_dir / "preprocess_layer.keras"
        if self.preprocess_data:
            os.makedirs(train_dir, exist_ok=True)
            keras.saving.save_model(self.preprocess_layer, self.preprocess_model_path)
        
        self.generate_train_data_examples(train_frame, output_dir=train_dir, use_dataframe_format=use_df_format)
        self.generate_eval_data_examples(eval_frame, output_dir=eval_dir, use_dataframe_format=use_df_format)         

        self._write_metadata(train_dir / "metadata.json", metadata)
        self._write_metadata(eval_dir / "metadata.json", metadata)
        return train_dir, eval_dir

    def load_target_data(
        self,
        bucket_name: str,
        symbol_pair: str,
        instrument_group: str,
        sequence_length: int,
        stride: int,
        target_model: TARGET_MODEL_TYPES
      ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_model = target_model

        pair_name = symbol_pair.strip()
        group_name = instrument_group.strip()

        if not pair_name:
            raise ValueError("symbol_pair is required.")
        if not group_name:
            raise ValueError("instrument_group is required.")

        bucket_manager = self._build_data_manager(bucket_name)
        _raw_frame, self.train_properties = bucket_manager.load_data(pair_name, group_name)
        
        targets_dict = self._build_target_data(_raw_frame, self.train_properties)
        
        return targets_dict[0]

    def generate_train_data_examples(self, dataframe: pd.DataFrame, *, output_dir: Path, use_dataframe_format: bool = False) -> Path:
        if use_dataframe_format:
            return self._save_sequence_data_to_dataframe(dataframe, output_path=output_dir, split="train", symbol_properties=self.train_properties)
        else:
            self._write_examples_sharded(dataframe, output_path=output_dir, split="train", symbol_properties=self.train_properties)
            return output_dir

    def generate_eval_data_examples(self, dataframe: pd.DataFrame, *, output_dir: Path, use_dataframe_format: bool = False) -> Path:
        if use_dataframe_format:
            return self._save_sequence_data_to_dataframe(dataframe, output_path=output_dir, split="eval", symbol_properties=self.eval_properties)
        else:
            self._write_examples_sharded(dataframe, output_path=output_dir, split="eval", symbol_properties=self.eval_properties)
            return output_dir

    # ------------------------------------------------------------------ #
    #  Core write path                                                    #
    # ------------------------------------------------------------------ #

    def _build_shard_paths(self, output_dir: Path, split: str, idx_start: int=0, end_idx=16) -> list[Path]:
        return [
            output_dir / f"{split}_{i:05d}_of_{idx_start+end_idx:05d}.gz"
            for i in range(idx_start, idx_start+end_idx)
        ]

    def _write_examples_sharded(
        self,
        dataframe: pd.DataFrame,
        symbol_properties: SymbolProperties,
        *,
        output_path: list[Path],
        split: str, 
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)

        target_counts, target_seq_length = self._count_targets(dataframe)
        num_examples  = self._count_examples(dataframe)
        target_counts = min(target_counts, num_examples) if target_counts is not None else num_examples
        
        print("Examples counts:",target_counts)
        
        seq = self.sequence_length
        stride = self.stride
        chunk_size = self.chunk_size
        import math

        chunks = max(1, math.ceil(target_counts/self.chunk_size))
        next_start_idx = seq
        chunked_indices = []

        for ch_idx in range(chunks):
            start_idx = next_start_idx
            end_idx   = start_idx + chunk_size*self.stride - 1
            if target_seq_length is not None:
                next_start_idx = end_idx + 1
                end_idx = end_idx + target_seq_length - 1
            else:
                next_start_idx = end_idx + 1

            chunked_indices.append(slice(start_idx - seq, end_idx))
        
        output_paths = self._build_shard_paths(output_path, split, 0, len(chunked_indices))
        num_shards = len(output_paths)
        num_shards = min(num_shards, num_examples)

        _worker_initializer(self.sequence_length, stride, self.target_model, self.preprocess_data, self.preprocess_model_path,
                           self.filter_by_model, self.filter_target_label_id, self.filter_model_id)

        write_parallelism = self._resolve_write_parallelism(len(chunked_indices))
        print("Write parallelism:", write_parallelism)

        def build_arg(idx: int, chunk_: slice) -> tuple[str, pd.DataFrame, SymbolProperties]:
            return (
                str(output_paths[idx]),
                dataframe.iloc[chunk_],
                symbol_properties,
            )

        if write_parallelism == 1:
            for idx, chunk_ in enumerate(chunked_indices):
                arg = build_arg(idx, chunk_)
                try:
                    process_save_data_tf(arg)
                except Exception as e:
                    print("Error encountered processing: ", arg[0], ": Error-", str(e))
                    raise e
            return

        pending = {}
        max_pending = write_parallelism * 2
        with ThreadPoolExecutor(max_workers=write_parallelism) as pool:
            for idx, chunk_ in enumerate(chunked_indices):
                while len(pending) >= max_pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        output_file = pending.pop(future)
                        try:
                            future.result()
                        except Exception as e:
                            print("Error encountered processing: ", output_file, ": Error-", str(e))
                            raise e

                arg = build_arg(idx, chunk_)
                pending[pool.submit(process_save_data_tf, arg)] = arg[0]

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    output_file = pending.pop(future)
                    try:
                        future.result()
                    except Exception as e:
                        print("Error encountered processing: ", output_file, ": Error-", str(e))
                        raise e

    def _resolve_write_parallelism(self, chunks: int) -> int:
        if chunks <= 1:
            return 1

        configured = self.write_parallelism
        if configured is None:
            env_value = os.getenv("WRITE_PARALLELISM")
            configured = int(env_value) if env_value else min(_CPU_COUNT, chunks)

        return max(1, min(int(configured), chunks))

    def _save_sequence_data_to_dataframe(
        self,
        dataframe: pd.DataFrame,
        symbol_properties: SymbolProperties,
        *,
        output_path: Path,
        split: str,
    ) -> Path:
        """Save sequence data as a parquet file instead of TFRecords."""
        output_path.mkdir(parents=True, exist_ok=True)
        target_counts, target_seq_length = self._count_targets(dataframe)
        num_examples = self._count_examples(dataframe)
        target_counts = min(target_counts, num_examples) if target_counts is not None else num_examples
        
        print("Examples counts:", target_counts)
        
        seq = self.sequence_length
        stride = self.stride
        chunk_size = self.chunk_size
        import math
        
        chunks = max(1, math.ceil(target_counts / self.chunk_size))
        next_start_idx = seq
        
        all_sequence_data = {}
        _worker_initializer(self.sequence_length, stride, self.target_model, self.preprocess_data, self.preprocess_model_path,
                           self.filter_by_model, self.filter_target_label_id, self.filter_model_id)
       
        for ch_idx in range(chunks):
            start_idx = next_start_idx
            end_idx = start_idx + chunk_size * self.stride - 1
            if target_seq_length is not None:
                next_start_idx = end_idx + 1
                end_idx = end_idx + target_seq_length - 1
            else:
                next_start_idx = end_idx + 1
            
            features_data = _build_sequence_data(dataframe.iloc[start_idx - seq:end_idx], symbol_properties)
            data_features = build_process_data(features_data)
           
            # Accumulate sequence data
            for key, values in data_features.items():
                if key != "num_examples":
                    values_np = values.numpy() if isinstance(values, tf.Tensor) else values
                    if key not in all_sequence_data:
                        all_sequence_data[key] = values_np
                    else:
                        all_sequence_data[key] = np.concatenate([all_sequence_data[key], values_np], axis=0)

        # Convert to DataFrame
        result_df = pd.DataFrame(all_sequence_data)
        
        # Save as parquet
        parquet_path = output_path / f"{split}_data.parquet"
        result_df.to_parquet(parquet_path, index=False)
        
        return output_path

    def _count_targets(self, dataframe: pd.DataFrame):
        if self.target_model:
            target_df_len = len(dataframe) - self.target_model.stop_minutes + 1
            targets = max(0, (target_df_len - self.sequence_length)//self.stride + 1)
            return targets, self.target_model.stop_minutes
        else:
            return None, None    

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
                    _calculate_points_diff(
                        np.max(target_sequences["high"][self.sequence_length :: self.stride], axis=-1),
                        pos_prices, points, self.target_model.mode,
                    ).astype("float32"), axis=-1,
                ),
                "target_lowest": np.expand_dims(
                    _calculate_points_diff(
                        pos_prices,
                        np.min(target_sequences["low"][self.sequence_length :: self.stride], axis=-1),
                        points, self.target_model.mode,
                    ).astype("float32"), axis=-1,
                ),
                "target_value": np.expand_dims(
                    _calculate_points_diff(
                        pos_prices,
                        target_sequences["close"][self.sequence_length :: self.stride, -1],
                        points, self.target_model.mode,
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
        required_columns = {"time", *BASE_FEATURE_COLUMNS}
        missing_columns = required_columns.difference(normalized.columns)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataframe is missing required columns: {missing_list}.")

        normalized["time"] = pd.to_datetime(normalized["time"], errors="coerce")
        if normalized["time"].isna().any():
            raise ValueError("The 'time' column contains invalid datetime values.")

        normalized = normalized.sort_values("time").drop_duplicates(subset="time", keep="last")
        normalized = normalized.reset_index(drop=True)
        return normalized[["time", *BASE_FEATURE_COLUMNS]]

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
            "filter_by_model":self.filter_by_model,
"filter_model_id":self.filter_model_id,
"filter_target_label_id":self.filter_target_label_id,
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

    def _find_existing_version_path_single(self, *, symbol_pair, instrument_group, metadata, bucket_name: str, split:str="train"):
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
            train_dir = train_root / str(version_number) / split
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


def _worker_initializer(seq_len, stride, target_model, preprocess_data, 
                         preprocess_model_path, filter_by_model, filter_label_id, filter_model_id):
    import os
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import keras

    # Declare ALL globals you want to set
    global PREPROCESS_INSTANCE, SEQUENCE_LENGTH, STRIDE, TARGET_MODEL
    global PREPROCESS_DATA, FILTER_BY_MODEL, FILTER_LABEL_ID, FILTER_AUX_MODEL

    SEQUENCE_LENGTH = seq_len
    STRIDE          = stride
    TARGET_MODEL    = target_model
    PREPROCESS_DATA = preprocess_data
    FILTER_BY_MODEL = filter_by_model
    FILTER_LABEL_ID = filter_label_id

    if preprocess_data and preprocess_model_path and not PREPROCESS_INSTANCE:
        PREPROCESS_INSTANCE = keras.saving.load_model(preprocess_model_path)

    if filter_by_model and filter_model_id and not FILTER_AUX_MODEL:
        FILTER_AUX_MODEL = AX(filter_model_id)

    print(f"[WORKER INIT] PID={os.getpid()} seq={SEQUENCE_LENGTH} model={PREPROCESS_INSTANCE}, filter_modeel={FILTER_AUX_MODEL}", flush=True)

# ------------------------------------------------------------------ #
#  Module-level helpers (must be top-level for pickling by workers)  #
# ------------------------------------------------------------------ #
def build_process_data(features):
    features = {key:tf.constant(value) for key,value in features.items() if key != "num_examples"}
    
    if FILTER_BY_MODEL:
        features = filter_data_by_model(features)

    preprocessed = {}
    if not PREPROCESS_DATA:
        preprocessed = features
    else:
        tf_data = tf.data.Dataset.from_tensor_slices(features)
        batch_size = int(os.getenv("BATCH_SIZE","128"))
        processed = tf_data.batch(batch_size).map(_preprocess_batch_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        batch_values = {}
        for batch in processed.take(-1): 
            for key, values in batch.items():
                arr = np.atleast_1d(values.numpy())
                batch_values.setdefault(key, []).append(arr)

        preprocessed = {
            key: np.concatenate(values, axis=0)
            for key, values in batch_values.items()
        }
    return preprocessed

def filter_data_by_model(data_in:dict):
    predictions = FILTER_AUX_MODEL.predict(data_in)
    
    if DEBUG_MODE:
        print(np.unique_counts(predictions))
        
    valid_mask  = np.array(predictions)==FILTER_LABEL_ID
    
    valid_ds = {
      key:value[valid_mask]
      for key,value in data_in.items()
    }
    return valid_ds

def _normalize_feature_lengths(
    feature_data: dict[str, np.ndarray | tf.Tensor],
    output_file: str | Path,
) -> tuple[dict[str, np.ndarray | tf.Tensor], int]:
    lengths = {
        key: int(value.shape[0])
        for key, value in feature_data.items()
        if len(value.shape) > 0
    }
    if not lengths:
        return feature_data, 0

    num_examples = min(lengths.values())
    if num_examples <= 0:
        if DEBUG_MODE:
            print(f"No examples to save for {output_file}. Feature lengths: {lengths}")
        return {}, 0

    if len(set(lengths.values())) > 1 and DEBUG_MODE:
        print(
            f"Feature length mismatch for {output_file}; clipping to {num_examples}. "
            f"Feature lengths: {lengths}",
            flush=True,
        )

    normalized = {
        key: value[:num_examples]
        for key, value in feature_data.items()
    }
    return normalized, num_examples

@tf.function
def _preprocess_batch_data(data):
    result = PREPROCESS_INSTANCE(data, training=True)
    return result

def _build_sequence_data(
    dataframe: pd.DataFrame,
    symbol_properties: SymbolProperties | None = None,
) -> dict[str, np.ndarray | int]:
    if len(dataframe) < SEQUENCE_LENGTH:
        raise ValueError(
            f"At least {SEQUENCE_LENGTH} rows are required to create sequences."
        )

    # ── time ──────────────────────────────────────────────────────────────── #
    time_values = (
        dataframe["time"].astype("datetime64[s]").astype("int64").to_numpy(copy=False)
    ).astype(np.int64, copy=False)

    # ── features ──────────────────────────────────────────────────────────── #
    feature_matrix = np.stack(
        [dataframe[col].to_numpy(dtype=np.float32, copy=False) for col in BASE_FEATURE_COLUMNS],
        axis=1,
    )
    windowed = np.lib.stride_tricks.sliding_window_view(
        feature_matrix, (SEQUENCE_LENGTH, feature_matrix.shape[1])
    )[:, 0, :, :]

    if STRIDE > 1:
        windowed = windowed[:: STRIDE]

    time_windowed = _window_array(time_values, dtype=np.int64)
    sequence_data: dict[str, np.ndarray | int] = {"time": time_windowed}
    for i, col in enumerate(BASE_FEATURE_COLUMNS):
        sequence_data[col] = np.ascontiguousarray(windowed[:, :, i])

    num_examples = time_windowed.shape[0]
    # ── targets (optional) ────────────────────────────────────────────────── #
    if TARGET_MODEL is not None:
        if symbol_properties is None:
            raise ValueError(
                "symbol_properties is required when target_model is set."
            )

        if isinstance(TARGET_MODEL, TimeBasedTarget):
            required_cols = ["high", "low", "close"]
            target_seq_length = TARGET_MODEL.stop_minutes

            target_sequences = {
                col: np.lib.stride_tricks.sliding_window_view(
                    dataframe[col].to_numpy(), target_seq_length
                )
                for col in required_cols
            }

            # Entry price: closing price at the last bar of each input window.
            # Shape: (N,)
            close_windows = np.lib.stride_tricks.sliding_window_view(
                dataframe["close"].to_numpy(), target_seq_length
            )
            pos_prices = close_windows[SEQUENCE_LENGTH - 1 : -1 : STRIDE, 0]
            points = symbol_properties.point_size
            if TARGET_MODEL.mode != "prices":
                sequence_data["target_highest"] = np.expand_dims(
                    _calculate_points_diff(
                        np.max(
                            target_sequences["high"][SEQUENCE_LENGTH :: STRIDE],
                            axis=-1,
                        ),
                        pos_prices,
                        points, TARGET_MODEL.mode
                    ).astype("float32"),
                    axis=-1,
                )
                sequence_data["target_lowest"] = np.expand_dims(
                    _calculate_points_diff(
                        pos_prices,
                        np.min(
                            target_sequences["low"][SEQUENCE_LENGTH :: STRIDE],
                            axis=-1,
                        ),
                        points, TARGET_MODEL.mode
                    ).astype("float32"),
                    axis=-1,
                )
                sequence_data["target_value"] = np.expand_dims(
                    _calculate_points_diff(
                        pos_prices,
                        target_sequences["close"][SEQUENCE_LENGTH :: STRIDE, -1],
                        points, TARGET_MODEL.mode
                    ).astype("float32"),
                    axis=-1,
                )
                sequence_data["pos_prices"] = pos_prices
                target_exmps = sequence_data["target_value"].shape[0]
                num_examples = min(num_examples, target_exmps)
            elif TARGET_MODEL.mode=="prices":
                sequence_data['target_high'] = target_sequences["high"][SEQUENCE_LENGTH :: STRIDE]
                sequence_data['target_low']  = target_sequences['low'][SEQUENCE_LENGTH :: STRIDE]
                sequence_data['target_close']= target_sequences['close'][SEQUENCE_LENGTH :: STRIDE]
                target_exmps = sequence_data["target_high"].shape[0]
                num_examples = min(num_examples, target_exmps)

        elif isinstance(TARGET_MODEL, PointsBasedTarget):
            raise NotImplementedError("mode not implemented: use TimeBasedTarget.")
    for col in sequence_data:
        sequence_data[col] = sequence_data[col][:num_examples]

    sequence_data["num_examples"] = num_examples
    return sequence_data

def _calculate_points_diff(price_arr1, price_arr2, points_size, mode: str="points") -> np.ndarray:
    if mode=="points":
      return (price_arr1 - price_arr2) // points_size
    else:
      return (price_arr1 - price_arr2)

def _window_array(values: np.ndarray, *, dtype: type[np.generic]) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, SEQUENCE_LENGTH)
    if STRIDE > 1:
        windows = windows[:: STRIDE]
    return np.ascontiguousarray(windows, dtype=dtype)
    
def process_save_data_tf(args):
    (output_file, df, symbol_properties) = args
    features_data = _build_sequence_data(df, symbol_properties)
    data_features = build_process_data(features_data)
    data_features, data_length = _normalize_feature_lengths(data_features, output_file)
    if data_length == 0:
        if DEBUG_MODE:
           print("No features to save.")
        return
    _write_shard_worker((output_file, data_features, SEQUENCE_LENGTH, data_length))

def _partition_into_shards(total: int, num_shards: int) -> list[range]:
    """Divide [0, total) into num_shards roughly equal contiguous ranges."""
    base, remainder = divmod(total, num_shards)
    ranges = []
    start = 0
    for i in range(num_shards):
        if start<total:
          end = start + base + (1 if i < remainder else 0)
          ranges.append(range(start, end))
          start = end
        else:
          ranges.append(range(0,0))

    return ranges

def _get_features_from_feature_frame(feature_raw, target_lists):
    pass

def _write_shard_worker(args: tuple) -> None:
    """
    Top-level function (required for ProcessPoolExecutor pickling).
    Builds and writes all TFExamples for a single shard.
    """
    (
        output_path,
        feature_data,
        seq,
        num_examples,
    ) = args

    # import tensorflow as tf  # re-import in worker process
    feature_data, num_examples = _normalize_feature_lengths(feature_data, output_path)
    if num_examples == 0:
        return

    options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=1)

    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for example_idx in range(num_examples):
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
