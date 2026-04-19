from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_manager import DataManager
from src.settings import Settings


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
        data_manager: DataManager | type[DataManager],
        *,
        data_manager_kwargs: Optional[dict] = None,
        settings: Optional[Settings] = None,
        sequence_length: int,
        timeframe: str = "1m",
        stride: int = 1,
        indicators: Optional[Sequence[dict[str, Any]]] = None,
        train_base_bucket: str = "forexgrand-train",
        eval_base_bucket: str = "forexgrand-eval",
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if stride <= 0:
            raise ValueError("stride must be greater than zero.")

        self.settings = settings or Settings()
        self.sequence_length = sequence_length
        self.timeframe = self._normalize_timeframe(timeframe)
        self.stride = stride
        self.indicator_specs = self._normalize_indicator_specs(indicators or [])
        self.feature_columns = self.BASE_FEATURE_COLUMNS + tuple(
            column_name
            for spec in self.indicator_specs
            for column_name in spec["column_names"]
        )
        self.train_base_bucket = train_base_bucket.strip() or "forexgrand-train"
        self.eval_base_bucket = eval_base_bucket.strip() or "forexgrand-eval"
        manager_kwargs = data_manager_kwargs or {}
        self.train_data_manager = self._build_data_manager(
            data_manager=data_manager,
            data_manager_kwargs=manager_kwargs,
            base_bucket_name=self.train_base_bucket,
        )
        self.eval_data_manager = self._build_data_manager(
            data_manager=data_manager,
            data_manager_kwargs=manager_kwargs,
            base_bucket_name=self.eval_base_bucket,
        )
        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()

    def load_data(
        self,
        symbol_pair: str,
        instrument_group: str,
        *,
        hot_reload: bool = False,
    ) -> tuple[Path, Path]:
        pair_name = symbol_pair.strip().upper()
        group_name = instrument_group.strip().lower()
        if not pair_name:
            raise ValueError("symbol_pair is required.")
        if not group_name:
            raise ValueError("instrument_group is required.")

        train_raw_frame, _ = self.train_data_manager.load_data(pair_name, group_name)
        eval_raw_frame, _ = self.eval_data_manager.load_data(pair_name, group_name)
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
        self._write_examples(dataframe, output_path=output_path)
        return output_path

    def generate_eval_data_examples(
        self,
        dataframe: pd.DataFrame,
        *,
        output_path: Path,
    ) -> Path:
        self._write_examples(dataframe, output_path=output_path)
        return output_path

    def _build_data_manager(
        self,
        *,
        data_manager: DataManager | type[DataManager],
        data_manager_kwargs: dict,
        base_bucket_name: str,
    ) -> DataManager:
        if isinstance(data_manager, DataManager):
            manager_class = type(data_manager)
            return manager_class(
                settings=data_manager.settings,
                data_source=data_manager.data_source,
                base_bucket_name=base_bucket_name,
            )
        if isinstance(data_manager, type) and issubclass(data_manager, DataManager):
            manager_kwargs = dict(data_manager_kwargs)
            manager_kwargs["base_bucket_name"] = base_bucket_name
            return data_manager(**manager_kwargs)
        raise TypeError("data_manager must be a DataManager instance or DataManager subclass.")

    def _prepare_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized_frame = self._prepare_dataframe(dataframe)
        timeframe_frame = self._apply_timeframe(normalized_frame)
        feature_frame = self._apply_indicators(normalized_frame, timeframe_frame)
        if len(feature_frame) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows are required to create sequences."
            )
        return feature_frame

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

        normalized["time"] = pd.to_datetime(normalized["time"], utc=True, errors="coerce")
        if normalized["time"].isna().any():
            raise ValueError("The 'time' column contains invalid datetime values.")

        normalized = normalized.sort_values("time").drop_duplicates(subset="time", keep="last")
        normalized = normalized.reset_index(drop=True)
        return normalized[["time", *self.BASE_FEATURE_COLUMNS]]

    def _apply_timeframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.timeframe == "1m":
            return dataframe.reset_index(drop=True)

        resample_rule = self.TIMEFRAME_ALIASES[self.timeframe]
        indexed = dataframe.set_index("time")
        resampled = indexed.resample(resample_rule).agg(
            {
                "open": "first",
                "close": "last",
                "high": "max",
                "low": "min",
                "real_volume": "sum",
                "spread": "last",
                "tick_volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open", "close", "high", "low", "spread"])
        return resampled.reset_index()

    def _apply_indicators(
        self,
        base_frame: pd.DataFrame,
        main_timeframe_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_frame = main_timeframe_frame.copy()
        if not self.indicator_specs:
            return feature_frame[["time", *self.feature_columns]]

        for spec in self.indicator_specs:
            indicator_frame = self._apply_timeframe_for_value(base_frame, spec["timeframe"])
            indicator_values = spec["function"](indicator_frame.copy(), **spec["kwargs"])
            normalized_indicator = self._normalize_indicator_output(
                indicator_values,
                indicator_frame.index,
                spec["column_names"],
                spec["buffers"],
            )
            aligned_frame = self._align_indicator_to_main_timeframe(
                indicator_frame[["time"]].reset_index(drop=True),
                normalized_indicator,
                main_timeframe_frame[["time"]],
                allow_exact_matches=spec["timeframe"] == self.timeframe,
            )
            feature_frame = feature_frame.merge(aligned_frame, on="time", how="left")

        feature_frame = feature_frame.dropna(subset=self.feature_columns).reset_index(drop=True)
        if len(feature_frame) < self.sequence_length:
            raise ValueError(
                "Indicator warmup removed too many rows to build sequences."
            )
        return feature_frame[["time", *self.feature_columns]]

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
            / self.timeframe
            / str(self.sequence_length)
            / str(version_number)
            / filename
        )

    def _write_examples(self, dataframe: pd.DataFrame, *, output_path: Path) -> None:
        sequence_data = self._build_sequence_data(dataframe)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
            for index in range(sequence_data["num_examples"]):
                example = self._build_tf_example({
                    column_name: values[index]
                    for column_name, values in sequence_data.items()
                    if column_name != "num_examples"
                })
                writer.write(example.SerializeToString())

    def _build_sequence_data(self, dataframe: pd.DataFrame) -> dict[str, np.ndarray | int]:
        if len(dataframe) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} rows are required to create sequences."
            )

        time_values = (
            dataframe["time"].astype("int64").to_numpy(copy=False) // 10**9
        ).astype(np.int64, copy=False)
        sequence_data: dict[str, np.ndarray | int] = {
            "time": self._window_array(time_values, dtype=np.int64),
        }
        for column in self.feature_columns:
            values = dataframe[column].to_numpy(dtype=np.float32, copy=False)
            sequence_data[column] = self._window_array(values, dtype=np.float32)

        sequence_data["num_examples"] = int(sequence_data["time"].shape[0])
        return sequence_data

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
            "timeframe": self.timeframe,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "feature_columns": list(self.feature_columns),
            "indicator_specs": self._metadata_indicator_specs(),
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
            / self.timeframe
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

    def _normalize_indicator_specs(
        self,
        indicators: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_specs: list[dict[str, Any]] = []
        seen_column_names: set[str] = set()
        seen_indicator_names: set[str] = set()
        main_minutes = self._timeframe_to_minutes(self.timeframe)

        for raw_spec in indicators:
            function = raw_spec.get("function")
            if not callable(function):
                raise ValueError("Each indicator spec must include a callable 'function'.")

            name = str(raw_spec.get("name", "")).strip().lower()
            if not name:
                raise ValueError("Each indicator spec must include a non-empty 'name'.")
            if name in seen_indicator_names:
                raise ValueError(f"Duplicate indicator name detected: {name}.")
            seen_indicator_names.add(name)

            indicator_timeframe = self._normalize_timeframe(str(raw_spec.get("timeframe", "")))
            if self._timeframe_to_minutes(indicator_timeframe) < main_minutes:
                raise ValueError(
                    f"Indicator timeframe '{indicator_timeframe}' cannot be lower than main timeframe '{self.timeframe}'."
                )

            buffers = raw_spec.get("buffers")
            if not isinstance(buffers, Sequence) or isinstance(buffers, (str, bytes)) or len(buffers) == 0:
                raise ValueError("Each indicator spec must include a non-empty 'buffers' list.")

            normalized_buffers = [int(buffer_index) for buffer_index in buffers]
            kwargs = dict(raw_spec.get("kwargs") or {})
            column_names = tuple(
                f"{name}_{indicator_timeframe}_{buffer_index}" for buffer_index in normalized_buffers
            )
            duplicate_names = seen_column_names.intersection(column_names)
            if duplicate_names:
                duplicates = ", ".join(sorted(duplicate_names))
                raise ValueError(f"Duplicate indicator column names detected: {duplicates}.")
            seen_column_names.update(column_names)

            normalized_specs.append(
                {
                    "function": function,
                    "function_name": getattr(function, "__name__", function.__class__.__name__),
                    "name": name,
                    "timeframe": indicator_timeframe,
                    "buffers": normalized_buffers,
                    "kwargs": kwargs,
                    "column_names": column_names,
                }
            )
        return normalized_specs

    def _apply_timeframe_for_value(self, dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if timeframe == self.timeframe:
            return self._apply_timeframe(dataframe)
        if timeframe == "1m":
            return dataframe.reset_index(drop=True)

        resample_rule = self.TIMEFRAME_ALIASES[timeframe]
        indexed = dataframe.set_index("time")
        resampled = indexed.resample(resample_rule).agg(
            {
                "open": "first",
                "close": "last",
                "high": "max",
                "low": "min",
                "real_volume": "sum",
                "spread": "last",
                "tick_volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open", "close", "high", "low", "spread"])
        return resampled.reset_index(drop=False)

    @staticmethod
    def _normalize_indicator_output(
        indicator_values: pd.Series | pd.DataFrame,
        index: pd.Index,
        column_names: Sequence[str],
        buffers: Sequence[int],
    ) -> pd.DataFrame:
        if isinstance(indicator_values, pd.Series):
            if set(buffers) != {0}:
                raise ValueError("Series indicator outputs only support buffer index 0.")
            return pd.DataFrame({column_names[0]: indicator_values.to_numpy(copy=False)}, index=index)

        if not isinstance(indicator_values, pd.DataFrame):
            raise ValueError("Indicator functions must return a pandas Series or DataFrame.")

        columns = list(indicator_values.columns)
        normalized: dict[str, np.ndarray] = {}
        for buffer_index, column_name in zip(buffers, column_names):
            if buffer_index < 0 or buffer_index >= len(columns):
                raise ValueError(
                    f"Buffer index {buffer_index} is out of range for indicator output with {len(columns)} buffers."
                )
            normalized[column_name] = indicator_values.iloc[:, buffer_index].to_numpy(copy=False)
        return pd.DataFrame(normalized, index=index)

    @staticmethod
    def _align_indicator_to_main_timeframe(
        indicator_timeframe: pd.DataFrame,
        indicator_values: pd.DataFrame,
        main_times: pd.DataFrame,
        *,
        allow_exact_matches: bool,
    ) -> pd.DataFrame:
        source = indicator_timeframe.copy().reset_index(drop=True)
        source = pd.concat([source, indicator_values.reset_index(drop=True)], axis=1)
        aligned = pd.merge_asof(
            main_times.sort_values("time"),
            source.sort_values("time"),
            on="time",
            direction="backward",
            allow_exact_matches=allow_exact_matches,
        )
        return aligned

    def _metadata_indicator_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec["name"],
                "function_name": spec["function_name"],
                "timeframe": spec["timeframe"],
                "buffers": list(spec["buffers"]),
                "kwargs": spec["kwargs"],
                "column_names": list(spec["column_names"]),
            }
            for spec in self.indicator_specs
        ]

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
