from abc import abstractmethod
import json
from pathlib import Path
from typing import Any, List, Optional, Union

import keras
import tensorflow as tf

from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.settings import Settings
from forexgrand_core.schemas import ModelBuildTrainArguments

class BaseModel:
    def __init__(self, sequence_length:int, preprocessor: PreprocessBase):
        self.preprocessor = preprocessor.get_transform_layer()
        self.model:tf.keras.Model = None
        self.xgb_model: Any = None
        self.settings = Settings()
        self.data_directory = Path(self.settings.data_directory).expanduser().resolve()
        self.history: keras.callbacks.History = None
        self.sequence_length = sequence_length
        self.feature_transformer = None

    @abstractmethod
    def build_train_model(self, train_ds, eval_ds, fn_args:ModelBuildTrainArguments)->tf.keras.Model:
        pass
    
    def generate_model_path(
        self,
        sequence_length: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Path:
        resolved_sequence_length = self._resolve_sequence_length(sequence_length)
        resolved_model_name = (model_name or self.__class__.__name__).strip().lower()
        model_root = (
            self.data_directory
            / "models"
            / (self.settings.data_source or "unknown").strip().lower()
            / resolved_model_name
            / str(resolved_sequence_length)
        )
        version = self._resolve_next_version(model_root)
        export_path = model_root / str(version)
        export_path.mkdir(parents=True, exist_ok=True)
        return export_path

    def get_serving_signature(self):
        input_signature = {
            "time": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.int64, name="time"),
            "open": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="open"),
            "high": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="high"),
            "close": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="close"),
            "low": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="low"),
            "spread": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="spread"),
            "real_volume": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="real_volume"),
            "tick_volume": tf.TensorSpec(shape=[None, self.sequence_length], dtype=tf.float32, name="tick_volume"),
        }

        @tf.function(input_signature=[input_signature])
        def serve(examples):
            if isinstance(self.model.input, list):
                ordered_inputs = [examples[name] for name in ["time", "open", "high", "close", "low", "spread", "real_volume", "tick_volume"]]
                return {"output": self.model(ordered_inputs)}
            return {"output": self.model(examples)}

        return serve

    def _build_input_signature(self, sequence_length: Optional[int] = None) -> List[keras.Input]:
        """
            Creates the Input signature for inference.
        """
        resolved_sequence_length = self.sequence_length if sequence_length is None else sequence_length
        inputs_list = []
        inputs_list.append(keras.Input(shape=(resolved_sequence_length,), name="time", dtype=tf.int64))
        float_fields = ['open','close','high','low','spread','real_volume','tick_volume']
        for field in float_fields:
            inputs_list.append(keras.Input((resolved_sequence_length,), name=field, dtype=tf.float32))
        
        return inputs_list

    def _resolve_sequence_length(self, sequence_length: Optional[int] = None) -> int:
        resolved_sequence_length = sequence_length
        if resolved_sequence_length is None:
            resolved_sequence_length = getattr(self, "sequence_length", None)
        if resolved_sequence_length is None:
            resolved_sequence_length = getattr(self.preprocessor, "sequence_length", None)
        if resolved_sequence_length is None:
            raise ValueError("sequence_length is required to generate a model save path.")
        return int(resolved_sequence_length)

    @staticmethod
    def _resolve_next_version(model_root: Path) -> int:
        version_numbers: list[int] = []
        if model_root.exists():
            for child in model_root.iterdir():
                if child.is_dir() and child.name.isdigit():
                    version_numbers.append(int(child.name))
        return (max(version_numbers) + 1) if version_numbers else 1

    @staticmethod
    def _write_metadata(metadata_path: Path, metadata: dict[str, object]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle, indent=2)
    
    
