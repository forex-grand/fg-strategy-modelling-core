from typing import List, Optional, Literal
from pydantic import BaseModel
import tensorflow as tf
from keras import callbacks
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


class TimeBasedTarget(BaseModel):
    type: str = "time_based_stop"
    stop_minutes: int = 60
    mode: Literal["points","raw_difference"] = "points"

class PointsBasedTarget(BaseModel):
    type: str = "points_based_target"
    stop_points: int = 100

TARGET_MODEL_TYPES = TimeBasedTarget | PointsBasedTarget

class SymbolProperties(BaseModel):
    symbol: str
    source: str
    group: str
    contract_size: int
    point_size: float
    digits:   int
    data_start: Optional[str]
    data_end: Optional[str]

class ModelBuildTrainArguments(BaseModel):
    learning_rate: float = 1e-3
    epochs: int = 50
    callbacks: List[dict]
    steps_per_epoch: int

class SymbolIn(BaseModel):
    symbol: str
    group: str

@dataclass
class TrainingResult:
    """Container for per-run training status and output path."""

    symbol: str
    model_type: str
    benchmark_passed: bool
    evaluator_passed: bool
    metrics: dict[str, float]
    model: tf.keras.Model
    model_id: Optional[str] = None 

class EpochMetricsLogger(callbacks.Callback):
    """Logs key metrics after each epoch."""

    def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
        payload = logs or {}
        LOGGER.info("Epoch %d metrics: %s", epoch + 1, payload)
