"""Public location for training-data generation.

The implementation remains import-compatible with older ``pipeline`` users
while this module becomes the package-level API.
"""

from fg_core.pipeline.generate_train_data import GenerateTrainData

__all__ = ["GenerateTrainData"]