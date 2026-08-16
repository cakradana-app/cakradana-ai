"""Training, and the decision about whether the result is worth shipping."""

from cakradana.training.dataset import (
    TrainingData,
    TrainingRow,
    build_training_data,
    to_frame,
)
from cakradana.training.pipeline import (
    HUMAN_LABELS,
    SYNTHETIC_LABELS,
    LabelBasis,
    TrainingConfig,
    TrainingResult,
    manifest_json,
    train,
)

__all__ = [
    "HUMAN_LABELS",
    "LabelBasis",
    "SYNTHETIC_LABELS",
    "TrainingConfig",
    "TrainingData",
    "TrainingResult",
    "TrainingRow",
    "build_training_data",
    "manifest_json",
    "to_frame",
    "train",
]
