"""next-sequence: multi-task next-event prediction with a transformer encoder (PyTorch)."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("next-sequence")
except PackageNotFoundError:          # running from a source tree without installing
    __version__ = "0.0.0"

from next_sequence.collate_fn import CustomCollate
from next_sequence.core import CoreDataset, Feature, TaskType
from next_sequence.estimator import NextSequence, NotFittedError
from next_sequence.fusion_registry import FusionStrategy
from next_sequence.loss_fn import UncertaintyWeightedLoss, calculate_loss
from next_sequence.model import NextSequenceConfig, NextSequenceModel
from next_sequence.position_embedding import RoPEStrategy
from next_sequence.train import default_device
from next_sequence.utils import create_bow

__all__ = [
    "__version__",
    "CoreDataset",
    "CustomCollate",
    "Feature",
    "FusionStrategy",
    "NextSequence",
    "NextSequenceConfig",
    "NextSequenceModel",
    "NotFittedError",
    "RoPEStrategy",
    "TaskType",
    "UncertaintyWeightedLoss",
    "calculate_loss",
    "create_bow",
    "default_device",
]
