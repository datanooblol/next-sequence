from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch.utils.data import Dataset


class CoreDataset(Dataset):
    """Minimal map-style dataset over a list of samples (one dict per sequence)."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TaskType(Enum):
    CLASSIFICATION = 'classification'   # predicted, cross entropy, head outputs n_classes logits
    REGRESSION = 'regression'           # predicted, MSE, head outputs 1 value
    PASSTHROUGH = 'passthrough'         # input only, no head, no loss


@dataclass
class Feature:
    """One input column of the sequence, and optionally what to predict from it.

    Args:
        name (str): Key of this feature in each sample and in the batch dicts. Must not contain ".".
        task (TaskType | str): What to predict for this feature.
        vocab_size (int): Number of token ids for a **categorical input** (``is_linear=False``),
            including PAD (and UNK). Usually ``len(bow)``. Ignored for numeric inputs.
        is_linear (bool): True for a **numeric input** (embedded with ``nn.Linear(1, d)``),
            False for a categorical input (embedded with ``nn.Embedding``).
        n_classes (int | None): Number of output classes for a CLASSIFICATION **target**.
            Defaults to ``vocab_size``. Set it explicitly when the target classes differ from the
            input, e.g. a numeric input whose target is a bucket.

    Examples:
        >>> Feature("item", "classification", vocab_size=len(bow))          # predict the next item
        >>> Feature("price", "regression", is_linear=True)                   # predict a number
        >>> Feature("age", "classification", is_linear=True, n_classes=3)   # numeric in, 3 buckets out
        >>> Feature("weekday", "passthrough", vocab_size=8)                  # input only
    """
    name: str
    task: TaskType
    vocab_size: int = 0
    is_linear: bool = False
    n_classes: int | None = None

    def __post_init__(self):
        self.task = TaskType(self.task)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Feature name must be a non-empty string, got {self.name!r}")
        if "." in self.name:
            raise ValueError(f"Feature name must not contain '.', got {self.name!r}")
        if self.vocab_size < 0:
            raise ValueError(f"{self.name}: vocab_size must be >= 0, got {self.vocab_size}")
        if not self.is_linear and self.vocab_size < 2:
            raise ValueError(
                f"{self.name}: a categorical input needs vocab_size >= 2 (including PAD). "
                f"For a numeric input, set is_linear=True."
            )
        if self.task == TaskType.CLASSIFICATION:
            if self.n_classes is None:
                self.n_classes = self.vocab_size
            if self.n_classes < 2:
                raise ValueError(f"{self.name}: classification needs n_classes >= 2, got {self.n_classes}")
        elif self.n_classes is not None:
            raise ValueError(f"{self.name}: n_classes is only used with task=classification")

    def to_dict(self) -> dict[str, Any]:
        return dict(
            name=self.name,
            task=self.task.value,
            vocab_size=self.vocab_size,
            is_linear=self.is_linear,
            n_classes=self.n_classes,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feature":
        return cls(**data)
