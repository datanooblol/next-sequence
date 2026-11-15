from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from next_sequence.core import Feature


class BaseFusion(nn.Module, ABC):
    """Combine per-feature embeddings ``{name: (B, T, d)}`` into one ``(B, T, d_out)`` tensor.

    Args:
        features (list[Feature]): Features in a fixed order.
        d_model (int): Width of the combined tensor before projection.
        d_out (int | None): Output width. Defaults to ``d_model`` (no projection).
    """

    def __init__(self, features: list[Feature], d_model: int, d_out: int | None = None):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.d_out = d_out or d_model
        # only a real layer when the width changes, otherwise no unused parameters
        self.proj = nn.Linear(d_model, self.d_out) if d_model != self.d_out else nn.Identity()

    @abstractmethod
    def forward(self, X: dict[str, Any]) -> Any: ...
