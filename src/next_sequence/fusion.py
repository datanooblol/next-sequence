"""Base class for combining per-feature embeddings into one sequence representation."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from next_sequence.core import Feature


class BaseFusion(nn.Module, ABC):
    """Combine per-feature embeddings ``{name: (B, T, d)}`` into one ``(B, T, d_out)`` tensor.

    Subclasses implement ``forward`` and call ``self.proj`` on the combined tensor. ``proj`` is a
    real ``nn.Linear`` only when the width changes, so no unused parameters are created.

    Args:
        features (list[Feature]): Features in a fixed order.
        d_model (int): Width of the combined tensor before projection.
        d_out (int | None): Output width. ``None`` keeps ``d_model`` (no projection).

    Attributes:
        d_out (int): Width of the fusion output; the rest of the model is built with it.
    """

    def __init__(self, features: list[Feature], d_model: int, d_out: int | None = None):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.d_out: int = d_out or d_model
        self.proj: nn.Module = (
            nn.Linear(d_model, self.d_out) if d_model != self.d_out else nn.Identity()
        )

    @abstractmethod
    def forward(self, X: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine the embeddings.

        Args:
            X (dict[str, torch.Tensor]): One ``(B, T, d)`` embedding per feature.

        Returns:
            torch.Tensor: Combined tensor of shape ``(B, T, d_out)``.
        """
