"""Built-in fusion strategies and the registry used by ``NextSequenceConfig.fusion_strategy``."""
from enum import Enum

import torch

from next_sequence.core import Feature
from next_sequence.fusion import BaseFusion


class FusionStrategy(Enum):
    """How per-feature embeddings are combined.

    Attributes:
        SUM: Element-wise sum. Output width before projection is ``emb_dim``.
        CONCAT: Concatenation. Output width before projection is ``n_features * emb_dim``.
    """
    SUM = 'sum'
    CONCAT = 'concat'


class SumFusion(BaseFusion):
    """Element-wise sum of the feature embeddings, then an optional projection to ``d_out``.

    Args:
        features (list[Feature]): Features in a fixed order.
        d_model (int): Width of each feature embedding.
        d_out (int | None): Output width. ``None`` keeps ``d_model``.
    """

    def forward(self, X: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum the embeddings.

        Args:
            X (dict[str, torch.Tensor]): One ``(B, T, d_model)`` embedding per feature.

        Returns:
            torch.Tensor: ``(B, T, d_out)``.
        """
        x = torch.stack([X[f.name] for f in self.features]).sum(0)
        return self.proj(x)


class ConcatFusion(BaseFusion):
    """Concatenate the feature embeddings, then an optional projection to ``d_out``.

    Args:
        features (list[Feature]): Features in a fixed order.
        d_model (int): Width of each feature embedding. The concatenation is
            ``len(features) * d_model`` wide.
        d_out (int | None): Output width. ``None`` keeps the concatenated width.
    """

    def __init__(self, features: list[Feature], d_model: int, d_out: int | None = None):
        super().__init__(features, len(features) * d_model, d_out)

    def forward(self, X: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate the embeddings along the last dimension.

        Args:
            X (dict[str, torch.Tensor]): One ``(B, T, d_model)`` embedding per feature.

        Returns:
            torch.Tensor: ``(B, T, d_out)``.
        """
        x = torch.cat([X[f.name] for f in self.features], dim=-1)
        return self.proj(x)


FUSION_REGISTRY: dict[FusionStrategy, type[BaseFusion]] = {
    FusionStrategy.SUM: SumFusion,
    FusionStrategy.CONCAT: ConcatFusion,
}
"""Maps each ``FusionStrategy`` to the class that implements it."""
