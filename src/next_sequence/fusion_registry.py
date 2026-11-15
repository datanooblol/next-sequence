from enum import Enum
from typing import Any

import torch

from next_sequence.fusion import BaseFusion


class FusionStrategy(Enum):
    SUM = 'sum'
    CONCAT = 'concat'


class SumFusion(BaseFusion):
    """Element-wise sum of the feature embeddings, then optional projection."""

    def forward(self, X: dict[str, Any]) -> Any:
        x = torch.stack([X[f.name] for f in self.features]).sum(0)
        return self.proj(x)


class ConcatFusion(BaseFusion):
    """Concatenate the feature embeddings, then optional projection."""

    def __init__(self, features, d_model, d_out=None):
        super().__init__(features, len(features) * d_model, d_out)

    def forward(self, X: dict[str, Any]) -> Any:
        x = torch.cat([X[f.name] for f in self.features], dim=-1)
        return self.proj(x)


FUSION_REGISTRY = {
    FusionStrategy.SUM: SumFusion,
    FusionStrategy.CONCAT: ConcatFusion,
}
