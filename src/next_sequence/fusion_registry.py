import torch
from next_sequence.fusion import BaseFusion
from typing import Any
from enum import Enum

class FusionStrategy(Enum):
    SUM = 'sum'

class SumFusion(BaseFusion):
    def forward(self, X:dict[str, Any])->Any:
        return torch.stack([
            X[feature.name]
            for feature in self.features
        ]).sum(0)

    def output_dim(self)->int:
        return self.d_model

FUSION_REGISTRY = {
    FusionStrategy.SUM:SumFusion
}