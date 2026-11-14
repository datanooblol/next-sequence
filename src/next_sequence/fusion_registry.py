import torch
import torch.nn as nn
from next_sequence.fusion import BaseFusion
from typing import Any
from enum import Enum

class FusionStrategy(Enum):
    SUM = 'sum'
    CONCAT = 'concat'

class SumFusion(BaseFusion):

    def forward(self, X:dict[str, Any])->Any:
        _x = torch.stack([
            X[feature.name]
            for feature in self.features
        ]).sum(0)
        if self.d_model == self.d_out:
            return _x
        return self.proj(_x)

class ConcatFusion(BaseFusion):
    def __init__(self, features, d_model, d_out=None):
        super().__init__(features, len(features)*d_model, d_out)

    def forward(self, X:dict[str, Any])->Any:
        _x = torch.cat([
            X[feature.name]
            for feature in self.features
        ], dim=-1)
        if self.d_model == self.d_out:
            return _x
        return self.proj(_x)

FUSION_REGISTRY = {
    FusionStrategy.SUM:SumFusion,
    FusionStrategy.CONCAT:ConcatFusion
}