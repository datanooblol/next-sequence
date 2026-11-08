from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from next_sequence.core import Feature
from typing import Any

class BaseFusion(nn.Module, ABC):
    """embs: {name: (B,T,d_model)} -> (B,T,d_model). Order = order of `features`."""
    def __init__(self, features: list[Feature], d_model: int):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.d_out = self.output_dim()

    @abstractmethod
    def output_dim(self)->int: ...

    @abstractmethod
    def forward(self, X:dict[str, Any])->Any: ...
