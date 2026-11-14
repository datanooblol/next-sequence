# done
from abc import ABC, abstractmethod
import torch.nn as nn
from next_sequence.core import Feature
from typing import Any

class BaseFusion(nn.Module, ABC):
    """embs: {name: (B,T,d_model)} -> (B,T,d_model). Order = order of `features`."""
    def __init__(self, features: list[Feature], d_model: int, d_out: int|None=None):
        super().__init__()
        # self._target = d_out or d_model
        self.d_out = d_out or d_model
        self.features = features
        self.d_model = d_model
        # proj will run only if d_model != d_out only
        self.proj = nn.Linear(d_model, self.d_out)

    @abstractmethod
    def forward(self, X:dict[str, Any])->Any: ...
