# done
import torch
import torch.nn as nn
from next_sequence.core import TaskType, Feature
from typing import Any

class MultiTaskHeads(nn.Module):
    def __init__(self, features:list[Feature], d_model:int):
        super().__init__()
        self.features = features
        self.heads = nn.ModuleDict()
        for feature in features:
            if feature.task == TaskType.CLASSIFICATION:
                # return this one as logits then softmax outside it at prediction
                self.heads[feature.name] = nn.Linear(d_model, feature.vocab_size)
            elif feature.task == TaskType.REGRESSION:
                self.heads[feature.name] = nn.Linear(d_model, 1)
            # if more TaskType, then implement below
            # also, if we need explicit TaskType like: TaskType.DELTA, 
            # that will work with softplus. However, we can do it later

    def forward(self, X:torch.Tensor)->dict[str, Any]:
        # we can be lesser than or equal to a number of features
        return {
            feature.name: self.heads[feature.name](X)
            for feature in self.features
            if feature.task != TaskType.PASSTHROUGH
        }