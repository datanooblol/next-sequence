import torch.nn as nn
from next_sequence.core import Feature
from typing import Any
from collections import defaultdict

class EmbeddingModel(nn.Module):
    def __init__(self, features:list[Feature], d_model:int, pad_id:int=0):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.embedder = nn.ModuleDict({
            feature.name: self.get_embed_model(feature, pad_id)
            for feature in features
        })

    def get_embed_model(self, feature:Feature, pad_id):
        if (feature.is_linear==False) & (feature.vocab_size > 0):
            return nn.Embedding(feature.vocab_size, self.d_model, padding_idx=pad_id)
        return nn.Linear(1, self.d_model)

    def forward(self, X:dict[str, Any])->dict[str, Any]:
        _X = defaultdict(list)
        for feature in self.features:
            x = self.embedder[feature.name](X[feature.name])
            _X[feature.name] = x
        return _X