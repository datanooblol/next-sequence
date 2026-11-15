import torch.nn as nn
from next_sequence.core import Feature
from typing import Any

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
        if feature.is_linear:
            return nn.Linear(1, self.d_model)
        return nn.Embedding(feature.vocab_size, self.d_model, padding_idx=pad_id)

    def forward(self, X:dict[str, Any])->dict[str, Any]:
        out = {}
        for feature in self.features:
            x = X[feature.name]
            # cast here so a collate function that returns the "wrong" dtype can't crash the layer
            x = x.float() if feature.is_linear else x.long()
            out[feature.name] = self.embedder[feature.name](x)
        return out