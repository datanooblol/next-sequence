# This is done
from next_sequence.core import Feature
from typing import Dict, Callable, Any
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch

class CustomCollate:
    def __init__(
            self, 
            features:list[Feature], 
            feature_mask:str,
            funcs:Dict[str, Callable], 
            inference:bool=False
        ):
        self.features = features
        self.feature_mask = feature_mask
        self.funcs = funcs
        self.inference = inference

    def __call__(self, batch:list[dict[str, Any]])->dict[str, Any]:
        X = defaultdict(list)
        Y = defaultdict(list)
        for feature in self.features:
            _batch = [b[feature.name] for b in batch]
            x, y = self.funcs[feature.name](_batch, self.inference)
            X[feature.name].extend(x)
            Y[feature.name].extend(y)
        lengths = torch.tensor([len(x) for x in X[self.feature_mask]])

        for feature in self.features:
            x = pad_sequence(X[feature.name], batch_first=True, padding_value=0.0, padding_side='right')
            y = torch.tensor(Y[feature.name])
            X[feature.name] = x.unsqueeze(-1) if feature.is_linear else x
            Y[feature.name] = y

        max_len = X[self.feature_mask].size(1)
        pad_mask = torch.arange(max_len)[None, :] >= lengths[:, None]    
        return dict(X=X, Y=Y, mask=pad_mask, lengths=lengths)            