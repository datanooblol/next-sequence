import torch.nn as nn
from next_sequence.core import TaskType

class MultiTaskHeads(nn.Module):
    def __init__(self, features, d_model):
        super().__init__()
        self.heads = nn.ModuleDict()
        for f in features:
            if f.task == TaskType.CLASSIFICATION:
                # return this one as logits then softmax outside it at prediction
                self.heads[f.name] = nn.Linear(d_model, f.vocab_size)
            elif f.task == TaskType.REGRESSION:
                self.heads[f.name] = nn.Linear(d_model, 1)
            # if more TaskType, then implement below
            # also, if we need explicit TaskType like: TaskType.DELTA, 
            # that will work with softplus. However, we can do it later

    def forward(self, X):
        return {n: h(X) for n, h in self.heads.items()}