# done
import torch
import torch.nn as nn
from next_sequence.core import Feature, TaskType


def calculate_loss(y_true, y_pred, features):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    mse = nn.MSELoss()
    losses = {}
    for feature in features:
        if feature.task == TaskType.CLASSIFICATION:
            losses[feature.name] = ce(y_pred[feature.name], y_true[feature.name])
        if feature.task == TaskType.REGRESSION:
            losses[feature.name] = mse(y_pred[feature.name].squeeze(-1), y_true[feature.name])
    return losses
        
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, features: list[Feature]):
        super().__init__()
        self.features = features
        self.log_vars = nn.ParameterDict({
            feature.name: nn.Parameter(torch.zeros(())) 
            for feature in features 
            if feature.task != TaskType.PASSTHROUGH
        })

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = []
        for feature in self.features:
            # noise as a start but it's a learnable parameter
            s = self.log_vars[feature.name]
            # e^-s, task weight
            precision = torch.exp(-s)
            loss = losses[feature.name]
            # regularizer = (1/2) * noise
            # 0.5 here comes from 1/2 acting like a regularizer (from paper)
            regularizer = 0.5 * s
            if feature.task == TaskType.CLASSIFICATION:
                error_term = precision * loss
                terms.append(error_term + regularizer)
            elif feature.task == TaskType.REGRESSION:  # regression / MSE
                error_term = 0.5 * precision * loss
                terms.append(error_term + regularizer)
        return torch.stack(terms).sum()

    def weights(self):
        """Effective loss weights (what multiplies each task loss)."""
        return {
            f.name: (1.0 if f.task == TaskType.CLASSIFICATION else 0.5)
                    * torch.exp(-self.log_vars[f.name]).item()
            for f in self.features
        }        