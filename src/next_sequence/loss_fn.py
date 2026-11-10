import torch
import torch.nn as nn
from next_sequence.core import Feature, TaskType
        
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, features: list[Feature]):
        super().__init__()
        self.features = features
        self.log_vars = nn.ParameterDict(
            {f.name: nn.Parameter(torch.zeros(())) for f in features}
        )

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = []
        for f in self.features:
            # noise as a start but it's a learnable parameter
            s = self.log_vars[f.name]
            # e^-s, task weight
            precision = torch.exp(-s)
            loss = losses[f.name]
            # regularizer = (1/2) * noise
            # 0.5 here comes from 1/2 acting like a regularizer (from paper)
            regularizer = 0.5 * s
            if f.task == TaskType.CLASSIFICATION:
                error_term = precision * loss
            else:  # regression / MSE
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