"""One output head per predicted feature."""
import torch
import torch.nn as nn

from next_sequence.core import Feature
from next_sequence.tasks import get_task


class MultiTaskHeads(nn.Module):
    """A linear head for every predicted feature, sized and post-processed by its task.

    - CLASSIFICATION: ``n_classes`` raw logits (no softmax).
    - REGRESSION: one value.
    - POSITIVE_REGRESSION: one value passed through softplus (always > 0).
    - QUANTILE: one column per quantile in ``feature.quantiles``, never crossing.
    - PASSTHROUGH: no head.

    Args:
        features (list[Feature]): Feature configuration.
        d_model (int): Width of the sequence representation fed to the heads.
    """

    def __init__(self, features: list[Feature], d_model: int):
        super().__init__()
        self.features = [f for f in features if get_task(f).predicted]
        self.heads = nn.ModuleDict({
            f.name: nn.Linear(d_model, get_task(f).n_outputs(f)) for f in self.features
        })

    def forward(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply every head to the sequence representation.

        Args:
            X (torch.Tensor): ``(B, d_model)``.

        Returns:
            dict[str, torch.Tensor]: ``(B, n_outputs)`` predictions per predicted feature.
        """
        return {f.name: get_task(f).transform(self.heads[f.name](X), f) for f in self.features}
