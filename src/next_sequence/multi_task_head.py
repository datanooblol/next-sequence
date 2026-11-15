"""One output head per predicted feature."""
import torch
import torch.nn as nn

from next_sequence.core import Feature, TaskType


class MultiTaskHeads(nn.Module):
    """A linear head for every non-PASSTHROUGH feature.

    - CLASSIFICATION: ``nn.Linear(d_model, n_classes)``, returning raw logits (no softmax).
    - REGRESSION: ``nn.Linear(d_model, 1)``.
    - PASSTHROUGH: no head.

    Args:
        features (list[Feature]): Feature configuration.
        d_model (int): Width of the sequence representation fed to the heads.
    """

    def __init__(self, features: list[Feature], d_model: int):
        super().__init__()
        self.features = [f for f in features if f.task != TaskType.PASSTHROUGH]
        self.heads = nn.ModuleDict()
        for feature in self.features:
            if feature.task == TaskType.CLASSIFICATION:
                assert feature.n_classes is not None      # set by Feature.__post_init__
                self.heads[feature.name] = nn.Linear(d_model, feature.n_classes)
            elif feature.task == TaskType.REGRESSION:
                self.heads[feature.name] = nn.Linear(d_model, 1)
            else:
                raise ValueError(f"No head defined for task {feature.task} (feature '{feature.name}')")

    def forward(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply every head to the sequence representation.

        Args:
            X (torch.Tensor): ``(B, d_model)``.

        Returns:
            dict[str, torch.Tensor]: Classification ``(B, n_classes)`` logits, regression ``(B, 1)``.
        """
        return {feature.name: self.heads[feature.name](X) for feature in self.features}
