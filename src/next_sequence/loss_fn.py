import torch
import torch.nn as nn
import torch.nn.functional as F

from next_sequence.core import Feature, TaskType


def predicted_features(features: list[Feature]) -> list[Feature]:
    """Features that have a prediction head (everything except PASSTHROUGH)."""
    return [f for f in features if f.task != TaskType.PASSTHROUGH]


def calculate_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    features: list[Feature],
) -> dict[str, torch.Tensor]:
    """Per-feature loss, chosen by ``feature.task``.

    Args:
        pred (dict[str, Tensor]): Model outputs. Classification ``(B, C)`` logits,
            regression ``(B, 1)`` or ``(B,)``.
        target (dict[str, Tensor]): Targets. Classification ``(B,)`` class ids, regression ``(B,)`` values.
        features (list[Feature]): Feature configuration. PASSTHROUGH features are skipped.

    Returns:
        dict[str, Tensor]: One scalar loss per predicted feature.
    """
    losses: dict[str, torch.Tensor] = {}
    for feature in predicted_features(features):
        p, y = pred[feature.name], target[feature.name]
        if feature.task == TaskType.CLASSIFICATION:
            losses[feature.name] = F.cross_entropy(p, y.long())
        elif feature.task == TaskType.REGRESSION:
            losses[feature.name] = F.mse_loss(p.reshape(-1).float(), y.reshape(-1).float())
        else:
            raise ValueError(f"No loss defined for task {feature.task} (feature '{feature.name}')")
    return losses


class UncertaintyWeightedLoss(nn.Module):
    """Learned task weighting from Kendall, Gal & Cipolla (CVPR 2018).

    With ``s = log(sigma^2)`` learned per task:

    - classification: ``exp(-s) * L + 0.5 * s``
    - regression: ``0.5 * exp(-s) * L + 0.5 * s``

    The parameters ``log_vars`` must be passed to the optimizer, without weight decay.
    """

    def __init__(self, features: list[Feature]):
        super().__init__()
        self.features = predicted_features(features)
        if not self.features:
            raise ValueError("UncertaintyWeightedLoss needs at least one non-PASSTHROUGH feature")
        self.log_vars = nn.ParameterDict(
            {f.name: nn.Parameter(torch.zeros(())) for f in self.features}
        )

    @staticmethod
    def _scale(feature: Feature) -> float:
        if feature.task == TaskType.CLASSIFICATION:
            return 1.0
        if feature.task == TaskType.REGRESSION:
            return 0.5
        raise ValueError(f"No weighting defined for task {feature.task} (feature '{feature.name}')")

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = []
        for feature in self.features:
            s = self.log_vars[feature.name]
            terms.append(self._scale(feature) * torch.exp(-s) * losses[feature.name] + 0.5 * s)
        return torch.stack(terms).sum()

    def weights(self) -> dict[str, float]:
        """Effective weight multiplying each task loss."""
        return {
            f.name: self._scale(f) * torch.exp(-self.log_vars[f.name]).item()
            for f in self.features
        }
