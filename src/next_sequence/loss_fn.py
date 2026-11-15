"""Per-feature losses and learned uncertainty weighting."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from next_sequence.core import Feature, TaskType


def predicted_features(features: list[Feature]) -> list[Feature]:
    """Features that have a prediction head (everything except PASSTHROUGH).

    Args:
        features (list[Feature]): Feature configuration.

    Returns:
        list[Feature]: The CLASSIFICATION and REGRESSION features, in their original order.
    """
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

    The parameters ``log_vars`` must be passed to the optimizer, without weight decay
    (``build_optimizer`` does this).

    Args:
        features (list[Feature]): Feature configuration; PASSTHROUGH features are ignored.

    Raises:
        ValueError: If no feature is predicted.
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
        """Constant in front of ``exp(-s) * L``: 1 for classification, 0.5 for regression."""
        if feature.task == TaskType.CLASSIFICATION:
            return 1.0
        if feature.task == TaskType.REGRESSION:
            return 0.5
        raise ValueError(f"No weighting defined for task {feature.task} (feature '{feature.name}')")

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine per-feature losses into one scalar.

        Args:
            losses (dict[str, torch.Tensor]): One scalar loss per predicted feature, e.g. from
                ``calculate_loss``.

        Returns:
            torch.Tensor: The weighted total (a scalar). It can be negative once the model is
            confident, because ``0.5 * s`` becomes negative.
        """
        terms = []
        for feature in self.features:
            s = self.log_vars[feature.name]
            terms.append(self._scale(feature) * torch.exp(-s) * losses[feature.name] + 0.5 * s)
        return torch.stack(terms).sum()

    def weights(self) -> dict[str, float]:
        """Effective weight multiplying each task loss.

        Returns:
            dict[str, float]: Feature name -> ``scale * exp(-s)``.
        """
        return {
            f.name: self._scale(f) * torch.exp(-self.log_vars[f.name]).item()
            for f in self.features
        }
