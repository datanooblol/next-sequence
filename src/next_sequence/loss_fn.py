"""Per-feature losses and learned uncertainty weighting."""
import torch
import torch.nn as nn

from next_sequence.core import Feature
from next_sequence.tasks import get_task


def predicted_features(features: list[Feature]) -> list[Feature]:
    """Features that have a prediction head (everything except PASSTHROUGH).

    Args:
        features (list[Feature]): Feature configuration.

    Returns:
        list[Feature]: The predicted features, in their original order.
    """
    return [f for f in features if get_task(f).predicted]


def calculate_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    features: list[Feature],
) -> dict[str, torch.Tensor]:
    """Per-feature loss, chosen by ``feature.task``.

    - CLASSIFICATION: cross entropy; REGRESSION / POSITIVE_REGRESSION: MSE; QUANTILE: pinball loss.

    Args:
        pred (dict[str, torch.Tensor]): Model outputs per feature (see ``NextSequenceModel``).
        target (dict[str, torch.Tensor]): Targets per feature, ``(B,)`` each.
        features (list[Feature]): Feature configuration. PASSTHROUGH features are skipped.

    Returns:
        dict[str, torch.Tensor]: One scalar loss per predicted feature.

    Raises:
        KeyError: If a predicted feature is missing from ``pred`` or ``target``.
        ValueError: If targets are invalid for their task (e.g. a class id out of range).
    """
    losses: dict[str, torch.Tensor] = {}
    for feature in predicted_features(features):
        if feature.name not in pred:
            raise KeyError(f"no prediction for feature '{feature.name}'")
        if feature.name not in target:
            raise KeyError(f"no target for feature '{feature.name}'. Did the collate function return y?")
        losses[feature.name] = get_task(feature).loss(pred[feature.name], target[feature.name], feature)
    return losses


class UncertaintyWeightedLoss(nn.Module):
    """Learned task weighting from Kendall, Gal & Cipolla (CVPR 2018).

    With ``s = log(sigma^2)`` learned per task, each task adds ``c * exp(-s) * L + 0.5 * s``, where
    ``c`` is the task's ``weight_scale``: 1 for classification and quantile, 0.5 for regression and
    positive regression.

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
            scale = get_task(feature).weight_scale
            terms.append(scale * torch.exp(-s) * losses[feature.name] + 0.5 * s)
        return torch.stack(terms).sum()

    def weights(self) -> dict[str, float]:
        """Effective weight multiplying each task loss.

        Returns:
            dict[str, float]: Feature name -> ``weight_scale * exp(-s)``.
        """
        return {
            f.name: get_task(f).weight_scale * torch.exp(-self.log_vars[f.name]).item()
            for f in self.features
        }
