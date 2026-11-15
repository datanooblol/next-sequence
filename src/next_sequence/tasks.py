"""Task definitions: what each ``TaskType`` predicts and how it is trained.

Every task type is one ``Task`` object that defines, in one place:

- ``n_outputs``: the width of the prediction head,
- ``transform``: how raw head outputs become predictions (e.g. softplus to stay positive),
- ``loss``: the training loss against the targets,
- ``weight_scale``: the constant used by ``UncertaintyWeightedLoss``.

The heads, the loss and the task weighting all look tasks up here, so adding a task type means
adding one class to ``TASKS``.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from next_sequence.core import Feature, TaskType


class Task(ABC):
    """Behaviour of one task type. Subclasses implement ``n_outputs`` and ``loss``."""

    predicted: bool = True
    """False for input-only tasks (no head, no loss)."""

    weight_scale: float = 1.0
    """``c`` in ``c * exp(-s) * L + 0.5 * s`` (``UncertaintyWeightedLoss``)."""

    @abstractmethod
    def n_outputs(self, feature: Feature) -> int:
        """Width of the head for ``feature``.

        Args:
            feature (Feature): The feature being predicted.

        Returns:
            int: Number of outputs per sample.
        """

    def transform(self, raw: torch.Tensor, feature: Feature) -> torch.Tensor:
        """Turn raw head outputs ``(B, n_outputs)`` into predictions. Identity by default.

        Args:
            raw (torch.Tensor): Raw head outputs.
            feature (Feature): The feature being predicted.

        Returns:
            torch.Tensor: Predictions with the same shape.
        """
        return raw

    @abstractmethod
    def loss(self, pred: torch.Tensor, target: torch.Tensor, feature: Feature) -> torch.Tensor:
        """Training loss for one feature.

        Args:
            pred (torch.Tensor): Predictions ``(B, n_outputs)`` (after ``transform``).
            target (torch.Tensor): Targets ``(B,)``.
            feature (Feature): The feature being predicted.

        Returns:
            torch.Tensor: A scalar loss.
        """


class ClassificationTask(Task):
    """Cross entropy over ``n_classes`` logits. Targets are class ids ``0 .. n_classes - 1``."""

    weight_scale = 1.0

    def n_outputs(self, feature: Feature) -> int:
        assert feature.n_classes is not None          # set by Feature.__post_init__
        return feature.n_classes

    def loss(self, pred: torch.Tensor, target: torch.Tensor, feature: Feature) -> torch.Tensor:
        target = target.long()
        n_classes = pred.shape[-1]
        if target.numel() and (int(target.min()) < 0 or int(target.max()) >= n_classes):
            raise ValueError(
                f"{feature.name}: class targets must be in 0..{n_classes - 1}, got values from "
                f"{int(target.min())} to {int(target.max())}. Check n_classes / vocab_size and that "
                f"classes start at 0."
            )
        return F.cross_entropy(pred, target)


class RegressionTask(Task):
    """MSE on one output of any sign."""

    weight_scale = 0.5

    def n_outputs(self, feature: Feature) -> int:
        return 1

    def loss(self, pred: torch.Tensor, target: torch.Tensor, feature: Feature) -> torch.Tensor:
        return F.mse_loss(pred.reshape(-1).float(), target.reshape(-1).float())


class PositiveRegressionTask(RegressionTask):
    """MSE on one output that is always > 0 (``softplus``). Targets must be >= 0."""

    def transform(self, raw: torch.Tensor, feature: Feature) -> torch.Tensor:
        return F.softplus(raw)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, feature: Feature) -> torch.Tensor:
        if target.numel() and float(target.min()) < 0:
            raise ValueError(
                f"{feature.name}: positive_regression targets must be >= 0, got {float(target.min()):.4g}. "
                f"Scale them by dividing only (y / std), without subtracting the mean, or use "
                f"task='regression'."
            )
        return super().loss(pred, target, feature)


class QuantileTask(Task):
    """Pinball loss for the quantiles of ``feature.coverage`` plus the median.

    Outputs are sorted from lowest to highest quantile (``feature.quantiles``). They are built as
    the median plus cumulative positive offsets, so the bounds can never cross.
    """

    weight_scale = 1.0

    def n_outputs(self, feature: Feature) -> int:
        return len(feature.quantiles)

    def transform(self, raw: torch.Tensor, feature: Feature) -> torch.Tensor:
        m = len(feature.coverages)                     # index of the median column
        median = raw[..., m:m + 1]
        upper = median + torch.cumsum(F.softplus(raw[..., m + 1:]), dim=-1)
        lower = median - torch.cumsum(F.softplus(raw[..., :m].flip(-1)), dim=-1).flip(-1)
        return torch.cat([lower, median, upper], dim=-1)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, feature: Feature) -> torch.Tensor:
        q = torch.tensor(feature.quantiles, dtype=pred.dtype, device=pred.device)
        error = target.reshape(-1, 1).to(pred.dtype) - pred          # (B, K)
        return torch.maximum(q * error, (q - 1) * error).mean()


class PassthroughTask(Task):
    """Input only: no head and no loss."""

    predicted = False

    def n_outputs(self, feature: Feature) -> int:
        return 0

    def loss(self, pred: torch.Tensor, target: torch.Tensor, feature: Feature) -> torch.Tensor:
        raise ValueError(f"{feature.name}: passthrough features have no loss")


TASKS: dict[TaskType, Task] = {
    TaskType.CLASSIFICATION: ClassificationTask(),
    TaskType.REGRESSION: RegressionTask(),
    TaskType.POSITIVE_REGRESSION: PositiveRegressionTask(),
    TaskType.QUANTILE: QuantileTask(),
    TaskType.PASSTHROUGH: PassthroughTask(),
}
"""Maps every ``TaskType`` to its behaviour."""


def get_task(feature: Feature) -> Task:
    """Look up the task of a feature.

    Args:
        feature (Feature): A feature.

    Returns:
        Task: Its task behaviour.
    """
    return TASKS[TaskType(feature.task)]
