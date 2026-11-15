"""Ready-made target strategies for ``make_collate(y=...)``.

A target strategy turns a training history into the value to predict, and knows how to turn a
prediction back into real units (``decode``). Using one object for both keeps them consistent:

>>> delta = targets.Delta(scale=targets.target_std(train_prices, targets.Delta()))
>>> funcs = {"price": make_collate("float", y=delta)}
>>> predicted_price = delta.decode(out["price"], last=last_known_price)

``history[-1]`` is the event to predict and ``history[-2]`` the last one the model sees. Every
strategy takes ``value``: a function that picks the number from one step, for samples that store
several values per step, e.g. ``value=lambda step: step[1]`` for ``(scaled, raw)`` steps.

Any other target can still be a plain function: ``make_collate(y=lambda history: ...)``.
"""
import bisect
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

ValueFn = Callable[[Any], float]


def _identity(step: Any) -> Any:
    """Default ``value``: the step itself."""
    return step


class Target:
    """Base class: ``target(history)`` gives the target; ``decode`` maps predictions back.

    Args:
        value (Callable[[Any], float] | None): Picks the number from one step. Default: the step.
    """

    #: Task the target is meant for, for documentation and checks.
    task: str = "regression"

    def __init__(self, value: ValueFn | None = None):
        self.value: ValueFn = value if value is not None else _identity

    def __call__(self, history: Sequence[Any]) -> Any:
        """Compute the target from a training history (last step = event to predict)."""
        raise NotImplementedError

    def decode(self, pred: torch.Tensor, last: torch.Tensor | Sequence[float] | None = None) -> torch.Tensor:
        """Turn predictions back into real units.

        Args:
            pred (torch.Tensor): Model output for this feature (``(N,)``, ``(N, 1)`` or quantile
                columns ``(N, K)``; every column is decoded).
            last (torch.Tensor | Sequence[float] | None): The last value the model saw for each
                sample, in real units. Needed by change-based targets.

        Returns:
            torch.Tensor: Decoded values with the shape of ``pred`` (a trailing 1 is dropped).
        """
        raise NotImplementedError

    def _two(self, history: Sequence[Any]) -> tuple[float, float]:
        """The last seen value and the value to predict, as floats."""
        if len(history) < 2:
            raise IndexError(f"{type(self).__name__} needs at least 2 steps, got {len(history)}")
        return float(self.value(history[-2])), float(self.value(history[-1]))

    @staticmethod
    def _shape(pred: torch.Tensor) -> torch.Tensor:
        """Predictions as float, with a trailing dimension of 1 dropped."""
        pred = torch.as_tensor(pred, dtype=torch.float32)
        return pred.squeeze(-1) if pred.dim() == 2 and pred.shape[-1] == 1 else pred

    @staticmethod
    def _last(last: torch.Tensor | Sequence[float] | None, pred: torch.Tensor, name: str) -> torch.Tensor:
        """Validate ``last`` and shape it to broadcast against ``pred``."""
        if last is None:
            raise ValueError(f"{name}.decode needs last=: the last value the model saw for each sample")
        t = torch.as_tensor(last, dtype=torch.float32)
        if t.shape[0] != pred.shape[0]:
            raise ValueError(f"last has {t.shape[0]} values but there are {pred.shape[0]} predictions")
        return t.unsqueeze(-1) if pred.dim() == 2 else t

    def __repr__(self) -> str:
        fields = {k: v for k, v in vars(self).items() if k != "value"}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


class NextValue(Target):
    """The next value itself, standardized: ``(next - shift) / scale``.

    Use with ``task="regression"`` (or ``"quantile"``). With ``task="positive_regression"`` keep
    ``shift=0`` so targets stay >= 0.

    Args:
        scale (float): Divide by this (e.g. the training std). Must be > 0.
        shift (float): Subtract this first (e.g. the training mean).
        value (Callable[[Any], float] | None): Picks the number from one step.
    """

    def __init__(self, scale: float = 1.0, shift: float = 0.0, value: ValueFn | None = None):
        super().__init__(value)
        _check_scale(scale)
        self.scale, self.shift = float(scale), float(shift)

    def __call__(self, history: Sequence[Any]) -> float:
        return (float(self.value(history[-1])) - self.shift) / self.scale

    def decode(self, pred: torch.Tensor, last: torch.Tensor | Sequence[float] | None = None) -> torch.Tensor:
        return self._shape(pred) * self.scale + self.shift


class Delta(Target):
    """The change since the last step: ``(next - last) / scale``. Decodes to ``last + pred * scale``.

    Can go negative, and so can a decoded value; for always-positive quantities prefer ``LogRatio``.

    Args:
        scale (float): Divide by this, e.g. ``target_std(train_histories, Delta())``. Must be > 0.
        value (Callable[[Any], float] | None): Picks the number from one step.
    """

    def __init__(self, scale: float = 1.0, value: ValueFn | None = None):
        super().__init__(value)
        _check_scale(scale)
        self.scale = float(scale)

    def __call__(self, history: Sequence[Any]) -> float:
        prev, nxt = self._two(history)
        return (nxt - prev) / self.scale

    def decode(self, pred: torch.Tensor, last: torch.Tensor | Sequence[float] | None = None) -> torch.Tensor:
        pred = self._shape(pred)
        return self._last(last, pred, "Delta") + pred * self.scale


class LogRatio(Target):
    """The relative change: ``log(next / last) / scale``. Decodes to ``last * exp(pred * scale)``.

    A decoded value always has the sign of ``last``, so positive quantities stay positive.
    Values must be > 0.

    Args:
        scale (float): Divide by this, e.g. ``target_std(train_histories, LogRatio())``. Must be > 0.
        value (Callable[[Any], float] | None): Picks the number from one step.
    """

    def __init__(self, scale: float = 1.0, value: ValueFn | None = None):
        super().__init__(value)
        _check_scale(scale)
        self.scale = float(scale)

    def __call__(self, history: Sequence[Any]) -> float:
        prev, nxt = self._two(history)
        if prev <= 0 or nxt <= 0:
            raise ValueError(f"LogRatio needs values > 0, got {prev} -> {nxt}. Use Delta for values "
                             f"that can be zero or negative.")
        return math.log(nxt / prev) / self.scale

    def decode(self, pred: torch.Tensor, last: torch.Tensor | Sequence[float] | None = None) -> torch.Tensor:
        pred = self._shape(pred)
        return self._last(last, pred, "LogRatio") * torch.exp(pred * self.scale)


class Bucket(Target):
    """The class of the next value, split by ``edges``. Use with ``task="classification"``.

    With ``edges=[a, b]``: class 0 is ``<= a``, class 1 is ``(a, b]``, class 2 is ``> b``. Set the
    feature's ``n_classes`` to ``target.n_classes`` (``len(edges) + 1``).

    Args:
        edges (Sequence[float]): Strictly increasing thresholds.
        labels (Sequence[str] | None): Optional names for the classes, used by ``decode``.
        value (Callable[[Any], float] | None): Picks the number from one step.
    """

    task = "classification"

    def __init__(self, edges: Sequence[float], labels: Sequence[str] | None = None,
                 value: ValueFn | None = None):
        super().__init__(value)
        self.edges = _check_edges(edges)
        self.labels = _check_labels(labels, len(self.edges) + 1)

    @property
    def n_classes(self) -> int:
        """Number of classes: ``len(edges) + 1``."""
        return len(self.edges) + 1

    def _classify(self, x: float) -> int:
        """Class index of ``x`` given the edges."""
        return bisect.bisect_left(self.edges, x)

    def __call__(self, history: Sequence[Any]) -> int:
        return self._classify(float(self.value(history[-1])))

    def decode(self, pred: torch.Tensor, last: torch.Tensor | Sequence[float] | None = None) -> Any:
        """Predicted class per sample: ids, or label names when ``labels`` were given.

        Args:
            pred (torch.Tensor): Logits ``(N, n_classes)``.
            last: Unused.

        Returns:
            torch.Tensor | list[str]: Class ids ``(N,)``, or their labels.
        """
        ids = torch.as_tensor(pred).argmax(-1)
        return [self.labels[i] for i in ids.tolist()] if self.labels else ids


class DeltaBucket(Bucket):
    """The class of the change since the last step, split by ``edges``.

    Example: ``DeltaBucket([-30, 30], labels=["down", "flat", "up"])``.

    Args:
        edges (Sequence[float]): Strictly increasing thresholds on ``next - last``.
        labels (Sequence[str] | None): Optional names for the classes.
        value (Callable[[Any], float] | None): Picks the number from one step.
    """

    def __call__(self, history: Sequence[Any]) -> int:
        prev, nxt = self._two(history)
        return self._classify(nxt - prev)


def target_std(histories: Iterable[Sequence[Any]], target: Target) -> float:
    """Standard deviation of ``target`` over every step of the training histories.

    Use it to set ``scale`` from the **training split**. Build ``target`` with ``scale=1`` (the
    default) and the same ``value`` you will use later.

    Args:
        histories (Iterable[Sequence[Any]]): Training histories of one feature,
            e.g. ``[u["price"] for u in train]``.
        target (Target): A target with ``scale=1``, e.g. ``Delta()``.

    Returns:
        float: The standard deviation (1.0 if there is nothing to measure or it is 0).

    Example:
        >>> scale = target_std([u["price"] for u in train_raw], Delta())
    """
    values: list[float] = []
    minimum = 1 if isinstance(target, NextValue) or type(target) is Bucket else 2
    for history in histories:
        for end in range(minimum, len(history) + 1):
            values.append(float(target(history[:end])))
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return std if std > 0 else 1.0


def _check_scale(scale: float) -> None:
    """Raise unless ``scale`` is a finite number > 0."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive number, got {scale!r}")


def _check_edges(edges: Sequence[float]) -> list[float]:
    """Validate bucket edges and return them as floats."""
    values = [float(e) for e in edges]
    if not values:
        raise ValueError("edges needs at least one threshold, e.g. [0.0]")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"edges must be strictly increasing, got {list(edges)}")
    return values


def _check_labels(labels: Sequence[str] | None, n: int) -> list[str] | None:
    """Validate that there is one label per class."""
    if labels is None:
        return None
    if len(labels) != n:
        raise ValueError(f"{n} classes need {n} labels, got {len(labels)}")
    return list(labels)
