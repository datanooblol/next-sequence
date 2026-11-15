"""Training building blocks: device handling, optimizer/scheduler factories, epoch loops.

These are plain functions with no hidden state. ``NextSequence`` uses them, and they can also be
used directly to write a custom training loop.
"""
import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn

from next_sequence.loss_fn import UncertaintyWeightedLoss, calculate_loss


# ---------- device ----------
def default_device() -> torch.device:
    """Pick the best available device.

    Returns:
        torch.device: ``cuda`` if available, then Apple ``mps``, then ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Turn a user-supplied device into a ``torch.device``.

    Args:
        device (str | torch.device | None): ``None`` means ``default_device()``.

    Returns:
        torch.device: The device to use.
    """
    return default_device() if device is None else torch.device(device)


def to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Move tensors inside nested dicts / lists / tuples to ``device``. Other values are kept.

    Args:
        obj (Any): A tensor, or a dict / list / tuple that contains tensors (e.g. a batch).
        device (torch.device): Target device.
        non_blocking (bool): Passed to ``Tensor.to``; overlaps copies with compute on CUDA.

    Returns:
        Any: The same structure with every tensor on ``device``.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device, non_blocking) for v in obj)
    return obj


# ---------- factories ----------
def build_optimizer(
    model: nn.Module,
    criterion: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """AdamW with weight decay only on weight matrices.

    No decay on biases, LayerNorm, embeddings, the CLS token, or the criterion's ``log_vars``
    (decay would pull every task weight toward 1). The criterion's parameters are included so the
    learned task weights actually train.

    Args:
        model (nn.Module): The network to train.
        criterion (nn.Module): The loss module (its parameters are trained too).
        lr (float): Learning rate.
        weight_decay (float): Weight decay for weight matrices.

    Returns:
        torch.optim.Optimizer: An ``AdamW`` optimizer with separate parameter groups.
    """
    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or id(p) in no_decay_ids or name.endswith("cls"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": [p for p in criterion.parameters() if p.requires_grad], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to 0. Call ``scheduler.step()`` after every batch.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.
        total_steps (int): Total number of optimizer steps (``epochs * batches_per_epoch``).
        warmup_ratio (float): Fraction of ``total_steps`` used for linear warmup.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The scheduler.
    """
    total_steps = max(1, total_steps)
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------- epoch loops ----------
def _model_device(model: nn.Module) -> torch.device:
    """Device of the model's first parameter."""
    return next(model.parameters()).device


def _accumulate(totals: dict[str, float], key: str, value: torch.Tensor, n: int) -> None:
    """Add ``value * n`` to ``totals[key]`` for a sample-weighted mean."""
    totals[key] = totals.get(key, 0.0) + value.detach().item() * n


def train_one_epoch(
    model: nn.Module,
    criterion: UncertaintyWeightedLoss,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """Train for one pass over ``loader``.

    Args:
        model (nn.Module): The network; set to train mode.
        criterion (UncertaintyWeightedLoss): Combines the per-feature losses.
        loader (Iterable[dict[str, Any]]): Training batches from ``CustomCollate(inference=False)``.
        optimizer (torch.optim.Optimizer): Usually from ``build_optimizer``.
        scheduler (LRScheduler | None): Stepped after every batch, if given.
        device (torch.device | None): ``None`` uses the model's device.
        grad_clip (float | None): Max gradient norm of the model parameters; ``None`` disables it.

    Returns:
        dict[str, float]: Sample-weighted mean of ``"loss"`` (total) and of each feature's loss.

    Raises:
        ValueError: If a batch has no targets.
        FloatingPointError: If the loss becomes NaN or infinite.
    """
    device = device or _model_device(model)
    model.train()
    criterion.train()
    totals: dict[str, float] = {}
    n_seen = 0
    for batch in loader:
        if batch["Y"] is None:
            raise ValueError("training batch has Y=None. Build the train loader with inference=False.")
        batch = to_device(batch, device)
        outputs = model(batch["X"], batch["mask"])
        losses = calculate_loss(outputs, batch["Y"], criterion.features)
        loss = criterion(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"loss became {loss.item()}; per-feature: "
                f"{ {k: v.item() for k, v in losses.items()} }. Check input/target scaling."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = batch["mask"].size(0)
        n_seen += n
        _accumulate(totals, "loss", loss, n)
        for name, value in losses.items():
            _accumulate(totals, name, value, n)
    return {k: v / max(n_seen, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: UncertaintyWeightedLoss,
    loader: Iterable[dict[str, Any]],
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute losses over ``loader`` without updating anything.

    Args:
        model (nn.Module): The network; set to eval mode.
        criterion (UncertaintyWeightedLoss): Combines the per-feature losses.
        loader (Iterable[dict[str, Any]]): Batches from ``CustomCollate(inference=False)``.
        device (torch.device | None): ``None`` uses the model's device.

    Returns:
        dict[str, float]: Sample-weighted mean of ``"loss"`` (total) and of each feature's loss.

    Raises:
        ValueError: If a batch has no targets.
    """
    device = device or _model_device(model)
    model.eval()
    criterion.eval()
    totals: dict[str, float] = {}
    n_seen = 0
    for batch in loader:
        if batch["Y"] is None:
            raise ValueError("evaluation batch has Y=None. Build the loader with inference=False.")
        batch = to_device(batch, device)
        outputs = model(batch["X"], batch["mask"])
        losses = calculate_loss(outputs, batch["Y"], criterion.features)
        n = batch["mask"].size(0)
        n_seen += n
        _accumulate(totals, "loss", criterion(losses), n)
        for name, value in losses.items():
            _accumulate(totals, name, value, n)
    return {k: v / max(n_seen, 1) for k, v in totals.items()}
