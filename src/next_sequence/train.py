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
    """CUDA if available, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    return default_device() if device is None else torch.device(device)


def to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Move tensors inside nested dicts / lists / tuples to ``device``. Other values are kept."""
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
    """
    no_decay_ids = set()
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
    """Linear warmup, then cosine decay to 0. Call ``scheduler.step()`` after every batch."""
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
    return next(model.parameters()).device


def _accumulate(totals: dict[str, float], key: str, value: torch.Tensor, n: int) -> None:
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
    """One pass over ``loader``. Returns the sample-weighted mean of the total and per-feature losses."""
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
    """Mean total and per-feature losses over ``loader``, without updating anything."""
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
