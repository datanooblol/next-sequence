"""High-level wrapper: ``NextSequence(config).fit(train_loader, val_loader).predict(test_loader)``."""
import copy
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

from next_sequence.core import TaskType
from next_sequence.loss_fn import UncertaintyWeightedLoss
from next_sequence.model import NextSequenceConfig, NextSequenceModel
from next_sequence.train import (
    build_optimizer, build_scheduler, evaluate, resolve_device, to_device, train_one_epoch,
)

CHECKPOINT_FORMAT = "next-sequence"
CHECKPOINT_VERSION = 1


class NotFittedError(RuntimeError):
    """Raised when ``predict`` / ``save`` is called before ``fit`` or ``load``."""


class NextSequence:
    """Train and use a ``NextSequenceModel``.

    Data preparation (features, vocabularies, scaling, collate functions, DataLoaders) is done by
    the user. This class owns the model, the loss, the optimizer and the training loop.

    Args:
        config (NextSequenceConfig): Model architecture and features.
        epochs (int): Number of passes over the training loader.
        lr (float): Peak learning rate (AdamW).
        weight_decay (float): Weight decay for weight matrices only.
        warmup_ratio (float): Fraction of steps used for linear warmup before cosine decay.
        grad_clip (float | None): Max gradient norm. ``None`` disables clipping.
        device (str | torch.device | None): ``None`` picks CUDA, then MPS, then CPU.
        seed (int | None): If set, makes weight initialization and training reproducible.
        verbose (bool): Print one line of metrics per epoch.
        warm_start (bool): If True, a second ``fit`` continues from the current weights
            instead of starting from a fresh model.

    Attributes (after ``fit`` or ``load``):
        model_ (NextSequenceModel), criterion_ (UncertaintyWeightedLoss),
        history_ (list[dict]): per-epoch metrics, best_epoch_ (int | None).
    """

    def __init__(
        self,
        config: NextSequenceConfig,
        epochs: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        grad_clip: float | None = 1.0,
        device: str | torch.device | None = None,
        seed: int | None = None,
        verbose: bool = True,
        warm_start: bool = False,
    ):
        if not isinstance(config, NextSequenceConfig):
            raise TypeError(f"config must be a NextSequenceConfig, got {type(config).__name__}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
        self.config = config
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip
        self.device = resolve_device(device)
        self.seed = seed
        self.verbose = verbose
        self.warm_start = warm_start

    # ---------- helpers ----------
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "model_")

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("This NextSequence is not fitted yet. Call fit() or load() first.")

    def _build(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.model_ = NextSequenceModel(self.config).to(self.device)
        self.criterion_ = UncertaintyWeightedLoss(self.config.features).to(self.device)
        self.history_: list[dict[str, Any]] = []
        self.best_epoch_: int | None = None

    def train_settings(self) -> dict[str, Any]:
        return dict(epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
                    warmup_ratio=self.warmup_ratio, grad_clip=self.grad_clip, seed=self.seed)

    def task_weights(self) -> dict[str, float]:
        """Learned effective weight of each task loss."""
        self._check_fitted()
        return self.criterion_.weights()

    # ---------- fit ----------
    def fit(self, train_loader: Iterable[dict], val_loader: Iterable[dict] | None = None) -> "NextSequence":
        """Train on ``train_loader``. With ``val_loader``, the weights of the best epoch
        (lowest validation loss) are restored at the end."""
        if not (self.warm_start and self.is_fitted):
            self._build()
        elif self.seed is not None:
            torch.manual_seed(self.seed)

        try:
            steps_per_epoch = len(train_loader)          # type: ignore[arg-type]
        except TypeError:
            raise TypeError("train_loader must have a length (e.g. a DataLoader over a map-style dataset)")
        if steps_per_epoch == 0:
            raise ValueError("train_loader is empty")

        optimizer = build_optimizer(self.model_, self.criterion_, self.lr, self.weight_decay)
        scheduler = build_scheduler(optimizer, self.epochs * steps_per_epoch, self.warmup_ratio)

        best_val, best_state = float("inf"), None
        start = len(self.history_)
        for epoch in range(start + 1, start + self.epochs + 1):
            row: dict[str, Any] = {"epoch": epoch}
            train_m = train_one_epoch(self.model_, self.criterion_, train_loader, optimizer,
                                      scheduler, self.device, self.grad_clip)
            row.update({f"train_{k}": v for k, v in train_m.items()})
            if val_loader is not None:
                val_m = evaluate(self.model_, self.criterion_, val_loader, self.device)
                row.update({f"val_{k}": v for k, v in val_m.items()})
                if val_m["loss"] < best_val:
                    best_val = val_m["loss"]
                    self.best_epoch_ = epoch
                    best_state = (copy.deepcopy(self.model_.state_dict()),
                                  copy.deepcopy(self.criterion_.state_dict()))
            self.history_.append(row)
            if self.verbose:
                print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in row.items()))

        if best_state is not None:
            self.model_.load_state_dict(best_state[0])
            self.criterion_.load_state_dict(best_state[1])
        return self

    # ---------- predict ----------
    def _pad_masked_features(self, mask_pad: bool | Iterable[str]) -> set[str]:
        if mask_pad is False:
            return set()
        classification = {f.name: f for f in self.config.features if f.task == TaskType.CLASSIFICATION}
        if mask_pad is True:
            # features predicting a token from their own vocabulary, where class 0 is PAD
            return {n for n, f in classification.items() if not f.is_linear and f.n_classes == f.vocab_size}
        names = set(mask_pad)
        unknown = names - set(classification)
        if unknown:
            raise ValueError(f"mask_pad names must be classification features, got {sorted(unknown)}")
        return names

    @torch.no_grad()
    def predict(self, loader: Iterable[dict], mask_pad: bool | Iterable[str] = False) -> dict[str, torch.Tensor]:
        """Raw outputs for every sample, in loader order, on CPU.

        Args:
            loader: Batches from ``CustomCollate`` (``inference=True`` or False; ``Y`` is ignored).
                Must not shuffle, or outputs won't line up with your samples.
            mask_pad: If True, set the PAD logit (class 0) to ``-inf`` for classification features
                that predict a token from their own vocabulary. Or pass feature names explicitly.

        Returns:
            dict[str, Tensor]: classification ``(N, C)`` logits, regression ``(N, 1)`` values.
        """
        self._check_fitted()
        if isinstance(loader, DataLoader) and isinstance(loader.sampler, RandomSampler):
            raise ValueError("predict() needs a loader with shuffle=False so outputs match sample order")
        masked = self._pad_masked_features(mask_pad)
        self.model_.eval()
        chunks: dict[str, list[torch.Tensor]] = {}
        for batch in loader:
            batch = to_device(batch, self.device)
            outputs = self.model_(batch["X"], batch["mask"])
            for name, value in outputs.items():
                if name in masked:
                    value = value.clone()
                    value[..., 0] = float("-inf")
                chunks.setdefault(name, []).append(value.float().cpu())
        if not chunks:
            raise ValueError("loader produced no batches")
        return {name: torch.cat(parts) for name, parts in chunks.items()}

    # ---------- save / load ----------
    def save(self, path: str | PathLike, metadata: dict[str, Any] | None = None) -> None:
        """Save weights, config, training settings and history to one file.

        Args:
            path: Destination file, e.g. ``"model.pt"``.
            metadata: Your own preprocessing (vocabularies, scaler means/stds, ...), stored as is.
                Plain Python types (dict, list, str, int, float, bool, None) and tensors load
                without ``trusted=True``.
        """
        self._check_fitted()
        from next_sequence import __version__
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "checkpoint_version": CHECKPOINT_VERSION,
            "library_version": __version__,
            "config": self.config.to_dict(),
            "train_settings": self.train_settings(),
            "model_state": {k: v.cpu() for k, v in self.model_.state_dict().items()},
            "criterion_state": {k: v.cpu() for k, v in self.criterion_.state_dict().items()},
            "history": self.history_,
            "best_epoch": self.best_epoch_,
            "metadata": metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | PathLike,
        device: str | torch.device | None = None,
        trusted: bool = False,
    ) -> "NextSequence":
        """Load a model saved with ``save``. The saved metadata is available as ``.metadata_``.

        Args:
            path: File written by ``save``.
            device: ``None`` picks CUDA, then MPS, then CPU. A GPU-trained model loads on CPU.
            trusted: Allow arbitrary pickled objects in ``metadata`` (e.g. numpy arrays).
                Only use it for files you trust: unpickling can run code.
        """
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=not trusted)
        except Exception as exc:
            if trusted:
                raise
            raise RuntimeError(
                f"Could not load {path} in safe mode ({exc}). If the file is yours and its metadata "
                f"contains non-basic types, use NextSequence.load(path, trusted=True)."
            ) from exc
        if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a next-sequence checkpoint")
        if ckpt["checkpoint_version"] > CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint_version {ckpt['checkpoint_version']} is newer than this library "
                f"supports ({CHECKPOINT_VERSION}). Upgrade next-sequence."
            )
        config = NextSequenceConfig.from_dict(ckpt["config"])
        obj = cls(config, device=device, verbose=False, **ckpt["train_settings"])
        obj.model_ = NextSequenceModel(config)
        obj.model_.load_state_dict(ckpt["model_state"])
        obj.model_.to(obj.device).eval()
        obj.criterion_ = UncertaintyWeightedLoss(config.features)
        obj.criterion_.load_state_dict(ckpt["criterion_state"])
        obj.criterion_.to(obj.device)
        obj.history_ = list(ckpt.get("history") or [])
        obj.best_epoch_ = ckpt.get("best_epoch")
        obj.metadata_ = ckpt.get("metadata")
        return obj

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "not fitted"
        return (f"NextSequence(features={[f.name for f in self.config.features]}, "
                f"epochs={self.epochs}, lr={self.lr}, device='{self.device}', {state})")
