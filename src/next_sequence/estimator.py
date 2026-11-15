"""High-level wrapper: ``NextSequence(config).fit(train_loader, val_loader).predict(test_loader)``."""
import copy
import math
import warnings
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.utils.data import DataLoader, RandomSampler

from next_sequence.collate_fn import CustomCollate
from next_sequence.core import Feature, TaskType
from next_sequence.loss_fn import UncertaintyWeightedLoss
from next_sequence.model import NextSequenceConfig, NextSequenceModel
from next_sequence.train import (
    build_optimizer, build_scheduler, evaluate, resolve_device, to_device, train_one_epoch,
)

CHECKPOINT_FORMAT = "next-sequence"
CHECKPOINT_VERSION = 1


class NotFittedError(RuntimeError):
    """Raised when ``predict`` / ``save`` is called before ``fit`` or ``load``."""


class Interval(NamedTuple):
    """A prediction interval for every sample (all tensors are ``(N,)``).

    Attributes:
        lower (torch.Tensor): Lower bound.
        median (torch.Tensor): Median (the central prediction).
        upper (torch.Tensor): Upper bound.
        coverage (float): Intended share of true values inside ``[lower, upper]``.
    """
    lower: torch.Tensor
    median: torch.Tensor
    upper: torch.Tensor
    coverage: float


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

    Attributes:
        model_ (NextSequenceModel): The trained network (after ``fit`` or ``load``).
        criterion_ (UncertaintyWeightedLoss): The loss with the learned task weights.
        history_ (list[dict[str, Any]]): One row of metrics per epoch.
        best_epoch_ (int | None): Epoch whose weights were kept (lowest validation loss).
        metadata_ (dict[str, Any] | None): The ``metadata`` saved with the model (after ``load``).
        calibration_ (dict[str, dict[float, float]] | None): Interval adjustments from ``calibrate``.

    Raises:
        TypeError: If ``config`` is not a ``NextSequenceConfig``.
        ValueError: If a training setting is out of range.

    Example:
        >>> model = NextSequence(config, epochs=20, seed=0)
        >>> model.fit(train_loader, val_loader)
        >>> out = model.predict(test_loader)
        >>> model.save("model.pt", metadata={"bow": bow})
    """

    model_: NextSequenceModel
    criterion_: UncertaintyWeightedLoss
    history_: list[dict[str, Any]]
    best_epoch_: int | None
    metadata_: dict[str, Any] | None
    calibration_: dict[str, dict[float, float]] | None

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
        """True after ``fit`` or ``load``."""
        return hasattr(self, "model_")

    def _check_fitted(self) -> None:
        """Raise ``NotFittedError`` unless ``fit`` or ``load`` has been called."""
        if not self.is_fitted:
            raise NotFittedError("This NextSequence is not fitted yet. Call fit() or load() first.")

    def _build(self) -> None:
        """Create a fresh model and loss on ``self.device``."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.model_ = NextSequenceModel(self.config).to(self.device)
        self.criterion_ = UncertaintyWeightedLoss(self.config.features).to(self.device)
        self.history_ = []
        self.best_epoch_ = None
        self.metadata_ = None
        self.calibration_ = None

    def _feature(self, name: str) -> Feature:
        """Look up a feature by name, with a helpful error."""
        for f in self.config.features:
            if f.name == name:
                return f
        raise KeyError(f"'{name}' is not a feature; features are {[f.name for f in self.config.features]}")

    def _quantile_features(self) -> list[Feature]:
        """All features with task QUANTILE."""
        return [f for f in self.config.features if f.task == TaskType.QUANTILE]

    def _check_loader(self, loader: Any, needs_targets: bool, role: str) -> None:
        """Catch loaders whose settings don't match the model, before any work is done."""
        if not isinstance(loader, DataLoader):
            return
        if role == "predict" and isinstance(loader.sampler, RandomSampler):
            raise ValueError("predict() needs a loader with shuffle=False so outputs match sample order")
        collate = loader.collate_fn
        if not isinstance(collate, CustomCollate):
            return
        expected = [f.name for f in self.config.features]
        got = [f.name for f in collate.features]
        if got != expected:
            raise ValueError(f"the {role} loader has features {got} but the model has {expected}. "
                             f"Build it with CustomCollate.from_config(config, ...).")
        if collate.recent_first != self.config.recent_first:
            raise ValueError(f"the {role} loader has recent_first={collate.recent_first} but the model "
                             f"expects recent_first={self.config.recent_first}. Build it with "
                             f"CustomCollate.from_config(config, ...).")
        if collate.max_len is not None and collate.max_len > self.config.max_len:
            raise ValueError(f"the {role} loader keeps up to {collate.max_len} steps but the model "
                             f"supports max_len={self.config.max_len}. Use CustomCollate.from_config(config, ...).")
        if collate.max_len is None:
            warnings.warn(f"the {role} loader has no max_len: histories longer than {self.config.max_len} "
                          f"steps will raise an error. Use CustomCollate.from_config(config, ...).",
                          stacklevel=3)
        if needs_targets and collate.inference:
            raise ValueError(f"the {role} loader was built with inference=True, so it has no targets. "
                             f"Build it with inference=False.")

    def train_settings(self) -> dict[str, Any]:
        """Training settings as a dict (saved in checkpoints).

        Returns:
            dict[str, Any]: ``epochs``, ``lr``, ``weight_decay``, ``warmup_ratio``, ``grad_clip``, ``seed``.
        """
        return dict(epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
                    warmup_ratio=self.warmup_ratio, grad_clip=self.grad_clip, seed=self.seed)

    def task_weights(self) -> dict[str, float]:
        """Learned effective weight of each task loss.

        Returns:
            dict[str, float]: Feature name -> weight multiplying that feature's loss.

        Raises:
            NotFittedError: If called before ``fit`` or ``load``.
        """
        self._check_fitted()
        return self.criterion_.weights()

    # ---------- fit ----------
    def fit(
        self,
        train_loader: Iterable[dict[str, Any]],
        val_loader: Iterable[dict[str, Any]] | None = None,
    ) -> "NextSequence":
        """Train the model.

        Args:
            train_loader (Iterable[dict[str, Any]]): Batches from ``CustomCollate(inference=False)``;
                must have a length (e.g. a ``DataLoader``). Shuffle it.
            val_loader (Iterable[dict[str, Any]] | None): Validation batches. When given, the weights
                of the epoch with the lowest validation loss are restored at the end.

        Returns:
            NextSequence: ``self``, so calls can be chained.

        Raises:
            TypeError: If ``train_loader`` has no length.
            ValueError: If a loader is empty, has no targets or doesn't match the model.
            FloatingPointError: If the loss becomes NaN or infinite.
        """
        self._check_loader(train_loader, needs_targets=True, role="train")
        if val_loader is not None:
            self._check_loader(val_loader, needs_targets=True, role="validation")
        try:
            steps_per_epoch = len(train_loader)          # type: ignore[arg-type]
        except TypeError:
            raise TypeError("train_loader must have a length (e.g. a DataLoader over a map-style dataset)")
        if steps_per_epoch == 0:
            raise ValueError("train_loader is empty")

        if not (self.warm_start and self.is_fitted):
            self._build()
        else:
            self.calibration_ = None                     # new weights invalidate an old calibration
            if self.seed is not None:
                torch.manual_seed(self.seed)

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
        """Resolve ``mask_pad`` to the set of feature names whose PAD logit is masked."""
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
    def _run(
        self, loader: Iterable[dict[str, Any]], with_targets: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Raw model outputs (and targets) for the whole loader, on CPU, in loader order."""
        self.model_.eval()
        chunks: dict[str, list[torch.Tensor]] = {}
        targets: dict[str, list[torch.Tensor]] = {}
        for batch in loader:
            if with_targets and batch.get("Y") is None:
                raise ValueError("this needs targets; build the loader with inference=False")
            batch = to_device(batch, self.device)
            for name, value in self.model_(batch["X"], batch["mask"]).items():
                chunks.setdefault(name, []).append(value.float().cpu())
                if with_targets:
                    targets.setdefault(name, []).append(batch["Y"][name].float().cpu())
        if not chunks:
            raise ValueError("loader produced no batches")
        return ({n: torch.cat(v) for n, v in chunks.items()},
                {n: torch.cat(v) for n, v in targets.items()})

    def _apply_calibration(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Shift each interval by its calibrated offset, keeping all bounds ordered."""
        if not self.calibration_:
            return outputs
        outputs = dict(outputs)
        for f in self._quantile_features():
            offsets = self.calibration_.get(f.name)
            if not offsets:
                continue
            out = outputs[f.name].clone()
            m = len(f.coverages)
            for j, c in enumerate(f.coverages):
                q = offsets.get(c, 0.0)
                out[:, m - 1 - j] -= q
                out[:, m + 1 + j] += q
            upper = torch.cummax(out[:, m:], dim=-1).values                       # median outward
            lower = torch.cummin(out[:, :m + 1].flip(-1), dim=-1).values.flip(-1)
            outputs[f.name] = torch.cat([lower[:, :m], upper], dim=-1)
        return outputs

    def predict(
        self,
        loader: Iterable[dict[str, Any]],
        mask_pad: bool | Iterable[str] = False,
        calibrated: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Raw outputs for every sample, in loader order, on CPU.

        Use ``CustomCollate(inference=False)`` to evaluate (input = all but the last event,
        ``Y`` is ignored) or ``inference=True`` to predict the event after the full history.

        Args:
            loader (Iterable[dict[str, Any]]): Batches from ``CustomCollate``. Must not shuffle,
                or outputs won't line up with your samples.
            mask_pad (bool | Iterable[str]): If True, set the PAD logit (class 0) to ``-inf`` for
                classification features that predict a token from their own vocabulary
                (categorical input with ``n_classes == vocab_size``). Or pass feature names.
            calibrated (bool): Apply the interval adjustments from ``calibrate`` (if any).

        Returns:
            dict[str, torch.Tensor]: Per predicted feature, in the scale used for training:
            classification ``(N, n_classes)`` logits; regression and positive regression ``(N, 1)``;
            quantile ``(N, len(feature.quantiles))`` from the lowest quantile up (read them with
            ``interval``).

        Raises:
            NotFittedError: If called before ``fit`` or ``load``.
            ValueError: If the loader shuffles, is empty or doesn't match the model, or ``mask_pad``
                names a non-classification feature.
        """
        self._check_fitted()
        self._check_loader(loader, needs_targets=False, role="predict")
        masked = self._pad_masked_features(mask_pad)
        outputs, _ = self._run(loader, with_targets=False)
        for name in masked:
            outputs[name] = outputs[name].clone()
            outputs[name][..., 0] = float("-inf")
        return self._apply_calibration(outputs) if calibrated else outputs

    def interval(
        self,
        outputs: dict[str, torch.Tensor],
        name: str,
        coverage: float | None = None,
    ) -> Interval:
        """Read one prediction interval of a QUANTILE feature from ``predict`` output.

        Args:
            outputs (dict[str, torch.Tensor]): The result of ``predict``.
            name (str): A QUANTILE feature.
            coverage (float | None): One of the feature's ``coverage`` levels. ``None`` uses the
                widest one.

        Returns:
            Interval: ``lower``, ``median``, ``upper`` (each ``(N,)``) and ``coverage``.

        Raises:
            KeyError: If ``name`` is not a feature.
            ValueError: If the feature is not QUANTILE or that coverage was not trained.

        Example:
            >>> out = model.predict(test_loader)
            >>> iv = model.interval(out, "price", coverage=0.5)
            >>> iv.lower, iv.median, iv.upper
        """
        feature = self._feature(name)
        if feature.task != TaskType.QUANTILE:
            raise ValueError(f"'{name}' is a {TaskType(feature.task).value} feature; interval() needs "
                             f"task='quantile'")
        levels = feature.coverages
        chosen = levels[-1] if coverage is None else coverage
        match = [c for c in levels if math.isclose(c, chosen)]
        if not match:
            raise ValueError(f"'{name}' was trained for coverage {list(levels)}, not {chosen}. "
                             f"Add it to Feature(coverage=...) and train again to predict it.")
        j = levels.index(match[0])
        m = len(levels)
        pred = outputs[name]
        return Interval(lower=pred[:, m - 1 - j], median=pred[:, m], upper=pred[:, m + 1 + j],
                        coverage=match[0])

    def interval_coverage(
        self, loader: Iterable[dict[str, Any]], calibrated: bool = True
    ) -> list[dict[str, Any]]:
        """Measure how often the true value falls inside each interval.

        Args:
            loader (Iterable[dict[str, Any]]): Batches with targets (``inference=False``), e.g.
                the validation or test split.
            calibrated (bool): Use the adjustments from ``calibrate`` (if any).

        Returns:
            list[dict[str, Any]]: One row per QUANTILE feature and coverage level with
            ``feature``, ``coverage`` (intended), ``observed`` (measured share inside) and
            ``mean_width``. ``pandas.DataFrame(rows)`` turns it into a table.

        Raises:
            NotFittedError: If called before ``fit`` or ``load``.
            ValueError: If there is no QUANTILE feature or the loader has no targets.
        """
        self._check_fitted()
        if not self._quantile_features():
            raise ValueError("interval_coverage() needs at least one feature with task='quantile'")
        self._check_loader(loader, needs_targets=True, role="interval_coverage")
        outputs, targets = self._run(loader, with_targets=True)
        if calibrated:
            outputs = self._apply_calibration(outputs)
        rows = []
        for f in self._quantile_features():
            y = targets[f.name]
            for c in f.coverages:
                iv = self.interval(outputs, f.name, c)
                inside = ((y >= iv.lower) & (y <= iv.upper)).float().mean().item()
                rows.append({"feature": f.name, "coverage": c, "observed": inside,
                             "mean_width": (iv.upper - iv.lower).mean().item()})
        return rows

    def calibrate(self, loader: Iterable[dict[str, Any]]) -> dict[str, dict[float, float]]:
        """Adjust every interval so its coverage on ``loader`` matches the intended level.

        Conformalized quantile regression: for each interval, how far true values fall outside it
        is measured on ``loader``, and the interval is widened (or narrowed) by the matching
        quantile of those distances. Use data the model was **not trained on**, e.g. the
        validation split. ``predict`` applies the adjustment and ``save`` stores it; training
        again clears it.

        Args:
            loader (Iterable[dict[str, Any]]): Batches with targets (``inference=False``).

        Returns:
            dict[str, dict[float, float]]: Feature -> coverage -> offset added on both sides
            (negative means the interval was narrowed).

        Raises:
            NotFittedError: If called before ``fit`` or ``load``.
            ValueError: If there is no QUANTILE feature or the loader has no targets.
        """
        self._check_fitted()
        if not self._quantile_features():
            raise ValueError("calibrate() needs at least one feature with task='quantile'")
        self._check_loader(loader, needs_targets=True, role="calibrate")
        outputs, targets = self._run(loader, with_targets=True)
        calibration: dict[str, dict[float, float]] = {}
        for f in self._quantile_features():
            y = targets[f.name]
            n = len(y)
            calibration[f.name] = {}
            for c in f.coverages:
                iv = self.interval(outputs, f.name, c)
                scores = torch.maximum(iv.lower - y, y - iv.upper).sort().values
                k = math.ceil((n + 1) * c)
                if k > n:
                    warnings.warn(f"{f.name}: {n} samples are too few to calibrate coverage {c} exactly; "
                                  f"using the largest observed miss.", stacklevel=2)
                    k = n
                calibration[f.name][c] = scores[k - 1].item()
        self.calibration_ = calibration
        return calibration

    # ---------- save / load ----------
    def save(self, path: str | PathLike[str], metadata: dict[str, Any] | None = None) -> None:
        """Save weights, config, training settings, history, calibration and your metadata to one file.

        Args:
            path (str | PathLike[str]): Destination file, e.g. ``"model.pt"``. Parent folders
                are created.
            metadata (dict[str, Any] | None): Your own preprocessing (vocabularies, scaler
                means/stds, ...), stored as is. Plain Python types (dict, list, str, int, float,
                bool, None) and tensors load without ``trusted=True``.

        Raises:
            NotFittedError: If called before ``fit`` or ``load``.
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
            "calibration": self.calibration_,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.metadata_ = metadata

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        device: str | torch.device | None = None,
        trusted: bool = False,
    ) -> "NextSequence":
        """Load a model saved with ``save``. The saved metadata is available as ``.metadata_``.

        Args:
            path (str | PathLike[str]): File written by ``save``.
            device (str | torch.device | None): ``None`` picks CUDA, then MPS, then CPU. A
                GPU-trained model loads on a CPU-only machine.
            trusted (bool): Allow arbitrary pickled objects in ``metadata`` (e.g. numpy arrays).
                Only use it for files you trust: unpickling can run code.

        Returns:
            NextSequence: A fitted instance, ready for ``predict`` or further ``fit`` with
            ``warm_start=True``.

        Raises:
            RuntimeError: If the file can't be loaded in safe mode.
            ValueError: If the file is not a next-sequence checkpoint or is from a newer version.
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
        obj.calibration_ = ckpt.get("calibration")
        return obj

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "not fitted"
        return (f"NextSequence(features={[f.name for f in self.config.features]}, "
                f"epochs={self.epochs}, lr={self.lr}, device='{self.device}', {state})")
