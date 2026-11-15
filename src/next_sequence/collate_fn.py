"""Batch collation: per-feature conversion, truncation, padding and the padding mask."""
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import torch
from torch.nn.utils.rnn import pad_sequence

from next_sequence.core import Feature

ConvertFn: TypeAlias = Callable[[list[Any], bool], tuple[Sequence[Any], Sequence[Any] | None]]
"""``fn(values, inference) -> (inputs, targets)`` for one feature of a batch.

``values`` holds that feature's value from every sample. ``inputs`` must contain one 1-D tensor
(or list) per sample; ``targets`` one scalar per sample, or ``None`` when ``inference`` is True.
"""


class CustomCollate:
    """Turn a list of samples into one padded batch; use it as a DataLoader ``collate_fn``.

    For every feature, ``funcs[name](values, inference)`` converts the batch's raw values into
    inputs and targets. Inputs are then truncated to the newest ``max_len`` steps, right-padded with
    0, and a padding mask is built from the lengths of ``feature_mask``.

    Args:
        features (list[Feature]): Features in the same order the model was configured with.
        feature_mask (str): Name of the feature whose lengths define the padding mask.
        funcs (dict[str, ConvertFn]): One conversion function per feature name.
        inference (bool): If True, targets are not built and ``Y`` is ``None``.
        max_len (int | None): If set, keep only the most recent ``max_len`` steps of every input.
            Use the model's ``max_len`` so inputs never exceed what the model supports.

    Raises:
        ValueError: If ``feature_mask`` is not a feature, a function is missing, or ``max_len < 1``.

    Example:
        >>> collate = CustomCollate(features, "item", funcs, max_len=50)
        >>> batch = collate(samples[:4])
        >>> batch["X"]["item"].shape, batch["mask"].shape     # (4, T), (4, T)
    """

    def __init__(
        self,
        features: list[Feature],
        feature_mask: str,
        funcs: dict[str, ConvertFn],
        inference: bool = False,
        max_len: int | None = None,
    ):
        names = {f.name for f in features}
        if feature_mask not in names:
            raise ValueError(f"feature_mask '{feature_mask}' is not one of the features {sorted(names)}")
        missing = names - set(funcs)
        if missing:
            raise ValueError(f"funcs is missing a function for: {sorted(missing)}")
        if max_len is not None and max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self.features = features
        self.feature_mask = feature_mask
        self.funcs = funcs
        self.inference = inference
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into one batch.

        Args:
            batch (list[dict[str, Any]]): Samples, one dict per sequence.

        Returns:
            dict[str, Any]: ``{"X": {name: (B, T) or (B, T, 1)}, "Y": {name: (B,)} or None,
            "mask": (B, T) bool (True = padding), "lengths": (B,)}``. Numeric features
            (``is_linear=True``) get a trailing dimension of 1.

        Raises:
            ValueError: If a conversion function returns ``y=None`` while ``inference=False``.
        """
        X: dict[str, list[torch.Tensor]] = {}
        Y: dict[str, Sequence[Any]] = {}
        for feature in self.features:
            values = [sample[feature.name] for sample in batch]
            raw_x, y = self.funcs[feature.name](values, self.inference)
            xs = [torch.as_tensor(t) for t in raw_x]
            if self.max_len is not None:
                xs = [t[-self.max_len:] for t in xs]   # keep the most recent steps
            X[feature.name] = xs
            if not self.inference:
                if y is None:
                    raise ValueError(f"funcs['{feature.name}'] returned y=None while inference=False")
                Y[feature.name] = y

        lengths = torch.tensor([len(t) for t in X[self.feature_mask]])

        X_out: dict[str, torch.Tensor] = {}
        for feature in self.features:
            padded = pad_sequence(X[feature.name], batch_first=True, padding_value=0)   # right padding
            X_out[feature.name] = padded.unsqueeze(-1) if feature.is_linear else padded

        Y_out: dict[str, torch.Tensor] | None = None
        if not self.inference:
            Y_out = {name: _stack_targets(y) for name, y in Y.items()}

        max_t = X_out[self.feature_mask].size(1)
        pad_mask = torch.arange(max_t)[None, :] >= lengths[:, None]
        return dict(X=X_out, Y=Y_out, mask=pad_mask, lengths=lengths)


def _stack_targets(y: Sequence[Any]) -> torch.Tensor:
    """Stack per-sample targets (Python scalars or 0-d / 1-element tensors) into a ``(B,)`` tensor."""
    if all(torch.is_tensor(t) for t in y):
        return torch.stack([t.reshape(()) if t.numel() == 1 else t for t in y])
    return torch.tensor(y)
