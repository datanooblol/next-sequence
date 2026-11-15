from typing import Any, Callable

import torch
from torch.nn.utils.rnn import pad_sequence

from next_sequence.core import Feature


class CustomCollate:
    """Turn a list of samples into one padded batch.

    Each feature has its own function in ``funcs`` that converts the raw values of a batch into
    inputs and targets: ``funcs[name](values, inference) -> (x_list, y_list_or_None)``, where
    ``x_list`` holds one 1-D tensor per sample and ``y_list`` holds one scalar per sample.

    Args:
        features (list[Feature]): Features in the same order the model was configured with.
        feature_mask (str): Name of the feature whose lengths define the padding mask.
        funcs (dict[str, Callable]): One conversion function per feature name.
        inference (bool): If True, targets are not built and ``Y`` is ``None``.
        max_len (int | None): If set, keep only the most recent ``max_len`` steps of every input.
            Set it to the model's ``max_len`` so inputs never exceed what the model supports.

    Returns:
        dict: ``{"X": {name: (B, T) or (B, T, 1)}, "Y": {name: (B,)} or None,
        "mask": (B, T) bool (True = padding), "lengths": (B,)}``
    """

    def __init__(
        self,
        features: list[Feature],
        feature_mask: str,
        funcs: dict[str, Callable],
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
        X: dict[str, list[torch.Tensor]] = {}
        Y: dict[str, list[Any]] = {}
        for feature in self.features:
            values = [sample[feature.name] for sample in batch]
            x, y = self.funcs[feature.name](values, self.inference)
            x = [torch.as_tensor(t) for t in x]
            if self.max_len is not None:
                x = [t[-self.max_len:] for t in x]   # keep the most recent steps
            X[feature.name] = x
            if not self.inference:
                if y is None:
                    raise ValueError(f"funcs['{feature.name}'] returned y=None while inference=False")
                Y[feature.name] = y

        lengths = torch.tensor([len(t) for t in X[self.feature_mask]])

        X_out: dict[str, torch.Tensor] = {}
        for feature in self.features:
            x = pad_sequence(X[feature.name], batch_first=True, padding_value=0)   # right padding
            X_out[feature.name] = x.unsqueeze(-1) if feature.is_linear else x

        Y_out: dict[str, torch.Tensor] | None = None
        if not self.inference:
            Y_out = {name: _stack_targets(y) for name, y in Y.items()}

        max_t = X_out[self.feature_mask].size(1)
        pad_mask = torch.arange(max_t)[None, :] >= lengths[:, None]
        return dict(X=X_out, Y=Y_out, mask=pad_mask, lengths=lengths)


def _stack_targets(y: list[Any]) -> torch.Tensor:
    if all(torch.is_tensor(t) for t in y):
        return torch.stack([t.reshape(()) if t.numel() == 1 else t for t in y])
    return torch.tensor(y)
