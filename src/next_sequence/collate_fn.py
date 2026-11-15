"""Batch collation: per-feature collate functions, truncation, ordering, padding and the mask."""
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from torch.nn.utils.rnn import pad_sequence

from next_sequence.core import Feature

if TYPE_CHECKING:
    from next_sequence.model import NextSequenceConfig

ConvertFn: TypeAlias = Callable[[list[Any], bool], tuple[Sequence[Any], Sequence[Any] | None]]
"""``fn(values, inference) -> (inputs, targets)`` for one feature of a batch.

``values`` holds that feature's value from every sample. ``inputs`` must contain one 1-D tensor
(or list) per sample; ``targets`` one scalar per sample, or ``None`` when ``inference`` is True.
Build these with ``make_collate``.
"""

_DTYPES = {"long": torch.long, "int": torch.long, "float": torch.float, "categorical": torch.long,
           "numeric": torch.float}


def make_collate(
    dtype: torch.dtype | str = torch.float,
    x: Callable[[Any], Any] | None = None,
    y: Callable[[list[Any]], Any] | None = None,
) -> ConvertFn:
    """Build a per-feature collate function whose training and inference branches always agree.

    In training mode the last step of each history is the event to predict: inputs are every step
    but the last, and the target is ``y(history)``. In inference mode inputs are the full history
    and there is no target.

    Args:
        dtype (torch.dtype | str): Input dtype: ``torch.long`` (or ``"long"`` / ``"categorical"``)
            for token ids, ``torch.float`` (or ``"float"`` / ``"numeric"``) for numbers.
        x (Callable[[Any], Any] | None): Input value of one step. Default: the step itself.
        y (Callable[[list[Any]], Any] | None): Target computed from the whole training history;
            ``history[-1]`` is the event to predict and ``history[-2]`` the last one the model sees.
            Default: ``history[-1]``.

    Returns:
        ConvertFn: A function to put in ``funcs`` for ``CustomCollate``.

    Raises:
        ValueError: If ``dtype`` is not recognised.

    Examples:
        >>> make_collate("long")                                       # next token
        >>> make_collate("float", y=lambda h: (h[-1] - h[-2]) / std)  # delta
        >>> make_collate("float", x=lambda s: s[0], y=lambda h: h[-1][1])   # (scaled, raw) steps
    """
    if isinstance(dtype, str):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be a torch dtype or one of {sorted(_DTYPES)}, got {dtype!r}")
        dtype = _DTYPES[dtype]
    pick_x: Callable[[Any], Any] = x if x is not None else (lambda step: step)
    make_y: Callable[[list[Any]], Any] = y if y is not None else (lambda history: history[-1])
    torch_dtype: torch.dtype = dtype

    def collate(values: list[Any], inference: bool = False) -> tuple[list[torch.Tensor], list[Any] | None]:
        xs = [torch.tensor([pick_x(step) for step in history], dtype=torch_dtype) for history in values]
        if inference:
            return xs, None
        targets = []
        for i, history in enumerate(values):
            if len(history) < 2:
                raise ValueError(
                    f"sample {i} has a history of length {len(history)}; training needs at least 2 "
                    f"steps (inputs + the target). Drop shorter sequences from the training data."
                )
            try:
                targets.append(make_y(list(history)))
            except IndexError as exc:
                raise ValueError(
                    f"the target function needs a longer history than sample {i} has "
                    f"(length {len(history)}): {exc}. Drop shorter sequences from the training data."
                ) from exc
        return [t[:-1] for t in xs], targets

    return collate


def check_collate(fn: ConvertFn, values: list[Any]) -> dict[str, Any]:
    """Check a per-feature collate function on a few histories before training with it.

    Verifies that inference returns no targets, that there is one target per sample, that both
    branches produce the same dtype, and that training drops exactly the last step.

    Args:
        fn (ConvertFn): The collate function to check.
        values (list[Any]): A few histories of that feature, e.g. ``[s["price"] for s in train[:5]]``.

    Returns:
        dict[str, Any]: ``{"input_dtype": ..., "targets": [...]}`` for a quick look.

    Raises:
        ValueError: With a description of the first problem found.
    """
    x_train, y_train = fn(values, False)
    x_infer, y_infer = fn(values, True)
    if y_infer is not None:
        raise ValueError("with inference=True the function must return targets=None")
    if y_train is None or len(y_train) != len(values):
        raise ValueError(f"with inference=False the function must return one target per sample "
                         f"({len(values)}), got {None if y_train is None else len(y_train)}")
    if len(x_train) != len(values) or len(x_infer) != len(values):
        raise ValueError("the function must return one input sequence per sample")
    for i, (a, b) in enumerate(zip(x_train, x_infer)):
        a, b = torch.as_tensor(a), torch.as_tensor(b)
        if a.dtype != b.dtype:
            raise ValueError(f"training inputs are {a.dtype} but inference inputs are {b.dtype}; "
                             f"both branches must build the same inputs (use make_collate)")
        if len(a) != len(b) - 1:
            raise ValueError(f"sample {i}: training inputs should be the inference inputs without the "
                             f"last step (lengths {len(a)} vs {len(b)})")
    first = torch.as_tensor(x_train[0]) if x_train else torch.empty(0)
    return {"input_dtype": first.dtype, "targets": list(y_train)}


class CustomCollate:
    """Turn a list of samples into one padded batch; use it as a DataLoader ``collate_fn``.

    For every feature, ``funcs[name](values, inference)`` converts the batch's raw values into
    inputs and targets. Inputs are then truncated to the newest ``max_len`` steps, optionally
    reversed (``recent_first``), right-padded with 0, and a padding mask is built from the lengths
    of ``feature_mask``.

    Prefer ``CustomCollate.from_config(config, ...)``: it takes ``max_len`` and ``recent_first``
    from the model configuration, so the data always matches what the model expects.

    Args:
        features (list[Feature]): Features in the same order the model was configured with.
        feature_mask (str): Name of the feature whose lengths define the padding mask.
        funcs (dict[str, ConvertFn]): One collate function per feature name (see ``make_collate``).
        inference (bool): If True, targets are not built and ``Y`` is ``None``.
        max_len (int | None): If set, keep only the most recent ``max_len`` steps of every input.
            Use the model's ``max_len`` so inputs never exceed what the model supports.
        recent_first (bool): Reverse every input so the newest step comes first (after truncation).
            Must match ``NextSequenceConfig.recent_first``.
        check_leakage (bool): In training mode, raise if a function returns inputs as long as the
            history: they would still contain the last step, which is the target, and the model would
            learn to copy the answer. Turn it off only if your steps store a label for the *next*
            event, so the full history is a legitimate input.

    Raises:
        ValueError: If ``feature_mask`` is not a feature, a function is missing, or ``max_len < 1``.

    Example:
        >>> collate = CustomCollate.from_config(config, "item", funcs)
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
        recent_first: bool = True,
        check_leakage: bool = True,
    ):
        names = [f.name for f in features]
        if feature_mask not in names:
            raise ValueError(f"feature_mask '{feature_mask}' is not one of the features {names}")
        missing = [n for n in names if n not in funcs]
        if missing:
            raise ValueError(f"funcs is missing a collate function for: {missing}")
        extra = sorted(set(funcs) - set(names))
        if extra:
            raise ValueError(f"funcs has functions for unknown features {extra}; features are {names}")
        if max_len is not None and max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self.features = features
        self.feature_mask = feature_mask
        self.funcs = funcs
        self.inference = inference
        self.max_len = max_len
        self.recent_first = recent_first
        self.check_leakage = check_leakage

    @classmethod
    def from_config(
        cls,
        config: "NextSequenceConfig",
        feature_mask: str,
        funcs: dict[str, ConvertFn],
        inference: bool = False,
        check_leakage: bool = True,
    ) -> "CustomCollate":
        """Build a collate that matches a model configuration (features, ``max_len``, ``recent_first``).

        Args:
            config (NextSequenceConfig): The model configuration.
            feature_mask (str): Name of the feature whose lengths define the padding mask.
            funcs (dict[str, ConvertFn]): One collate function per feature name.
            inference (bool): If True, targets are not built.
            check_leakage (bool): See ``CustomCollate``.

        Returns:
            CustomCollate: The collate function for a DataLoader.
        """
        return cls(config.features, feature_mask, funcs, inference=inference,
                   max_len=config.max_len, recent_first=config.recent_first,
                   check_leakage=check_leakage)

    @staticmethod
    def _check_no_leakage(name: str, values: list[Any], xs: list[torch.Tensor]) -> None:
        """Raise if training inputs are as long as the history, i.e. still contain the target step."""
        for i, (history, x) in enumerate(zip(values, xs)):
            if isinstance(history, (str, bytes)) or not hasattr(history, "__len__"):
                continue                              # not a list of steps: nothing to compare
            if len(history) > 0 and len(x) >= len(history):
                raise ValueError(
                    f"funcs['{name}'] returned training inputs with {len(x)} steps for sample {i}, whose "
                    f"history has {len(history)}: the inputs still contain the last step, which is the "
                    f"target, so the model would learn to copy the answer. In training mode drop it "
                    f"(return x[:-1]) or build the function with make_collate. If your steps really hold "
                    f"a label for the next event, pass check_leakage=False."
                )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into one batch.

        Args:
            batch (list[dict[str, Any]]): Samples, one dict per sequence.

        Returns:
            dict[str, Any]: ``{"X": {name: (B, T) or (B, T, 1)}, "Y": {name: (B,)} or None,
            "mask": (B, T) bool (True = padding), "lengths": (B,)}``. Numeric features
            (``is_linear=True``) get a trailing dimension of 1.

        Raises:
            KeyError: If a sample lacks a feature.
            ValueError: If a collate function returns something malformed, training inputs still
                contain the target step (see ``check_leakage``), or the features of one sample have
                different lengths.
        """
        X: dict[str, list[torch.Tensor]] = {}
        Y: dict[str, Sequence[Any]] = {}
        for feature in self.features:
            try:
                values = [sample[feature.name] for sample in batch]
            except KeyError:
                raise KeyError(f"a sample has no key '{feature.name}'; every sample needs all features "
                               f"{[f.name for f in self.features]}") from None
            raw_x, y = self.funcs[feature.name](values, self.inference)
            if len(raw_x) != len(batch):
                raise ValueError(f"funcs['{feature.name}'] returned {len(raw_x)} inputs for a batch of {len(batch)}")
            xs = [torch.as_tensor(t) for t in raw_x]
            for i, t in enumerate(xs):
                if t.dim() != 1:
                    raise ValueError(f"funcs['{feature.name}'] must return one 1-D sequence per sample; "
                                     f"sample {i} has shape {tuple(t.shape)}")
            if not self.inference and self.check_leakage:
                self._check_no_leakage(feature.name, values, xs)
            if self.max_len is not None:
                xs = [t[-self.max_len:] for t in xs]   # keep the most recent steps
            if self.recent_first:
                xs = [t.flip(0) for t in xs]           # newest step first, next to CLS
            X[feature.name] = xs
            if not self.inference:
                if y is None:
                    raise ValueError(f"funcs['{feature.name}'] returned y=None while inference=False")
                if len(y) != len(batch):
                    raise ValueError(f"funcs['{feature.name}'] returned {len(y)} targets for a batch of {len(batch)}")
                Y[feature.name] = y

        lengths = torch.tensor([len(t) for t in X[self.feature_mask]])
        for feature in self.features:
            other = torch.tensor([len(t) for t in X[feature.name]])
            if not torch.equal(other, lengths):
                i = int((other != lengths).nonzero()[0])
                raise ValueError(
                    f"sample {i}: '{feature.name}' has {int(other[i])} input steps but "
                    f"'{self.feature_mask}' has {int(lengths[i])}; all features of a sample must have "
                    f"the same length"
                )

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
