"""Preprocessing helpers."""
from collections.abc import Hashable, Iterable
from typing import Any

PAD_TOKEN = 'PAD'
"""Token reserved for padding; always id 0."""
UNK_TOKEN = 'UNK'
"""Token reserved for tokens missing from the vocabulary; always id 1."""


def create_bow(
    sequences: Iterable[Iterable[Hashable]],
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> dict[Hashable, int]:
    """Build a token -> id vocabulary from training sequences.

    Ids are deterministic: ``pad_token`` is 0, ``unk_token`` is 1, and the remaining tokens are
    sorted and numbered from 2. Build it from the **training split only**, and map tokens that
    are missing from it to ``bow[unk_token]``.

    Args:
        sequences (Iterable[Iterable[Hashable]]): Token sequences, e.g. ``[["a", "b"], ["a"]]``.
        pad_token (str): Token reserved for padding (id 0).
        unk_token (str): Token reserved for unknown tokens (id 1).

    Returns:
        dict[Hashable, int]: The vocabulary. Use ``len(bow)`` as the feature's ``vocab_size``.

    Example:
        >>> create_bow([["b", "a"], ["a", "c"]])
        {'PAD': 0, 'UNK': 1, 'a': 2, 'b': 3, 'c': 4}
    """
    tokens: set[Hashable] = set()
    for seq in sequences:
        tokens.update(seq)
    tokens -= {pad_token, unk_token}
    ordered: list[Any]
    try:
        ordered = sorted(tokens)            # type: ignore[type-var]  # TypeError handled below
    except TypeError:                       # mixed types, e.g. ints and strings
        ordered = sorted(tokens, key=lambda t: (type(t).__name__, repr(t)))
    bow: dict[Hashable, int] = {pad_token: 0, unk_token: 1}
    bow.update({token: idx for idx, token in enumerate(ordered, start=2)})
    return bow
