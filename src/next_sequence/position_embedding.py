"""Rotary position embeddings (RoPE, Su et al., 2021)."""
from enum import Enum

import torch
import torch.nn as nn


class RoPEStrategy(Enum):
    """How the dimensions of each head are paired for rotation.

    Attributes:
        INTERLEAVED: Pairs neighbouring dimensions ``(0, 1), (2, 3), ...``.
    """
    INTERLEAVED = "interleaved"


class RoPE(nn.Module):
    """Rotate query/key vectors by an angle that grows with their position.

    The cos/sin tables are precomputed for ``max_len`` positions and are not saved in the
    ``state_dict`` (they are rebuilt from the arguments).

    Args:
        head_dim (int): Width of one attention head. Must be even.
        max_len (int): Number of positions to precompute, including the CLS token.
        strategy (RoPEStrategy | str): How dimensions are paired for rotation.
        base (float): Base of the geometric frequency progression.

    Raises:
        ValueError: If ``head_dim`` is odd or ``max_len`` is not positive.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_len: int,
        strategy: RoPEStrategy | str = RoPEStrategy.INTERLEAVED,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self.strategy = RoPEStrategy(strategy)
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        theta = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(max_len).float(), theta)      # (max_len, head_dim / 2)
        emb = angles.repeat_interleave(2, dim=-1)                       # (max_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Map each pair ``(x1, x2)`` to ``(-x2, x1)``."""
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rotation for positions ``0 .. T-1``.

        Args:
            x (torch.Tensor): Queries or keys of shape ``(B, T, n_heads, head_dim)``.

        Returns:
            torch.Tensor: Rotated tensor with the same shape.

        Raises:
            ValueError: If ``T`` is larger than ``max_len``.
        """
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError(
                f"seq len {T} > max_len {self.max_len} (including CLS). Truncate inputs with "
                f"CustomCollate(max_len=...) or increase NextSequenceConfig.max_len."
            )
        cos = self.cos[:T][None, :, None, :]
        sin = self.sin[:T][None, :, None, :]
        return x * cos + self._rotate(x) * sin
