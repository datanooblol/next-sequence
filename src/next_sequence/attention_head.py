"""Multi-head self-attention with optional rotary position embeddings."""
import torch
import torch.nn as nn

from next_sequence.position_embedding import RoPE


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    RoPE, when given, is applied to the queries and keys of every head, so attention scores depend
    on the relative distance between positions.

    Args:
        d_model (int): Width of the input and output. Must be divisible by ``n_heads``.
        n_heads (int): Number of attention heads. Each head has width ``d_model // n_heads``.
        rope (RoPE | None): Rotary position embedding built with ``head_dim = d_model // n_heads``.
            ``None`` disables positional information inside attention.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``n_heads``.
    """

    def __init__(self, d_model: int, n_heads: int, rope: RoPE | None = None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wQ, self.wK, self.wV, self.wO = (nn.Linear(d_model, d_model) for _ in range(4))
        self.rope = rope

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attend over the sequence.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, d_model)``.
            pad_mask (torch.Tensor | None): Bool tensor ``(B, T)``, True where the position is
                padding. Padded positions are never attended to.

        Returns:
            torch.Tensor: Output of shape ``(B, T, d_model)``.
        """
        B, T, d = x.shape
        q = self.wQ(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wK(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wV(x).view(B, T, self.n_heads, self.head_dim)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B, heads, T, head_dim)
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5    # (B, heads, T, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        out = scores.softmax(-1) @ v
        return self.wO(out.transpose(1, 2).reshape(B, T, d))
