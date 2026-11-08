from enum import Enum
import torch
import torch.nn as nn

class RoPEStrategy(Enum):
    INTERLEAVED = "interleaved"
    HALF_SPLIT = "half_split"

class RoPE(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
            self, 
            head_dim: int, 
            max_len: int,
            strategy: RoPEStrategy = RoPEStrategy.INTERLEAVED, 
            base: float = 10000.0
        ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.strategy = RoPEStrategy(strategy)

        theta = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(max_len).float(), theta)

        if self.strategy == RoPEStrategy.INTERLEAVED:
            emb = angles.repeat_interleave(2, dim=-1)
        else:
            emb = torch.cat([angles, angles], dim=-1)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def _rotate(self, x):
        if self.strategy == RoPEStrategy.INTERLEAVED:
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        T = x.shape[1]
        assert T <= self.cos.shape[0], f"seq len {T} > max_len {self.cos.shape[0]}"
        cos = self.cos[:T][None, :, None, :]
        sin = self.sin[:T][None, :, None, :]
        return x * cos + self._rotate(x) * sin