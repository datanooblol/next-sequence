# torch
import torch.nn as nn
from next_sequence.position_embedding import RoPE
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, rope:RoPE|None=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wQ, self.wK, self.wV, self.wO = (nn.Linear(d_model, d_model) for _ in range(4))
        self.rope = rope

    def forward(self, x:torch.Tensor, pad_mask:torch.Tensor|None=None)->torch.Tensor:
        B, T, d = x.shape
        q = self.wQ(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wK(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wV(x).view(B, T, self.n_heads, self.head_dim)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        out = scores.softmax(-1) @ v
        return self.wO(out.transpose(1, 2).reshape(B, T, d))