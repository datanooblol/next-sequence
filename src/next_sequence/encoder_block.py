"""Transformer encoder block and the CLS backbone that ties the input pipeline together."""
import torch
import torch.nn as nn

from next_sequence.attention_head import MultiHeadAttention
from next_sequence.embedding import EmbeddingModel
from next_sequence.fusion import BaseFusion
from next_sequence.position_embedding import RoPE
from next_sequence.prepend_cls import PrependCLS


class EncoderBlock(nn.Module):
    """Pre-norm transformer block: ``x + Attn(LN(x))``, then ``x + FFN(LN(x))``.

    Args:
        d_model (int): Width of the sequence representation.
        n_heads (int): Number of attention heads.
        rope (RoPE | None): Rotary position embedding shared by all blocks.
        expansion (int): The feed-forward hidden width is ``expansion * d_model``.
        dropout (float): Dropout applied to the attention and feed-forward outputs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RoPE | None = None,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the block.

        Args:
            x (torch.Tensor): ``(B, T, d_model)``.
            pad_mask (torch.Tensor | None): Bool ``(B, T)``, True = padding.

        Returns:
            torch.Tensor: ``(B, T, d_model)``.
        """
        x = x + self.drop(self.attn(self.norm1(x), pad_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class EncoderCLSBackbone(nn.Module):
    """Embedding -> fusion -> CLS -> stacked encoder blocks -> CLS representation.

    Args:
        embedding (EmbeddingModel): Per-feature embeddings.
        fusion (BaseFusion): Combines the embeddings; its ``d_out`` sets the model width.
        prepend_cls (PrependCLS): CLS token of width ``fusion.d_out``.
        rope (RoPE): Rotary embedding with ``head_dim = fusion.d_out // n_heads`` and room for
            ``max_len + 1`` positions (the CLS token included).
        n_heads (int): Attention heads per block.
        num_layers (int): Number of encoder blocks.
        expansion (int): Feed-forward width multiplier.
        dropout (float): Dropout inside each block.

    Raises:
        ValueError: If the widths of the parts don't match.
    """

    def __init__(
        self,
        embedding: EmbeddingModel,
        fusion: BaseFusion,
        prepend_cls: PrependCLS,
        rope: RoPE,
        n_heads: int,
        num_layers: int = 4,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        d = fusion.d_out
        if d % n_heads:
            raise ValueError(f"fusion.d_out ({d}) must be divisible by n_heads ({n_heads})")
        if rope.head_dim != d // n_heads:
            raise ValueError(f"RoPE head_dim ({rope.head_dim}) must equal d_out // n_heads ({d // n_heads})")
        if prepend_cls.cls.shape[-1] != d:
            raise ValueError(f"CLS width ({prepend_cls.cls.shape[-1]}) must equal fusion.d_out ({d})")
        self.n_heads = n_heads
        self.d_model = d
        self.num_layers = num_layers
        self.expansion = expansion
        self.dropout = dropout
        self.embedding, self.fusion = embedding, fusion
        self.prepend_cls, self.rope = prepend_cls, rope
        self.layers = nn.ModuleList(
            EncoderBlock(d, n_heads, rope, expansion, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d)

    def forward(self, X: dict[str, torch.Tensor], pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch into one vector per sequence.

        Args:
            X (dict[str, torch.Tensor]): Batch inputs, one tensor per feature (see ``EmbeddingModel``).
            pad_mask (torch.Tensor | None): Bool ``(B, T)``, True = padding. ``None`` means no padding.

        Returns:
            torch.Tensor: CLS representation of shape ``(B, d_model)``.
        """
        x = self.fusion(self.embedding(X))
        x, mask = self.prepend_cls(x, pad_mask)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)[:, 0]
