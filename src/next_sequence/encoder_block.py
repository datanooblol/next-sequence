from next_sequence.attention_head import MultiHeadAttention
from next_sequence.position_embedding import RoPE
from next_sequence.embedding import EmbeddingModel
from next_sequence.fusion import BaseFusion
from next_sequence.create_cls import PrependCLS
import torch.nn as nn
from typing import Any

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, rope=None, expansion=4, dropout=0.1):
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

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), pad_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class EncoderCLSBackbone(nn.Module):
    """Write it later after testing
    """
    def __init__(
            self, 
            embedding:EmbeddingModel, 
            fusion:BaseFusion, 
            prepend_cls:PrependCLS, 
            rope:RoPE, 
            n_heads:int,
            num_layers:int=4, 
            expansion:int=4, 
            dropout:float=0.1
        ):
        super().__init__()
        d = fusion.d_out
        assert d % n_heads == 0
        assert rope.cos.shape[-1] == d // n_heads, "RoPE must be built with head_dim"
        assert prepend_cls.cls.shape[-1] == d, "CLS width must equal fusion.d_out"
        self.n_heads = n_heads
        self.d_model = d
        self.num_layers = num_layers
        self.expansion = expansion
        self.dropout = dropout
        self.embedding, self.fusion = embedding, fusion
        self.prepend_cls, self.rope = prepend_cls, rope
        self.layers = nn.ModuleList(
            EncoderBlock(d, n_heads, rope, expansion, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d)        

    def forward(self, X, pad_mask)->Any: 
        """
        Args:
            X (dict): input
            pad_mask (array): (B,T), True = padding
        """
        x = self.fusion(self.embedding(X))
        x, mask = self.prepend_cls(x, pad_mask)
        for layer in self.layers:
            x = layer(x, mask)
        # this will return only the first sequence (CLS)
        return self.final_norm(x)[:, 0]