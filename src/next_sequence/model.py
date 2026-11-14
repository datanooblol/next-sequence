import torch
import torch.nn as nn
from dataclasses import dataclass
from next_sequence.core import Feature
from next_sequence.position_embedding import RoPEStrategy, RoPE
from next_sequence.fusion_registry import FusionStrategy, FUSION_REGISTRY
from next_sequence.embedding import EmbeddingModel
from next_sequence.encoder_block import EncoderCLSBackbone
from next_sequence.prepend_cls import PrependCLS
from next_sequence.multi_task_head import MultiTaskHeads
from next_sequence.multi_task_model import MultiTaskModel

field_description = lambda x: {'description': x}

@dataclass
class NextSequenceConfig:
    """
    Args:
        features (array of Feature): a list of Feature object containing feature configuration
        max_len (int): maximum length of input sequence
        emb_dim (int): an output dimension of embedding model
        fusion_dim (int): an output dimension of fusion model
        fusion_strategy (FusionStrategy): Fusion strategy will dictate how we combine all features into one
        rope_strategy (RoPEStrategy): Rope streategy will determine which rope strategy will be used for positional embedding
        attn_heads (int): a number of attention heads
        num_layers (int): a number of attention block
        expansion (int): an expansion will be used in EncoderBlock for feature extraction
        dropout (float): a dropout will reduce overfitting in EncoderBlock by randomly drop some weights out
    """
    features:list[Feature]
    max_len:int
    emb_dim:int = 128
    fusion_dim:int = 128
    fusion_strategy:FusionStrategy = FusionStrategy.SUM
    rope_strategy:RoPEStrategy = RoPEStrategy.INTERLEAVED
    attn_heads:int = 2
    num_layers:int = 2
    expansion:int = 2
    dropout:float = 0.1

    def __post_init__(self):
        self.fusion_strategy = FusionStrategy(self.fusion_strategy)
        self.rope_strategy = RoPEStrategy(self.rope_strategy)


class NextSequenceModel(nn.Module):
    def __init__(self, configs:NextSequenceConfig):
        super().__init__()
        self.configs = configs
        self.embedding = EmbeddingModel(configs.features, configs.emb_dim)
        self.fusion = FUSION_REGISTRY[configs.fusion_strategy](configs.features, configs.fusion_dim)
        self.prepend_cls = PrependCLS(configs.fusion_dim)
        self.rope = RoPE(configs.fusion_dim//configs.attn_heads, configs.max_len+1)
        self.backbone = EncoderCLSBackbone(self.embedding, self.fusion, self.prepend_cls, self.rope, configs.attn_heads, configs.num_layers, configs.expansion, configs.dropout)
        self.mlh = MultiTaskHeads(configs.features, configs.fusion_dim)
        self.model = MultiTaskModel(self.backbone, self.mlh)

    def forward(self, X:dict[str, torch.Tensor], pad_mask:torch.Tensor|None):
        return self.model(X, pad_mask)