from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from next_sequence.core import Feature, TaskType
from next_sequence.embedding import EmbeddingModel
from next_sequence.encoder_block import EncoderCLSBackbone
from next_sequence.fusion_registry import FUSION_REGISTRY, FusionStrategy
from next_sequence.multi_task_head import MultiTaskHeads
from next_sequence.position_embedding import RoPE, RoPEStrategy
from next_sequence.prepend_cls import PrependCLS


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
    features: list[Feature]
    max_len: int
    emb_dim: int = 128
    fusion_dim: int | None = None
    fusion_strategy: FusionStrategy = FusionStrategy.SUM
    rope_strategy: RoPEStrategy = RoPEStrategy.INTERLEAVED
    attn_heads: int = 2
    num_layers: int = 2
    expansion: int = 2
    dropout: float = 0.1

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self):
        self.fusion_strategy = FusionStrategy(self.fusion_strategy)
        self.rope_strategy = RoPEStrategy(self.rope_strategy)
        if self.fusion_dim is None:
            self.fusion_dim = self.emb_dim
        self._validate()

    def _validate(self) -> None:
        if not self.features:
            raise ValueError("features must not be empty")
        bad = [f for f in self.features if not isinstance(f, Feature)]
        if bad:
            raise TypeError(f"features must be Feature objects, got {type(bad[0]).__name__}")
        names = [f.name for f in self.features]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate feature names: {dupes}")
        if all(f.task == TaskType.PASSTHROUGH for f in self.features):
            raise ValueError("at least one feature must be predicted (not PASSTHROUGH)")
        for field_name in ("max_len", "emb_dim", "fusion_dim", "attn_heads", "num_layers", "expansion"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.fusion_dim % self.attn_heads:
            raise ValueError(
                f"fusion_dim ({self.fusion_dim}) must be divisible by attn_heads ({self.attn_heads})"
            )
        if (self.fusion_dim // self.attn_heads) % 2:
            raise ValueError(
                f"fusion_dim / attn_heads = {self.fusion_dim // self.attn_heads} must be even (RoPE)"
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (no enums or dataclasses), safe to save."""
        return dict(
            schema_version=self.SCHEMA_VERSION,
            features=[f.to_dict() for f in self.features],
            max_len=self.max_len,
            emb_dim=self.emb_dim,
            fusion_dim=self.fusion_dim,
            fusion_strategy=self.fusion_strategy.value,
            rope_strategy=self.rope_strategy.value,
            attn_heads=self.attn_heads,
            num_layers=self.num_layers,
            expansion=self.expansion,
            dropout=self.dropout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NextSequenceConfig":
        data = dict(data)
        version = data.pop("schema_version", cls.SCHEMA_VERSION)
        if version > cls.SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version {version} is newer than this library supports "
                f"({cls.SCHEMA_VERSION}). Upgrade next-sequence."
            )
        data["features"] = [Feature.from_dict(f) for f in data["features"]]
        return cls(**data)


class NextSequenceModel(nn.Module):
    """Embedding -> fusion -> CLS + transformer encoder -> one head per predicted feature.

    ``forward`` returns raw outputs: classification ``(B, C)`` logits, regression ``(B, 1)``.
    """

    def __init__(self, configs: NextSequenceConfig):
        super().__init__()
        self.configs = configs
        # parts are built as locals and owned by `backbone`, so each weight is registered once
        embedding = EmbeddingModel(configs.features, configs.emb_dim)
        fusion = FUSION_REGISTRY[configs.fusion_strategy](
            configs.features, embedding.d_model, configs.fusion_dim
        )
        rope = RoPE(
            head_dim=fusion.d_out // configs.attn_heads,
            max_len=configs.max_len + 1,                 # +1 for the CLS token
            strategy=configs.rope_strategy,
        )
        self.backbone = EncoderCLSBackbone(
            embedding, fusion, PrependCLS(fusion.d_out), rope,
            n_heads=configs.attn_heads,
            num_layers=configs.num_layers,
            expansion=configs.expansion,
            dropout=configs.dropout,
        )
        self.heads = MultiTaskHeads(configs.features, fusion.d_out)

    def forward(
        self, X: dict[str, torch.Tensor], pad_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        return self.heads(self.backbone(X, pad_mask))
