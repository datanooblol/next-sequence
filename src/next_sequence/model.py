"""Model configuration and the ``NextSequenceModel`` network."""
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
    """Architecture of a ``NextSequenceModel``. All values are validated on creation.

    Args:
        features (list[Feature]): Input features and what to predict from each. At least one
            feature must not be PASSTHROUGH, and names must be unique.
        max_len (int): Longest input sequence the model accepts, without the CLS token. Use the
            same value in ``CustomCollate(max_len=...)`` so longer histories are truncated.
        emb_dim (int): Width of each feature embedding.
        fusion_dim (int | None): Width after combining the feature embeddings; this is the width
            of the transformer. ``None`` uses ``emb_dim``.
        fusion_strategy (FusionStrategy | str): How embeddings are combined: ``"sum"`` or ``"concat"``.
        rope_strategy (RoPEStrategy | str): How rotary position embeddings pair dimensions.
        attn_heads (int): Attention heads per block. ``fusion_dim / attn_heads`` must be an even integer.
        num_layers (int): Number of transformer encoder blocks.
        expansion (int): Feed-forward hidden width is ``expansion * fusion_dim``.
        dropout (float): Dropout inside the encoder blocks, in ``[0, 1)``.
        recent_first (bool): Present each history to the model newest event first, so the latest
            event always sits right next to the CLS token. This makes "what happened last" easy to
            learn and usually improves accuracy a lot. ``CustomCollate.from_config`` applies it.

    Raises:
        ValueError: If any value is invalid.
        TypeError: If ``features`` contains something other than ``Feature`` objects.

    Example:
        >>> config = NextSequenceConfig(
        ...     features=[Feature("item", "classification", vocab_size=100),
        ...               Feature("price", "regression", is_linear=True)],
        ...     max_len=50, emb_dim=64, attn_heads=4)
        >>> config.to_dict()["fusion_strategy"]
        'sum'
    """
    features: list[Feature]
    max_len: int
    emb_dim: int = 128
    fusion_dim: int | None = None
    fusion_strategy: FusionStrategy | str = FusionStrategy.SUM
    rope_strategy: RoPEStrategy | str = RoPEStrategy.INTERLEAVED
    attn_heads: int = 2
    num_layers: int = 2
    expansion: int = 2
    dropout: float = 0.1
    recent_first: bool = True

    SCHEMA_VERSION: ClassVar[int] = 2
    """Version of the ``to_dict`` format, stored in checkpoints."""

    def __post_init__(self) -> None:
        self.fusion_strategy = FusionStrategy(self.fusion_strategy)
        self.rope_strategy = RoPEStrategy(self.rope_strategy)
        if self.fusion_dim is None:
            self.fusion_dim = self.emb_dim
        self._validate()

    @property
    def model_dim(self) -> int:
        """Width of the transformer (the resolved ``fusion_dim``)."""
        return self.fusion_dim if self.fusion_dim is not None else self.emb_dim

    def _validate(self) -> None:
        """Check every field; raise ``ValueError`` / ``TypeError`` with a clear message."""
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
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
        if not isinstance(self.recent_first, bool):
            raise ValueError(f"recent_first must be True or False, got {self.recent_first!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        d = self.model_dim
        if d % self.attn_heads:
            raise ValueError(f"fusion_dim ({d}) must be divisible by attn_heads ({self.attn_heads})")
        if (d // self.attn_heads) % 2:
            raise ValueError(f"fusion_dim / attn_heads = {d // self.attn_heads} must be even (RoPE)")

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (no enums or dataclasses), safe to save as JSON or in a checkpoint.

        Returns:
            dict[str, Any]: All fields plus ``schema_version``.
        """
        return dict(
            schema_version=self.SCHEMA_VERSION,
            features=[f.to_dict() for f in self.features],
            max_len=self.max_len,
            emb_dim=self.emb_dim,
            fusion_dim=self.model_dim,
            fusion_strategy=FusionStrategy(self.fusion_strategy).value,
            rope_strategy=RoPEStrategy(self.rope_strategy).value,
            attn_heads=self.attn_heads,
            num_layers=self.num_layers,
            expansion=self.expansion,
            dropout=self.dropout,
            recent_first=self.recent_first,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NextSequenceConfig":
        """Rebuild a config from ``to_dict`` output.

        Args:
            data (dict[str, Any]): Output of ``to_dict``.

        Returns:
            NextSequenceConfig: The rebuilt, validated config.

        Raises:
            ValueError: If ``data`` was written by a newer version of the library.
        """
        data = dict(data)
        version = data.pop("schema_version", cls.SCHEMA_VERSION)
        if version > cls.SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version {version} is newer than this library supports "
                f"({cls.SCHEMA_VERSION}). Upgrade next-sequence."
            )
        if version < 2:
            data.setdefault("recent_first", False)   # schema 1 models were trained oldest-first
        data["features"] = [Feature.from_dict(f) for f in data["features"]]
        return cls(**data)


class NextSequenceModel(nn.Module):
    """Embedding -> fusion -> CLS + transformer encoder -> one head per predicted feature.

    A plain ``nn.Module`` with only ``forward``: training lives in ``NextSequence`` and
    ``next_sequence.train``, so the model can also be used in any custom training loop.

    Args:
        configs (NextSequenceConfig): Architecture and features.

    Example:
        >>> model = NextSequenceModel(config)
        >>> out = model(batch["X"], batch["mask"])
        >>> out["item"].shape        # (B, n_classes) logits
    """

    def __init__(self, configs: NextSequenceConfig):
        super().__init__()
        self.configs = configs
        # parts are built as locals and owned by `backbone`, so each weight is registered once
        embedding = EmbeddingModel(configs.features, configs.emb_dim)
        fusion = FUSION_REGISTRY[FusionStrategy(configs.fusion_strategy)](
            configs.features, embedding.d_model, configs.model_dim
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
        """Predict the next event for every sequence in the batch.

        Args:
            X (dict[str, torch.Tensor]): One tensor per feature: categorical ``(B, T)`` token ids,
                numeric ``(B, T, 1)``. Usually ``batch["X"]`` from ``CustomCollate``.
            pad_mask (torch.Tensor | None): Bool ``(B, T)``, True = padding. ``None`` means no padding.

        Returns:
            dict[str, torch.Tensor]: Raw outputs per predicted feature: classification
            ``(B, n_classes)`` logits, regression ``(B, 1)``.
        """
        return self.heads(self.backbone(X, pad_mask))
