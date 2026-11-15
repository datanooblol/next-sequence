"""Per-feature input embeddings."""
import torch
import torch.nn as nn

from next_sequence.core import Feature


class EmbeddingModel(nn.Module):
    """Embed every feature into a vector of width ``d_model``.

    Categorical inputs (``is_linear=False``) use ``nn.Embedding`` with ``padding_idx``; numeric
    inputs (``is_linear=True``) use ``nn.Linear(1, d_model)``.

    Args:
        features (list[Feature]): Features to embed, in a fixed order.
        d_model (int): Width of each feature embedding.
        pad_id (int): Token id used for padding in categorical inputs.
    """

    def __init__(self, features: list[Feature], d_model: int, pad_id: int = 0):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.embedder = nn.ModuleDict({
            feature.name: self.get_embed_model(feature, pad_id) for feature in features
        })

    def get_embed_model(self, feature: Feature, pad_id: int) -> nn.Module:
        """Build the embedding layer for one feature.

        Args:
            feature (Feature): The feature to embed.
            pad_id (int): Padding token id for categorical inputs.

        Returns:
            nn.Module: ``nn.Linear(1, d_model)`` or ``nn.Embedding(vocab_size, d_model)``.
        """
        if feature.is_linear:
            return nn.Linear(1, self.d_model)
        return nn.Embedding(feature.vocab_size, self.d_model, padding_idx=pad_id)

    def forward(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Embed each feature of a batch.

        Args:
            X (dict[str, torch.Tensor]): Categorical inputs ``(B, T)`` of token ids and numeric
                inputs ``(B, T, 1)``. Dtypes are cast as needed (``long`` / ``float``).

        Returns:
            dict[str, torch.Tensor]: One ``(B, T, d_model)`` tensor per feature.
        """
        out: dict[str, torch.Tensor] = {}
        for feature in self.features:
            x = X[feature.name]
            # cast here so a collate function that returns the "wrong" dtype can't crash the layer
            x = x.float() if feature.is_linear else x.long()
            out[feature.name] = self.embedder[feature.name](x)
        return out
