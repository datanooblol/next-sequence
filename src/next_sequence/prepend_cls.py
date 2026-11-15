"""Learnable CLS token placed in front of every sequence."""
import torch
import torch.nn as nn


class PrependCLS(nn.Module):
    """Prepend a learnable CLS token and extend the padding mask with a never-masked column.

    After the encoder, the CLS position summarizes the whole sequence and feeds the task heads.
    Because CLS is never masked, a fully padded sequence still produces finite attention scores.

    Args:
        d_model (int): Width of the CLS vector; must equal the fusion output width.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self, X: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend CLS to each sequence.

        Args:
            X (torch.Tensor): Sequence of shape ``(B, T, d_model)``.
            pad_mask (torch.Tensor | None): Bool ``(B, T)``, True = padding. ``None`` means no padding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The sequence ``(B, T + 1, d_model)`` and the mask
            ``(B, T + 1)`` whose first column (CLS) is False.
        """
        B, T = X.shape[:2]
        if pad_mask is None:
            pad_mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
        x = torch.cat([self.cls.expand(B, -1, -1), X], dim=1)
        cls_col = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        return x, torch.cat([cls_col, pad_mask], dim=1)
