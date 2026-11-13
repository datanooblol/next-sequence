import torch
import torch.nn as nn

class PrependCLS(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, X:torch.Tensor, pad_mask:torch.Tensor):
        B = X.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), X], dim=1)
        cls_col = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        return x, torch.cat([cls_col, pad_mask], dim=1)