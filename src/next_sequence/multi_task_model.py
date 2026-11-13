# torch
import torch
import torch.nn as nn
from next_sequence.multi_task_head import MultiTaskHeads
from next_sequence.encoder_block import EncoderCLSBackbone
from typing import Any


class MultiTaskModel(nn.Module):
    def __init__(self, backbone:EncoderCLSBackbone, heads:MultiTaskHeads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, X:dict[str, Any], pad_mask:torch.Tensor|None=None)->dict[str, Any]:
        return self.heads(self.backbone(X, pad_mask))
