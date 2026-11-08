import torch.nn as nn
from next_sequence.multi_task_head import MultiTaskHeads
from next_sequence.encoder_block import EncoderCLSBackbone

class MultiTaskModel(nn.Module):
    def __init__(self, backbone:EncoderCLSBackbone, heads:MultiTaskHeads):
        super().__init__()
        self.backbone, self.heads = backbone, heads

    def forward(self, X, pad_mask):
        return self.heads(self.backbone(X, pad_mask))
