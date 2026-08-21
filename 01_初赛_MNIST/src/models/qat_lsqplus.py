"""LSQ+ QAT photonic MLP model.

References:
    - src_dsqlsq/train_lsq_plus.py (lines 134-175)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantization.lsqplus import LSQPlusQuantizer


class PhotonicMLP_LSQPlus(nn.Module):
    """3-layer MLP with LSQ+ learnable quantization."""

    def __init__(self, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim1, bias=False)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.fc3 = nn.Linear(hidden_dim2, 10, bias=False)

        self.input_quantizer = LSQPlusQuantizer(4, False, 1.0 / 15.0, 0.0)
        self.input_quantizer.scale.requires_grad = False

        self.w1_quantizer = LSQPlusQuantizer(4, True)
        self.h1_quantizer = LSQPlusQuantizer(4, False)
        self.w2_quantizer = LSQPlusQuantizer(4, True)
        self.h2_quantizer = LSQPlusQuantizer(4, False)
        self.w3_quantizer = LSQPlusQuantizer(4, True)

    def forward(self, x):
        x = x.view(-1, 784)

        x_q = self.input_quantizer(x)
        w1_q = self.w1_quantizer(self.fc1.weight)
        y1 = F.linear(x_q, w1_q)
        h1 = torch.relu(y1)
        h1_q = self.h1_quantizer(h1)

        w2_q = self.w2_quantizer(self.fc2.weight)
        y2 = F.linear(h1_q, w2_q)
        h2 = torch.relu(y2)
        h2_q = self.h2_quantizer(h2)

        w3_q = self.w3_quantizer(self.fc3.weight)
        out = F.linear(h2_q, w3_q)
        return out

    def get_quantization_info(self):
        return {
            "input_scale": abs(self.input_quantizer.scale.item()),
            "input_zp": self.input_quantizer.zero_point.item(),
            "w1_scale": abs(self.w1_quantizer.scale.item()),
            "w1_zp": self.w1_quantizer.zero_point.item(),
            "h1_scale": abs(self.h1_quantizer.scale.item()),
            "h1_zp": self.h1_quantizer.zero_point.item(),
            "w2_scale": abs(self.w2_quantizer.scale.item()),
            "w2_zp": self.w2_quantizer.zero_point.item(),
            "h2_scale": abs(self.h2_quantizer.scale.item()),
            "h2_zp": self.h2_quantizer.zero_point.item(),
            "w3_scale": abs(self.w3_quantizer.scale.item()),
            "w3_zp": self.w3_quantizer.zero_point.item(),
        }
