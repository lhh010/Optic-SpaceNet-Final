"""DSQ QAT photonic MLP model.

References:
    - src_dsqlsq/train_dsq.py (lines 126-162)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantization.dsq import DSQQuantizer


class PhotonicMLP_DSQ(nn.Module):
    """3-layer MLP with DSQ soft quantization and temperature annealing."""

    def __init__(self, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim1, bias=False)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.fc3 = nn.Linear(hidden_dim2, 10, bias=False)

        self.input_quantizer = DSQQuantizer(4, False, is_input=True)
        self.w1_quantizer = DSQQuantizer(4, True)
        self.h1_quantizer = DSQQuantizer(4, False)
        self.w2_quantizer = DSQQuantizer(4, True)
        self.h2_quantizer = DSQQuantizer(4, False)
        self.w3_quantizer = DSQQuantizer(4, True)

    def set_temperature(self, temp: float):
        for m in self.modules():
            if isinstance(m, DSQQuantizer):
                m.temp = temp

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
            "w1_scale": abs(self.w1_quantizer.scale.item()),
            "h1_scale": abs(self.h1_quantizer.scale.item()),
            "w2_scale": abs(self.w2_quantizer.scale.item()),
            "h2_scale": abs(self.h2_quantizer.scale.item()),
            "w3_scale": abs(self.w3_quantizer.scale.item()),
        }
