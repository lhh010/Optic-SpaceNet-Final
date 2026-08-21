"""STE-based QAT photonic MLP model.

References:
    - src_raw/train_and_quantize.py (lines 32-80)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantization.ste import FakeQuantizeSTE


class PhotonicMLP_STEQ(nn.Module):
    """3-layer MLP with inline STE fake-quantization and optional hardware noise injection."""

    def __init__(self, hidden_dim1: int = 128, hidden_dim2: int = 64, noise_std: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim1, bias=False)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.fc3 = nn.Linear(hidden_dim2, 10, bias=False)
        self.noise_std = noise_std

    def quantize(self, x, num_bits, is_signed, scale_factor=None):
        if scale_factor is None:
            max_val = x.abs().max().item()
            scale_factor = max_val / (
                (2 ** (num_bits - 1) - 1) if is_signed else (2**num_bits - 1)
            )
            scale_factor = max(scale_factor, 1e-8)

        return FakeQuantizeSTE.apply(x, num_bits, is_signed, scale_factor), scale_factor

    def forward(self, x):
        x = x.view(-1, 784)

        x_q, _ = self.quantize(x, num_bits=4, is_signed=False, scale_factor=1.0 / 15.0)

        w1_q, scale_w1 = self.quantize(self.fc1.weight, num_bits=4, is_signed=True)
        if self.training:
            w1_q = w1_q + torch.randn_like(w1_q) * self.noise_std * scale_w1

        y1 = F.linear(x_q, w1_q)
        y1_q, _ = self.quantize(y1, num_bits=8, is_signed=True)
        h1 = torch.relu(y1_q)
        h1_q, _ = self.quantize(h1, num_bits=4, is_signed=False)

        w2_q, scale_w2 = self.quantize(self.fc2.weight, num_bits=4, is_signed=True)
        if self.training:
            w2_q = w2_q + torch.randn_like(w2_q) * self.noise_std * scale_w2

        y2 = F.linear(h1_q, w2_q)
        y2_q, _ = self.quantize(y2, num_bits=8, is_signed=True)
        h2 = torch.relu(y2_q)
        h2_q, _ = self.quantize(h2, num_bits=4, is_signed=False)

        w3_q, scale_w3 = self.quantize(self.fc3.weight, num_bits=4, is_signed=True)
        if self.training:
            w3_q = w3_q + torch.randn_like(w3_q) * self.noise_std * scale_w3

        out = F.linear(h2_q, w3_q)
        out_q, _ = self.quantize(out, num_bits=8, is_signed=True)

        return out_q

    def get_quantization_info(self):
        """Return scale factors needed for numpy inference."""
        return {
            "input_scale": 1.0 / 15.0,
            "w1_scale": None,  # computed post-hoc in export
            "h1_scale": None,
            "w2_scale": None,
            "h2_scale": None,
            "w3_scale": None,
        }
