"""STE (Straight-Through Estimator) fake-quantization core.

References:
    - src_raw/train_and_quantize.py (lines 12-27)
    - src_raw/training.py (lines 9-24)
"""

import torch


class FakeQuantizeSTE(torch.autograd.Function):
    """PyTorch autograd Function implementing straight-through estimator for fake quantization."""

    @staticmethod
    def forward(ctx, x, num_bits, is_signed, scale):
        qmin = -(2 ** (num_bits - 1)) if is_signed else 0
        qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1

        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, qmin, qmax)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
