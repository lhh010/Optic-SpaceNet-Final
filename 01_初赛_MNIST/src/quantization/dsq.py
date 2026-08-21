"""DSQ (Differentiable Soft Quantization) fake-quantization core.

References:
    - src_dsqlsq/train_dsq.py (lines 21-121)
"""

import math
import torch
import torch.nn as nn


class DSQFakeQuantize(torch.autograd.Function):
    """Soft quantization using tanh to smooth fractional parts."""

    @staticmethod
    def forward(ctx, x, scale, num_bits, is_signed, temp):
        qmin = -(2 ** (num_bits - 1)) if is_signed else 0
        qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1

        scale = torch.clamp(scale.abs(), min=1e-6)

        x_scaled_unclamped = x / scale
        x_scaled = torch.clamp(x_scaled_unclamped, qmin, qmax)

        x_floor = torch.floor(x_scaled)
        x_frac = x_scaled - x_floor

        tanh_offset = math.tanh(0.5)
        soft_frac = torch.tanh(temp * (x_frac - 0.5)) / (2 * tanh_offset) + 0.5

        x_soft = x_floor + soft_frac
        x_q = x_soft * scale

        ctx.save_for_backward(x, x_scaled_unclamped, x_frac, scale)
        ctx.temp = temp
        ctx.qmin = qmin
        ctx.qmax = qmax
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, x_scaled_unclamped, x_frac, scale = ctx.saved_tensors
        temp = ctx.temp
        qmin, qmax = ctx.qmin, ctx.qmax

        mask_in = (x_scaled_unclamped >= qmin) & (x_scaled_unclamped <= qmax)

        tanh_val = torch.tanh(temp * (x_frac - 0.5))
        grad_soft_frac = temp * (1 - tanh_val**2) / (2 * math.tanh(0.5))
        grad_x = grad_output * grad_soft_frac * mask_in.float()

        mask_below = x_scaled_unclamped < qmin
        mask_above = x_scaled_unclamped > qmax
        grad_scale_val = torch.where(
            mask_in,
            torch.round(x_scaled_unclamped) - x_scaled_unclamped,
            torch.where(mask_below, torch.full_like(x, qmin), torch.full_like(x, qmax)),
        )
        grad_scale = (grad_output * grad_scale_val).sum()
        grad_scale = grad_scale / math.sqrt(x.numel() * qmax)

        return grad_x, grad_scale, None, None, None


class DSQQuantizer(nn.Module):
    """DSQ quantizer module with temperature annealing support."""

    def __init__(self, num_bits, is_signed, is_input=False):
        super().__init__()
        self.num_bits = num_bits
        self.is_signed = is_signed
        self.is_input = is_input

        if self.is_input:
            self.scale = nn.Parameter(torch.tensor(1.0 / 15.0))
            self.scale.requires_grad = False
            self.initialized = True
        else:
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.initialized = False

        self.temp = 1.0

    def forward(self, x):
        if not self.initialized and self.training:
            with torch.no_grad():
                mean_val = x.abs().mean()
                std_val = x.std()
                qmax = (2 ** (self.num_bits - 1) - 1) if self.is_signed else 2**self.num_bits - 1
                init_scale = max((mean_val + 2 * std_val).item() / qmax, 1e-5)
                self.scale.data.fill_(init_scale)
                self.initialized = True

        if not self.training:
            scale = torch.clamp(self.scale.abs(), min=1e-6)
            qmin = -(2 ** (self.num_bits - 1)) if self.is_signed else 0
            qmax = (2 ** (self.num_bits - 1) - 1) if self.is_signed else 2**self.num_bits - 1
            x_scaled = torch.clamp(x / scale, qmin, qmax)
            return torch.round(x_scaled) * scale

        return DSQFakeQuantize.apply(x, self.scale, self.num_bits, self.is_signed, self.temp)
