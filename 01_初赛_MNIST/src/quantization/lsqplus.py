"""LSQ+ (Learned Step Size Quantization Plus) fake-quantization core.

References:
    - src_dsqlsq/train_lsq_plus.py (lines 21-128)
"""

import math
import torch
import torch.nn as nn


class LSQPlusFakeQuantize(torch.autograd.Function):
    """Custom autograd Function with learnable scale and zero_point."""

    @staticmethod
    def forward(ctx, x, scale, zero_point, num_bits, is_signed):
        qmin = -(2 ** (num_bits - 1)) if is_signed else 0
        qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1

        scale = torch.clamp(torch.abs(scale), min=1e-5)

        x_scaled = x / scale
        x_int_unclamped = torch.round(x_scaled) + zero_point
        x_int_clamped = torch.clamp(x_int_unclamped, qmin, qmax)
        x_q = (x_int_clamped - zero_point) * scale

        ctx.save_for_backward(x, scale, zero_point)
        ctx.qmin = qmin
        ctx.qmax = qmax

        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax

        scale = torch.clamp(torch.abs(scale), min=1e-5)
        x_scaled = x / scale
        x_int_unclamped = torch.round(x_scaled) + zero_point

        mask_in = (x_int_unclamped >= qmin) & (x_int_unclamped <= qmax)
        mask_below = x_int_unclamped < qmin
        mask_above = x_int_unclamped > qmax

        grad_x = grad_output * mask_in.float()

        grad_scale_val = torch.where(
            mask_in,
            torch.round(x_scaled) - x_scaled,
            torch.where(mask_below, qmin - zero_point, qmax - zero_point),
        )
        grad_scale_factor = 1.0 / math.sqrt(x.numel() * qmax)
        grad_scale = (grad_output * grad_scale_val).sum() * grad_scale_factor

        grad_zp_val = torch.where(
            mask_in, torch.zeros_like(x), torch.full_like(x, -scale.item())
        )
        grad_zero_point = (grad_output * grad_zp_val).sum()

        return grad_x, grad_scale, grad_zero_point, None, None


class LSQPlusQuantizer(nn.Module):
    """nn.Module wrapping scale and zero_point as nn.Parameter."""

    def __init__(self, num_bits, is_signed, init_scale=None, init_zero_point=0.0):
        super().__init__()
        self.num_bits = num_bits
        self.is_signed = is_signed

        if init_scale is not None:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
            self.zero_point = nn.Parameter(torch.tensor(float(init_zero_point)))
            self.initialized = True
        else:
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.zero_point = nn.Parameter(torch.tensor(0.0))
            self.initialized = False

    def forward(self, x):
        if not self.initialized and self.training:
            with torch.no_grad():
                qmin = -(2 ** (self.num_bits - 1)) if self.is_signed else 0
                qmax = (2 ** (self.num_bits - 1) - 1) if self.is_signed else 2**self.num_bits - 1

                max_val = x.abs().mean() + 3 * x.std()
                init_scale = max(max_val.item() / qmax, 1e-5)
                self.scale.data.fill_(init_scale)

                if not self.is_signed:
                    self.zero_point.data.fill_(-x.min().item() / init_scale)
                else:
                    self.zero_point.data.fill_(0.0)

                self.initialized = True

        safe_scale = torch.clamp(torch.abs(self.scale), min=1e-5)
        return LSQPlusFakeQuantize.apply(
            x, safe_scale, self.zero_point, self.num_bits, self.is_signed
        )
