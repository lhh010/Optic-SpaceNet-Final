"""
===============================================================================
 qat_v9.py — v8 + 随机非线性残差 (RFF) QAT (C3 三轮, 覆盖未建模结构化分量)
===============================================================================
 依据: round5 §8 probe_dump 分解 — 列偏移/列增益/δW 线性合计只解释 ~25-50%,
   残余 50-75% 是"确定性但非线性"的输入依赖函数 (rep 实验: 95.2% 可复现)。
   v8 只建模了可分解的线性部分; v9 增加随机傅里叶特征 (RFF) 非线性残差:
     eps_c(x) = A_c · cos(B x_norm + phi),  B~(d×k), A~(C×d), phi~(d,)
   每 batch 重采样 (B, A, phi) → 对"未知确定性非线性残差"的 domain randomization。
   x_norm: 该 batch 量化后输入 x_int/255 (0..1), k=C_in。
   幅度: 归一化到 probe 实测 after-linear 残差 std (RESID_AL, 含 iid 260 之外的
   非线性结构化分量), 再乘 rff_scale (默认 1.0)。

 其余组分 (列偏移/列增益/δW/iid 260) 与 v8 相同, 幅度同 probe 实测。
===============================================================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qat_v5 import quant_uint8_affine, quant_symmetric, quant_int8_per_channel, \
    quant_output_12bit
from qat_v8 import QATConv2d_v8


class QATConv2d_v9(QATConv2d_v8):
    """v8 + RFF 随机非线性输入依赖残差。"""

    def __init__(self, conv_layer, rff_std=0.0, rff_dim=64, rff_gamma=3.0, **kwargs):
        super().__init__(conv_layer, **kwargs)
        self._rff_std = float(rff_std)
        self._rff_dim = int(rff_dim)
        self._rff_gamma = float(rff_gamma)

    def _forward_qat(self, x):
        # 与 v8 相同的前半, 但在 conv 前保留量化输入供 RFF 使用
        if self._act_style == "osim":
            x_dq, x_scale, _zp = quant_uint8_affine(x, return_params=True)
        else:
            x_dq = quant_symmetric(x, bits=8, per_channel=True, ch_dim=1)
            x_scale = None
        w_dq, w_scale = quant_int8_per_channel(self.weight, ch_dim=0)
        if self.training and self._dw_rms > 0:
            dw = torch.randn_like(w_dq) * self._dw_rms * w_scale.detach().view(-1, 1, 1, 1)
            w_dq = w_dq + dw
        if self._weight_noise and self.training:
            noise = torch.randn_like(w_dq) * self._weight_noise_ratio * w_scale.detach()
            w_dq = w_dq + noise
        y = F.conv2d(x_dq, w_dq, None,
                     self.stride, self.padding, self.dilation, self.groups)
        if self.training:
            C = y.shape[1]
            if self._col_gain > 0:
                g = 1.0 + torch.randn(1, C, 1, 1, device=y.device) * self._col_gain
                y = y * g
            if self._output_noise and (self._sigma_raw > 0 or self._col_off_raw > 0):
                eps = torch.randn_like(y) * self._sigma_raw
                if self._col_off_raw > 0:
                    eps = eps + torch.randn(1, C, 1, 1, device=y.device) * self._col_off_raw
                if x_scale is not None:
                    eps = eps * x_scale.detach() * w_scale.detach().view(1, -1, 1, 1)
                else:
                    eps = torch.randn_like(y) * self._output_noise_ratio * y.std()
                y = y + eps
            # RFF 非线性残差: eps_c(x) = A cos(B xn + phi), 归一化到 rff_std (raw 域换算)
            if self._rff_std > 0 and x_scale is not None:
                Bn, Cc, H, W = y.shape
                k = x_dq.shape[1]
                xn = x_dq.detach().permute(0, 2, 3, 1).reshape(-1, k)  # (m, k)
                xn = xn / (xn.abs().amax(dim=1, keepdim=True) + 1e-6)
                d = self._rff_dim
                Br = torch.randn(k, d, device=y.device) * self._rff_gamma
                A = torch.randn(d, C, device=y.device) / math.sqrt(d)
                phi = torch.rand(d, device=y.device) * 2 * math.pi
                f = torch.cos(xn @ Br + phi) @ A                    # (m, C)
                f = f / (f.std() + 1e-6) * self._rff_std
                f = f.reshape(Bn, H, W, C).permute(0, 3, 1, 2)
                y = y + f * x_scale.detach() * w_scale.detach().view(1, -1, 1, 1)
        if self._output_quant:
            y = quant_output_12bit(y)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def extra_repr(self):
        return (f"{self.in_channels}->{self.out_channels}, k={self.kernel_size}, "
                f"iid={self._sigma_raw:.1f} col_off={self._col_off_raw:.1f} "
                f"dw={self._dw_rms:.2f} rff={self._rff_std:.1f}")


def prepare_model_v9(model, layer_sigmas=None, layer_col_off=None,
                     layer_col_gain=None, layer_dw_rms=None, layer_rff_std=None,
                     rff_dim=64, rff_gamma=3.0,
                     stem_fp32=True, head_fp32=True,
                     weight_bits=8, output_noise=True, output_quant=True,
                     weight_noise=False, weight_noise_ratio=0.0016,
                     activation_style="osim"):
    """转换为 QAT v9。layer_rff_std: {path: RFF 残差 std (raw 域)}。"""
    layer_sigmas = layer_sigmas or {}
    layer_col_off = layer_col_off or {}
    layer_col_gain = layer_col_gain or {}
    layer_dw_rms = layer_dw_rms or {}
    layer_rff_std = layer_rff_std or {}
    converted = []

    def _convert(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                is_stem = path == "stem.0"
                setattr(module, name, QATConv2d_v9(
                    child, sigma_raw=layer_sigmas.get(path, 0.0),
                    col_off_raw=layer_col_off.get(path, 0.0),
                    col_gain=layer_col_gain.get(path, 0.0),
                    dw_rms=layer_dw_rms.get(path, 0.0),
                    rff_std=layer_rff_std.get(path, 0.0),
                    rff_dim=rff_dim, rff_gamma=rff_gamma,
                    weight_bits=weight_bits, output_noise=output_noise,
                    output_quant=output_quant,
                    weight_noise=weight_noise,
                    weight_noise_ratio=weight_noise_ratio,
                    activation_style=activation_style,
                    keep_fp32=(stem_fp32 and is_stem)))
                converted.append(path)
            elif isinstance(child, nn.Linear):
                if not head_fp32:
                    raise NotImplementedError("v9 head 光计算路径未实现, 用 head_fp32=True")
            else:
                _convert(child, path)

    _convert(model)
    for path in converted:
        print(f"[prepare_model_v9] {path}: iid={layer_sigmas.get(path, 0):.1f} "
              f"col_off={layer_col_off.get(path, 0):.1f} dw={layer_dw_rms.get(path, 0):.2f} "
              f"rff={layer_rff_std.get(path, 0):.1f}")
    print(f"[prepare_model_v9] {len(converted)} conv converted")
    return model
