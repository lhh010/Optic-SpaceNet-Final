"""
===============================================================================
 qat_v8.py — probe 实测组分结构化噪声 QAT (C3 二轮)
===============================================================================
 依据: round5 §7 probe_dump 残差分解 (2026-08-08, 板上 5 层 10 万行 pairs)
   hw 残差 (全局 alpha/beta 回归后) 的可识别结构:
     1. per-column 常量偏移 off_c,  std = 264..872 raw  (4-23% 方差)
     2. per-column 增益 g_c,        std = 1.2%..2.5%    (1-3%)
     3. per-element 等效权重扰动 δW, rms = 3.7..7.2 counts (线性解释 21-50%)
     4. 残余 (非线性结构化 + iid 260): 低阶模型不可分解
   rep 实验: hw 误差 95.2% 为 run 间可复现结构化 → 1-3 全部按"整跑固定"
   建模, 训练时 per-batch 重采样 (domain randomization)。
   sim_noise_proxy_v8 (同组分 + iid 260) 复现 hw: proxy 89.24 vs hw 88.07±1.3。

 与 v7 的差异: v7 的 per-channel offset 用 sqrt(resid²-260²) 全量,
   比实测列偏移大 2-4 倍, 且没有 δW 分量 → proxy 显示反而略伤精度。
   v8 全部用 probe 实测口径, 新增 per-element δW (int8 count 域)。

 光计算层仍为 stage1-3 的 5 个 1x1 conv, stem/head FP32 电计算。
===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qat_v5 import quant_uint8_affine, quant_symmetric, quant_int8_per_channel, \
    quant_output_12bit
from qat_v6 import QATConv2d_v6


class QATConv2d_v8(QATConv2d_v6):
    """v6 iid 快噪声 + probe 实测组分结构化噪声 (per-batch 重采样)。"""

    def __init__(self, conv_layer, col_off_raw=0.0, col_gain=0.0,
                 dw_rms=0.0, **kwargs):
        super().__init__(conv_layer, **kwargs)
        self._col_off_raw = float(col_off_raw)
        self._col_gain = float(col_gain)
        self._dw_rms = float(dw_rms)

    def _forward_qat(self, x):
        # 1. 输入量化 (osim 风格), 保留 scale 供 raw 域噪声换算
        if self._act_style == "osim":
            x_dq, x_scale, _zp = quant_uint8_affine(x, return_params=True)
        else:
            x_dq = quant_symmetric(x, bits=8, per_channel=True, ch_dim=1)
            x_scale = None
        # 2. 权重 per-channel int8 (+ 训练时 per-element δW, count 域换算)
        w_dq, w_scale = quant_int8_per_channel(self.weight, ch_dim=0)
        if self.training and self._dw_rms > 0:
            dw = torch.randn_like(w_dq) * self._dw_rms * w_scale.detach().view(-1, 1, 1, 1)
            w_dq = w_dq + dw
        if self._weight_noise and self.training:
            noise = torch.randn_like(w_dq) * self._weight_noise_ratio * w_scale.detach()
            w_dq = w_dq + noise
        # 3. 浮点域 conv (≡ 整数域 dequant)
        y = F.conv2d(x_dq, w_dq, None,
                     self.stride, self.padding, self.dilation, self.groups)
        # 4. 结构化噪声 (训练时, per-batch 重采样):
        if self.training:
            C = y.shape[1]
            # a. per-column 增益 (乘性, float 域)
            if self._col_gain > 0:
                g = 1.0 + torch.randn(1, C, 1, 1, device=y.device) * self._col_gain
                y = y * g
            # b. raw 域: iid 快噪声 sigma_raw + per-column 常量偏移
            if self._output_noise and (self._sigma_raw > 0 or self._col_off_raw > 0):
                eps = torch.randn_like(y) * self._sigma_raw
                if self._col_off_raw > 0:
                    eps = eps + torch.randn(1, C, 1, 1, device=y.device) * self._col_off_raw
                if x_scale is not None:
                    eps = eps * x_scale.detach() * w_scale.detach().view(1, -1, 1, 1)
                else:
                    eps = torch.randn_like(y) * self._output_noise_ratio * y.std()
                y = y + eps
        # 5. ADC 12-bit 输出量化
        if self._output_quant:
            y = quant_output_12bit(y)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def extra_repr(self):
        return (f"{self.in_channels}->{self.out_channels}, k={self.kernel_size}, "
                f"iid={self._sigma_raw:.1f} col_off={self._col_off_raw:.1f} "
                f"col_gain={self._col_gain:.4f} dw={self._dw_rms:.2f}")


def prepare_model_v8(model, layer_sigmas=None, layer_col_off=None,
                     layer_col_gain=None, layer_dw_rms=None,
                     stem_fp32=True, head_fp32=True,
                     weight_bits=8, output_noise=True, output_quant=True,
                     weight_noise=False, weight_noise_ratio=0.0016,
                     activation_style="osim"):
    """转换为 QAT v8。

    layer_sigmas:   {path: iid 快噪声 raw}      (=260.9)
    layer_col_off:  {path: per-column 偏移 raw} (probe 实测 264..872)
    layer_col_gain: {path: per-column 增益 std} (probe 实测 0.012..0.025)
    layer_dw_rms:   {path: δW rms, int8 counts} (probe 实测 3.7..7.2)
    """
    layer_sigmas = layer_sigmas or {}
    layer_col_off = layer_col_off or {}
    layer_col_gain = layer_col_gain or {}
    layer_dw_rms = layer_dw_rms or {}
    converted, warned = [], []

    def _convert(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                is_stem = path == "stem.0"
                if path not in layer_sigmas and not is_stem:
                    warned.append(path)
                setattr(module, name, QATConv2d_v8(
                    child, sigma_raw=layer_sigmas.get(path, 0.0),
                    col_off_raw=layer_col_off.get(path, 0.0),
                    col_gain=layer_col_gain.get(path, 0.0),
                    dw_rms=layer_dw_rms.get(path, 0.0),
                    weight_bits=weight_bits, output_noise=output_noise,
                    output_quant=output_quant,
                    weight_noise=weight_noise,
                    weight_noise_ratio=weight_noise_ratio,
                    activation_style=activation_style,
                    keep_fp32=(stem_fp32 and is_stem)))
                converted.append((path, stem_fp32 and is_stem))
            elif isinstance(child, nn.Linear):
                if not head_fp32:
                    raise NotImplementedError("v8 head 光计算路径未实现, 用 head_fp32=True")
            else:
                _convert(child, path)

    _convert(model)
    if warned:
        print(f"[prepare_model_v8] WARN: 以下 conv 无噪声标定 (=0, 不注噪): {warned}")
    for path, fp32 in converted:
        print(f"[prepare_model_v8] {path}: iid={layer_sigmas.get(path, 0):.1f} "
              f"col_off={layer_col_off.get(path, 0):.1f} "
              f"col_gain={layer_col_gain.get(path, 0):.4f} "
              f"dw={layer_dw_rms.get(path, 0):.2f} fp32={fp32}")
    print(f"[prepare_model_v8] {len(converted)} conv converted")
    return model
