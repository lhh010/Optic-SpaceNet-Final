"""
===============================================================================
 qat_v6.py — 真机标定 per-layer 噪声 QAT (C2 迭代)
===============================================================================
 依据: gazelle-crossval CROSSVAL_REPORT.md + round4 calib_j1_real.json 实测

 与 v5 的差异 (v5 缺陷 → v6 修复):
   1. 噪声结构: v5 inject_output_noise 的 σ = ratio × 当前 batch y.std()
      —— 实为信号相关相对噪声, 与 crossval "真机=绝对加性底噪" 结论冲突
      → v6 在硬件原生 raw MAC 域注入绝对噪声: eps ~ N(0, sigma_raw),
        经当前 batch 的 x_scale × w_scale 换算到浮点域。
        sigma_raw 为每层标定常数 (calib_j1_real.json 的 resid_std),
        不随信号幅度变化; 权重更新时 x_scale/w_scale 自动跟踪。
   2. 噪声量级: v5 单一全局 ratio 0.0392 (uint8 单层口径)
      → v6 per-layer sigma_raw (s1a..s3b 实测 684..1619, 随 k/深度增长)
   3. stem: v5 无特判全层 QAT, 但部署 stem 是电计算 float
      → v6 stem_fp32=True (keep_fp32, 训练/部署一致)
   4. head: v5 h1/h2 也 QAT (is_last 从未设置, logits 层吃噪声)
      → v6 head_fp32=True, Linear 不转换 (部署 HEAD_ELEC 电计算)

 光计算层仅剩 stage1-3 的 5 个 1x1 conv。
===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qat_v5 import (QATConv2d_v5, quant_uint8_affine, quant_symmetric,
                    quant_int8_per_channel, quant_output_12bit)


class QATConv2d_v6(QATConv2d_v5):
    """v5 + per-layer raw 域绝对噪声 (sigma_raw, 单位 = MAC counts×255)。"""

    def __init__(self, conv_layer, sigma_raw=0.0, gain_jitter=0.0,
                 off_jitter_raw=0.0, **kwargs):
        super().__init__(conv_layer, **kwargs)
        self._sigma_raw = float(sigma_raw)
        self._gain_jitter = float(gain_jitter)
        self._off_jitter_raw = float(off_jitter_raw)

    def _forward_qat(self, x):
        # 1. 输入量化 (osim 风格), 保留 scale 供 raw 域噪声换算
        if self._act_style == "osim":
            x_dq, x_scale, _zp = quant_uint8_affine(x, return_params=True)
        else:
            x_dq = quant_symmetric(x, bits=8, per_channel=True, ch_dim=1)
            x_scale = None
        # 2. 权重 per-channel int8
        w_dq, w_scale = quant_int8_per_channel(self.weight, ch_dim=0)
        if self._weight_noise and self.training:
            noise = torch.randn_like(w_dq) * self._weight_noise_ratio * w_scale.detach()
            w_dq = w_dq + noise
        # 3. 浮点域 conv (≡ 整数域 dequant)
        y = F.conv2d(x_dq, w_dq, None,
                     self.stride, self.padding, self.dilation, self.groups)
        # 4. 噪声 (真机标定), 训练时:
        #    a. 增益抖动 (慢漂 alpha): y *= (1+eps_a), per-layer scalar/batch
        #    b. raw 域绝对噪声: iid 快噪声 sigma_raw + 共模偏移抖动 off_jitter_raw
        #       (crossval: sigma_static 慢漂主导 — 共模/结构化, 非 iid)
        if self.training:
            if self._gain_jitter > 0:
                y = y * (1.0 + torch.randn((), device=y.device) * self._gain_jitter)
            if self._output_noise and (self._sigma_raw > 0 or self._off_jitter_raw > 0):
                eps = torch.randn_like(y) * self._sigma_raw \
                    + torch.randn((), device=y.device) * self._off_jitter_raw
                if x_scale is not None:
                    eps = eps * x_scale.detach() * w_scale.detach().view(1, -1, 1, 1)
                else:
                    # 非 osim 路径无法换算 raw→float, 退化为 v5 相对噪声
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
                f"sigma_raw={self._sigma_raw:.1f}, fp32={self._keep_fp32}, "
                f"act={self._act_style}")


def prepare_model_v6(model, layer_sigmas=None, stem_fp32=True, head_fp32=True,
                     gain_jitter=0.0, off_jitter_raw=0.0,
                     weight_bits=8, output_noise=True, output_quant=True,
                     weight_noise=False, weight_noise_ratio=0.0016,
                     activation_style="osim"):
    """转换为 QAT v6。

    layer_sigmas: {dotted_module_path: sigma_raw}, 如
      {"stage1.0": 683.5, "stage2.0": 995.0, ...} (calib_j1_real.json resid_std)
      — per-layer iid 快噪声 (raw 域)。全量 resid_std 含结构化分量,
        作 iid 注入会过悲观 (numpy 实证 84% vs hw 90.6%), 快噪声口径 ≈260。
    gain_jitter: 增益慢漂抖动 σ (乘性, 如 0.02)
    off_jitter_raw: 共模偏移抖动 σ (raw 域, 如 500)
    stem_fp32: stem.0 conv 保持 FP32 (部署为电计算)
    head_fp32: Linear 不转换 (部署为电计算 head)
    """
    layer_sigmas = layer_sigmas or {}
    converted, warned = [], []

    def _convert(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                is_stem = path == "stem.0"
                if path in layer_sigmas:
                    sigma = layer_sigmas[path]
                else:
                    sigma = 0.0
                    if not is_stem:
                        warned.append(path)
                setattr(module, name, QATConv2d_v6(
                    child, sigma_raw=sigma,
                    gain_jitter=gain_jitter, off_jitter_raw=off_jitter_raw,
                    weight_bits=weight_bits, output_noise=output_noise,
                    output_quant=output_quant,
                    weight_noise=weight_noise,
                    weight_noise_ratio=weight_noise_ratio,
                    activation_style=activation_style,
                    keep_fp32=(stem_fp32 and is_stem)))
                converted.append((path, sigma, stem_fp32 and is_stem))
            elif isinstance(child, nn.Linear):
                if not head_fp32:
                    raise NotImplementedError("v6 head 光计算路径未实现, 用 head_fp32=True")
            else:
                _convert(child, path)

    _convert(model)
    if warned:
        print(f"[prepare_model_v6] WARN: 以下 conv 无 sigma_raw 标定 (=0, 不注噪): {warned}")
    for path, sigma, fp32 in converted:
        print(f"[prepare_model_v6] {path}: sigma_raw={sigma:.1f} fp32={fp32}")
    print(f"[prepare_model_v6] {len(converted)} conv converted, "
          f"stem_fp32={stem_fp32}, head_fp32={head_fp32}")
    return model
