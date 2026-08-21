"""
===============================================================================
 qat_v7.py — 结构化噪声 QAT (C3 迭代: per-channel episode 噪声)
===============================================================================
 依据: round5 重复性实验 (2026-08-08)
   rep1/rep2 同权重同 calib 背靠背 500 样本: 预测一致率 95.6%,
   2-vote 无增益 (89.0% vs 单次 89.6/87.6), oracle 上界仅 90.4%;
   hw 稳定错误 48/500 中 33 个 FAKE 模型并不犯 —— 剩余 gap 主体是
   *可复现的结构化误差* (静态 LUT 残差 + 每列固定偏移), 不是 iid 快噪声。

 与 v6 的差异:
   v6 的 off_jitter_raw 是 per-batch *标量* 共模抖动 (整张图同一个数),
   没有 per-column 结构; gain_jitter 也是标量。
   → v7 把静态分量建模为 *per-channel (tile 列) 噪声*:
     a. 每通道偏移 off_c ~ N(0, sigma_static_l)   (raw 域, LUT 列偏置)
     b. 每通道增益 g_c  ~ N(1, gain_jitter_ch)     (列间增益不一致)
     采样频率: 每 batch 重采样 (domain randomization; 部署时偏移固定但
     未知, 训练目标是对任意固定偏移不变, per-batch 重采样是其标准近似)。
   sigma_static_l = sqrt(resid_std_l^2 - sigma_dyn^2), 来自板上探针
   calib_c2c_seg2.json (2026-08-08), sigma_dyn=260.9 (crossval 短窗快噪声)。

 光计算层仍为 stage1-3 的 5 个 1x1 conv, stem/head FP32 电计算。
===============================================================================
"""
import torch
import torch.nn.functional as F

from qat_v5 import quant_uint8_affine, quant_int8_per_channel, quant_output_12bit
from qat_v6 import QATConv2d_v6


class QATConv2d_v7(QATConv2d_v6):
    """v6 + per-channel 结构化噪声 (offset/gain 向量, raw 域)。"""

    def __init__(self, conv_layer, sigma_static=0.0, gain_jitter_ch=0.0, **kwargs):
        super().__init__(conv_layer, **kwargs)
        self._sigma_static = float(sigma_static)
        self._gain_jitter_ch = float(gain_jitter_ch)

    def _forward_qat(self, x):
        # 1. 输入量化 (osim 风格), 保留 scale 供 raw 域噪声换算
        if self._act_style == "osim":
            x_dq, x_scale, _zp = quant_uint8_affine(x, return_params=True)
        else:
            from qat_v5 import quant_symmetric
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
        if self.training:
            C = y.shape[1]
            # a. per-channel 增益 (列间不一致 + 慢漂), 乘性
            if self._gain_jitter_ch > 0:
                g = 1.0 + torch.randn(1, C, 1, 1, device=y.device) * self._gain_jitter_ch
                y = y * g
            elif self._gain_jitter > 0:
                y = y * (1.0 + torch.randn((), device=y.device) * self._gain_jitter)
            # b. raw 域噪声: iid 快噪声 sigma_raw + per-channel 静态偏移
            if self._output_noise and (self._sigma_raw > 0 or self._sigma_static > 0
                                       or self._off_jitter_raw > 0):
                eps = torch.randn_like(y) * self._sigma_raw
                if self._sigma_static > 0:
                    eps = eps + torch.randn(1, C, 1, 1, device=y.device) * self._sigma_static
                if self._off_jitter_raw > 0:
                    eps = eps + torch.randn((), device=y.device) * self._off_jitter_raw
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
                f"sigma_raw={self._sigma_raw:.1f} sigma_static={self._sigma_static:.1f} "
                f"gain_ch={self._gain_jitter_ch:.3f} fp32={self._keep_fp32}")


def prepare_model_v7(model, layer_sigmas=None, layer_static_sigmas=None,
                     stem_fp32=True, head_fp32=True,
                     gain_jitter_ch=0.0, gain_jitter=0.0, off_jitter_raw=0.0,
                     weight_bits=8, output_noise=True, output_quant=True,
                     weight_noise=False, weight_noise_ratio=0.0016,
                     activation_style="osim"):
    """转换为 QAT v7。

    layer_sigmas: {path: sigma_raw} iid 快噪声 (=260, crossval 短窗口径)
    layer_static_sigmas: {path: sigma_static} per-channel 静态偏移
      (= sqrt(resid_std^2 - 260.9^2), 板上探针实测)
    gain_jitter_ch: per-channel 增益抖动 σ (乘性, 如 0.02)
    其余同 v6。
    """
    import torch.nn as nn
    layer_sigmas = layer_sigmas or {}
    layer_static_sigmas = layer_static_sigmas or {}
    converted, warned = [], []

    def _convert(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                is_stem = path == "stem.0"
                sigma = layer_sigmas.get(path, 0.0)
                sigma_st = layer_static_sigmas.get(path, 0.0)
                if path not in layer_sigmas and not is_stem:
                    warned.append(path)
                setattr(module, name, QATConv2d_v7(
                    child, sigma_raw=sigma, sigma_static=sigma_st,
                    gain_jitter_ch=gain_jitter_ch,
                    gain_jitter=gain_jitter, off_jitter_raw=off_jitter_raw,
                    weight_bits=weight_bits, output_noise=output_noise,
                    output_quant=output_quant,
                    weight_noise=weight_noise,
                    weight_noise_ratio=weight_noise_ratio,
                    activation_style=activation_style,
                    keep_fp32=(stem_fp32 and is_stem)))
                converted.append((path, sigma, sigma_st, stem_fp32 and is_stem))
            elif isinstance(child, nn.Linear):
                if not head_fp32:
                    raise NotImplementedError("v7 head 光计算路径未实现, 用 head_fp32=True")
            else:
                _convert(child, path)

    _convert(model)
    if warned:
        print(f"[prepare_model_v7] WARN: 以下 conv 无噪声标定 (=0, 不注噪): {warned}")
    for path, sigma, sigma_st, fp32 in converted:
        print(f"[prepare_model_v7] {path}: sigma_raw={sigma:.1f} "
              f"sigma_static={sigma_st:.1f} fp32={fp32}")
    print(f"[prepare_model_v7] {len(converted)} conv converted, "
          f"stem_fp32={stem_fp32}, head_fp32={head_fp32}, "
          f"gain_jitter_ch={gain_jitter_ch}")
    return model
