"""
===============================================================================
 qat_v5.py — 硬件对齐 QAT (修复 v4 三处关键缺陷)
===============================================================================
 依据: osimulator/GAZELLE_ARCHITECTURE.md + gazelle_artifacts 逆向标定

 与 v4 的差异 (v4 缺陷 → v5 修复):
   1. 激活噪声死代码: v4 定义了 inject_activation_noise 但从未调用
      → v5 在 GEMM 输出端注入 TIA 噪声 (σ 标定自 random_benchmark:
        delta_std/ideal_std = 0.0457)
   2. 激活量化不对齐: v4 用 per-channel signed int8
      → v5 用 per-tensor unsigned affine uint8 + zero_point (与
        osimulator _matmul_real 的 quantize_to_int(signed=False) 一致)
   3. 无输出量化: v4 不做 12-bit ADC 量化
      → v5 输出做 12-bit 量化 (匹配硬件 output_precision=12)

 核心: 前向完全复刻 osimulator 整数域计算链
   x → uint8 affine quant (per-tensor, zp) → 整数 conv
   w → int8 per-channel quant → 整数 conv
   y_int = conv(x_int, w_int)
   y = x_scale * w_scale_j * y_int + x_zp * w_scale_j * col_sum_w_j   (反量化)
   y += TIA 噪声 (训练时) → 12-bit ADC 量化
 训练时 STE 保持梯度; 推理时与 osimulator 路径数学一致。
===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ============================================================
# 量化原语 (STE)
# ============================================================

def quant_uint8_affine(x, return_params=False):
    """per-tensor unsigned affine → uint8 + zero_point (同 osimulator)。"""
    x_min = x.min()
    x_max = x.max()
    span = (x_max - x_min).clamp(min=1e-8)
    scale = span / 255.0
    zp = torch.round(-x_min / scale).clamp(0, 255)
    x_int = (x / scale + zp).round().clamp(0, 255)
    x_dq = (x_int - zp) * scale
    out = x + (x_dq - x).detach()
    if return_params:
        return out, scale, zp
    return out


def quant_symmetric(x, bits=8, per_channel=False, ch_dim=1):
    """对称量化 (v4 兼容: per-channel signed)。"""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        reduce_dims = [i for i in range(x.dim()) if i != ch_dim]
        amax = x.abs()
        for d in sorted(reduce_dims, reverse=True):
            amax = amax.max(dim=d, keepdim=True)[0]
        scale = (amax / qmax).clamp(min=1e-8)
    else:
        scale = (x.abs().max() / qmax).clamp(min=1e-8)
    x_int = (x / scale).round().clamp(-qmax, qmax)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


def quant_int8_per_channel(w, ch_dim=0):
    """per-channel signed int8 (同 osimulator 权重路径)。"""
    reduce_dims = [i for i in range(w.dim()) if i != ch_dim]
    amax = w.abs()
    for d in sorted(reduce_dims, reverse=True):
        amax = amax.max(dim=d, keepdim=True)[0]
    scale = (amax / 127.0).clamp(min=1e-8)
    w_int = (w / scale).round().clamp(-127, 127)
    w_dq = w_int * scale
    return w + (w_dq - w).detach(), scale


def quant_output_12bit(y, per_channel=True):
    """12-bit ADC 输出量化 (匹配 output_precision=12)。"""
    qmax = 2047
    if per_channel:
        reduce_dims = [i for i in range(y.dim()) if i != 1]
        amax = y.abs()
        for d in sorted(reduce_dims, reverse=True):
            amax = amax.max(dim=d, keepdim=True)[0]
        scale = (amax / qmax).clamp(min=1e-8)
    else:
        scale = (y.abs().max() / qmax).clamp(min=1e-8)
    y_int = (y / scale).round().clamp(-qmax, qmax)
    y_dq = y_int * scale
    return y + (y_dq - y).detach()


def inject_output_noise(y, noise_ratio):
    """绝对加性底噪 (crossval 实测: hw σ_total 与信号幅度无关, uint8 ≈4.49 counts)。
    σ = noise_ratio × 输出 RMS (全局标量, 所有元素同一 σ) — 非 per-channel 相对噪声。"""
    rms = y.std()
    return y + torch.randn_like(y) * noise_ratio * rms


# ============================================================
# QAT 层
# ============================================================

class QATConv2d_v5(nn.Module):
    def __init__(self, conv_layer, weight_bits=8, output_noise=True,
                 output_noise_ratio=0.0457, output_quant=True,
                 weight_noise=False, weight_noise_ratio=0.0016,
                 activation_style="osim", keep_fp32=False):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.weight = nn.Parameter(conv_layer.weight.data.clone())
        self.bias = (nn.Parameter(conv_layer.bias.data.clone())
                     if conv_layer.bias is not None else None)
        self._qat_enabled = not keep_fp32
        self._weight_bits = weight_bits
        self._output_noise = output_noise
        self._output_noise_ratio = output_noise_ratio
        self._output_quant = output_quant
        self._weight_noise = weight_noise
        self._weight_noise_ratio = weight_noise_ratio
        self._act_style = activation_style
        self._keep_fp32 = keep_fp32

    @property
    def qat_enabled(self):
        return self._qat_enabled

    def enable_qat(self):
        if not self._keep_fp32:
            self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def _forward_fp32(self, x):
        return F.conv2d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def _forward_qat(self, x):
        # 1. 输入量化: osim=per-tensor unsigned affine | v4=per-channel signed
        #    STE: 前向取反量化浮点值 (x_dq), 反向直通
        if self._act_style == "osim":
            x_dq = quant_uint8_affine(x)
        else:
            x_dq = quant_symmetric(x, bits=8, per_channel=True, ch_dim=1)
        # 2. 权重: per-channel signed int8 (STE)
        w_dq, w_scale = quant_int8_per_channel(self.weight, ch_dim=0)
        if self._weight_noise and self.training:
            noise = torch.randn_like(w_dq) * self._weight_noise_ratio * w_scale.detach()
            w_dq = w_dq + noise
        # 3. 浮点域 conv: 数学上 ≡ osimulator 整数域 dequant (x_dq @ w_dq)
        y = F.conv2d(x_dq, w_dq, None,
                     self.stride, self.padding, self.dilation, self.groups)
        # 4. 绝对加性底噪 (crossval: hw 与信号幅度无关) — 训练时
        if self._output_noise and self.training:
            y = inject_output_noise(y, self._output_noise_ratio)
        # 5. ADC 12-bit 输出量化
        if self._output_quant:
            y = quant_output_12bit(y)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def forward(self, x):
        if not self._qat_enabled:
            return self._forward_fp32(x)
        return self._forward_qat(x)

    def extra_repr(self):
        return (f"{self.in_channels}->{self.out_channels}, k={self.kernel_size}, "
                f"w{self._weight_bits}/a8/o12, "
                f"out_noise={self._output_noise}(r={self._output_noise_ratio}), "
                f"out_quant={self._output_quant}, act={self._act_style}")


class QATLinear_v5(nn.Module):
    def __init__(self, linear_layer, weight_bits=8, output_noise=True,
                 output_noise_ratio=0.0457, output_quant=True,
                 weight_noise=False, weight_noise_ratio=0.0016,
                 activation_style="osim", is_last=False):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        self.bias = (nn.Parameter(linear_layer.bias.data.clone())
                     if linear_layer.bias is not None else None)
        self._qat_enabled = True
        self._weight_bits = weight_bits
        self._output_noise = output_noise
        self._output_noise_ratio = output_noise_ratio
        self._output_quant = output_quant
        self._weight_noise = weight_noise
        self._weight_noise_ratio = weight_noise_ratio
        self._act_style = activation_style
        self._is_last = is_last

    @property
    def qat_enabled(self):
        return self._qat_enabled

    def enable_qat(self):
        self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def forward(self, x):
        if not self._qat_enabled:
            return F.linear(x, self.weight, self.bias)
        if self._is_last:
            y = F.linear(x, self.weight, self.bias)
            return y
        if self._act_style == "osim":
            x_dq = quant_uint8_affine(x)
        else:
            x_dq = quant_symmetric(x, bits=8, per_channel=True, ch_dim=-1)
        w_dq, w_scale = quant_int8_per_channel(self.weight, ch_dim=0)
        if self._weight_noise and self.training:
            noise = torch.randn_like(w_dq) * self._weight_noise_ratio * w_scale.detach()
            w_dq = w_dq + noise
        y = F.linear(x_dq, w_dq, None)
        if self._output_noise and self.training:
            y = inject_output_noise(y, self._output_noise_ratio)
        if self._output_quant:
            y = quant_output_12bit(y)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return (f"{self.in_features}->{self.out_features}, w{self._weight_bits}, "
                f"last={self._is_last}")


# ============================================================
# 模型转换
# ============================================================

def prepare_model_v5(model, weight_bits=8, output_noise=True,
                     output_noise_ratio=0.0457, output_quant=True,
                     weight_noise=False, weight_noise_ratio=0.0016,
                     activation_style="osim",
                     quantize_linear=True, inplace=True):
    """将标准模型转换为 QAT v5 (全层光计算, 无 stem 特判)。"""
    if not inplace:
        model = copy.deepcopy(model)

    def _convert(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, QATConv2d_v5(
                    child, weight_bits=weight_bits,
                    output_noise=output_noise,
                    output_noise_ratio=output_noise_ratio,
                    output_quant=output_quant,
                    weight_noise=weight_noise,
                    weight_noise_ratio=weight_noise_ratio,
                    activation_style=activation_style,
                    keep_fp32=False))
            elif isinstance(child, nn.Linear) and quantize_linear:
                setattr(module, name, QATLinear_v5(
                    child, weight_bits=weight_bits,
                    output_noise=output_noise,
                    output_noise_ratio=output_noise_ratio,
                    output_quant=output_quant,
                    weight_noise=weight_noise,
                    weight_noise_ratio=weight_noise_ratio,
                    activation_style=activation_style))
            elif not isinstance(child, (QATConv2d_v5, QATLinear_v5)):
                _convert(child)

    _convert(model)
    n_qat = sum(1 for m in model.modules()
                if isinstance(m, (QATConv2d_v5, QATLinear_v5)))
    print(f"[prepare_model_v5] HW-aligned QAT: w{weight_bits}/a8/o12, "
          f"out_noise={output_noise}(r={output_noise_ratio}), "
          f"out_quant={output_quant}, act={activation_style}, "
          f"{n_qat} QAT layers")
    return model


def enable_qat(model):
    for m in model.modules():
        if isinstance(m, (QATConv2d_v5, QATLinear_v5)):
            m.enable_qat()


def disable_qat(model):
    for m in model.modules():
        if isinstance(m, (QATConv2d_v5, QATLinear_v5)):
            m.disable_qat()
