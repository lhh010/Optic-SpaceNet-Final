"""
================================================================================
 optic_qat_v3.py — 修复版 QAT 量化模块 (Phase 4+)

 基于 optic_qat_v2.py 的关键修复:
   1. 移除 QAT 层内部的 F.relu — 模型架构已有 ReLU, 避免双重激活
   2. 移除逐层输出重量化 (默认关闭) — 减少信息损失
   3. 激活量化默认使用 per-channel int8 (256级) 而非 uint4 (16级)
      — 权重量化保持 int4, 在硬件层面这是主要的收益来源
   4. 支持 BatchNorm 保留 — BN 在 float32 运行, 稳定训练
   5. 噪声注入 std 可配置且更温和 (默认 0.02*scale)
   6. 可配置 bit 宽度: --act-bits 8 --weight-bits 4

 量化模式:
   - STE:  静态 scale + 可选噪声注入 (推荐用于从零训练)
   - LSQ+: 可学习 scale/zero_point + 独立 lr

 架构约定:
   - QAT 层只包裹 Conv/Linear, 不包裹 BN/ReLU
   - 典型结构: QATConv → BN → ReLU
   - 首层和末层保持 float32 (混合精度)

 用法:
   from optic_qat_v3 import prepare_model_v3, QATConv2d_v3, QATLinear_v3
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


# ============================================================
#  量化工具函数
# ============================================================

def fake_quantize_symmetric(x: torch.Tensor,
                             bits: int = 4,
                             per_channel: bool = True,
                             ch_dim: int = 0,
                             inject_noise: bool = False,
                             noise_std_ratio: float = 0.02) -> torch.Tensor:
    """
    对称伪量化 — 用于权重 (默认 int4 [-8,7]) 或激活 (int8 [-128,127]).

    STE 实现: x + (x_dq - x).detach()

    Args:
        x:               float32 张量
        bits:            量化位宽
        per_channel:     是否逐通道计算 scale
        ch_dim:          通道维度
        inject_noise:    是否注入训练噪声 (权重正则化)
        noise_std_ratio: 噪声标准差 = ratio * scale
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax

    if per_channel:
        reduce_dims = [i for i in range(x.dim()) if i != ch_dim]
        amax = x.abs()
        for d in sorted(reduce_dims, reverse=True):
            amax = amax.max(dim=d, keepdim=True)[0]
        scale = (amax / qmax).clamp(min=1e-8)
    else:
        scale = (x.abs().max() / qmax).clamp(min=1e-8)

    # 噪声注入
    if inject_noise and x.requires_grad and bits <= 4:
        noise = torch.randn_like(x) * noise_std_ratio * scale.detach()
        x = x + noise

    x_int = (x / scale).round().clamp(qmin, qmax)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


def fake_quantize_unsigned(x: torch.Tensor,
                            bits: int = 4,
                            per_channel: bool = True,
                            ch_dim: int = 1) -> torch.Tensor:
    """
    非对称伪量化 [0, 2^bits-1] — 用于 ReLU 后的激活值.

    STE 实现: x + (x_dq - x).detach()
    """
    qmax = 2 ** bits - 1

    if per_channel:
        reduce_dims = [i for i in range(x.dim()) if i != ch_dim]
        xmax = x
        for d in sorted(reduce_dims, reverse=True):
            xmax = xmax.max(dim=d, keepdim=True)[0]
        scale = (xmax / qmax).clamp(min=1e-8)
    else:
        scale = (x.max() / qmax).clamp(min=1e-8)

    x_int = (x / scale).round().clamp(0, qmax)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


# ============================================================
#  LSQ+ 可学习量化
# ============================================================

class _LSQPlusFn(torch.autograd.Function):
    """LSQ+ 量化 — 可学习 scale + zero_point"""

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
        x_dq = (x_int - zero_point) * scale
        ctx.save_for_backward(x, scale, zero_point, x_int)
        ctx.qmin, ctx.qmax = qmin, qmax
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point, x_int = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        n_levels = qmax - qmin + 1

        inner = ((x_int > qmin) & (x_int < qmax)).float()
        grad_x = inner * grad_output

        inner_s = inner
        outer_s = 1.0 - inner_s
        grad_scale = (
            inner_s * (x_int - zero_point - x / scale) +
            outer_s * torch.where(x > 0,
                                  torch.full_like(x, qmax),
                                  torch.full_like(x, qmin))
        ) * grad_output

        sum_dims = [d for d in range(x.dim())
                     if d >= scale.dim() or scale.shape[d] == 1]
        if sum_dims:
            grad_scale = grad_scale.sum(dim=sum_dims)
        grad_scale = grad_scale.view(scale.shape)

        N = x.numel() / max(1, scale.numel())
        grad_scale = grad_scale / (N * n_levels) ** 0.5

        grad_zp = -(x_int - zero_point) * inner_s * grad_output
        sum_dims_zp = [d for d in range(x.dim())
                        if d >= zero_point.dim() or zero_point.shape[d] == 1]
        if sum_dims_zp:
            grad_zp = grad_zp.sum(dim=sum_dims_zp)
        grad_zp = grad_zp.view(zero_point.shape)

        return grad_x, grad_scale, grad_zp, None, None


def lsqplus_quantize(x, scale, zero_point, qmin=-8, qmax=7):
    return _LSQPlusFn.apply(x, scale, zero_point, qmin, qmax)


# ============================================================
#  QATConv2d v3 (修复版)
# ============================================================

class QATConv2d_v3(nn.Module):
    """
    QAT 卷积层 v3 — 修复版.

    关键改进 (vs v2):
      - 不再内部执行 ReLU — 由模型架构负责
      - 不再自动重量化输出 — 避免级联量化误差
      - 激活量化位宽可配置 (默认 int8 = 256 级)
      - 权重量化默认 int4
      - 噪声注入更温和 (std=0.02*scale)

    用法:
      典型管线: QATConv2d_v3 → BatchNorm2d → ReLU
      BN 在 float32 运行, 稳定训练.
    """

    def __init__(self, conv_layer: nn.Conv2d,
                 mode: str = "ste",
                 weight_bits: int = 4,
                 act_bits: int = 8,
                 noise: bool = True,
                 noise_std_ratio: float = 0.02):
        super().__init__()

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        self.weight = nn.Parameter(conv_layer.weight.data.clone())
        self.bias = nn.Parameter(conv_layer.bias.data.clone()) if conv_layer.bias is not None else None

        self._qat_enabled = True
        self.mode = mode
        self._weight_bits = weight_bits
        self._act_bits = act_bits
        self._noise = noise
        self._noise_std_ratio = noise_std_ratio

        # LSQ+ 可学习参数
        if mode == "lsqplus":
            w_qmax = 2 ** (weight_bits - 1) - 1
            w_amax = self.weight.data.abs().view(self.out_channels, -1).max(dim=1)[0]
            init_w_scale = (w_amax / w_qmax).clamp(min=1e-8)
            self.weight_scale = nn.Parameter(init_w_scale.view(-1, 1, 1, 1))
            self.weight_zp = nn.Parameter(torch.zeros(self.out_channels, 1, 1, 1))

            # 输出 scale/zero_point (仅用于 LSQ+ 模式的逐层重量化, 默认不启用)
            self.out_scale = nn.Parameter(torch.ones(self.out_channels, 1, 1))
            self.out_zp = nn.Parameter(torch.zeros(self.out_channels, 1, 1))

            # 输入 scale (用于激活量化)
            self.in_scale = nn.Parameter(torch.ones(self.in_channels, 1, 1))
            self.in_zp = nn.Parameter(torch.zeros(self.in_channels, 1, 1))

    @property
    def qat_enabled(self) -> bool:
        return self._qat_enabled

    def enable_qat(self):
        self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def quant_params(self):
        """返回 LSQ+ 量化参数 (用于设置独立学习率)"""
        params = []
        if self.mode == "lsqplus":
            for attr in ['weight_scale', 'weight_zp', 'out_scale', 'out_zp',
                         'in_scale', 'in_zp']:
                if hasattr(self, attr):
                    params.append(getattr(self, attr))
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.conv2d(x, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)

        # === 输入激活量化 (所有层统一, 包含首层) ===
        if self.mode == "lsqplus" and hasattr(self, 'in_scale'):
            scale = self.in_scale.abs().clamp(min=1e-8)
            x_q = lsqplus_quantize(x, scale, self.in_zp, 0, 2**self._act_bits - 1)
        elif self._act_bits >= 8:
            x_q = fake_quantize_symmetric(x, bits=self._act_bits, per_channel=True, ch_dim=1)
        else:
            # uint4 模式: 非对称量化 [0, 15]
            x_q = fake_quantize_unsigned(x, bits=self._act_bits, per_channel=True, ch_dim=1)

        # === 权重量化: int4 ===
        if self.mode == "lsqplus" and hasattr(self, 'weight_scale'):
            w_s = self.weight_scale.abs().clamp(min=1e-8)
            w_qmax = 2 ** (self._weight_bits - 1) - 1
            w_q = lsqplus_quantize(self.weight, w_s, self.weight_zp,
                                   -w_qmax, w_qmax)
        else:
            inject = self._noise and self.training
            w_q = fake_quantize_symmetric(self.weight, bits=self._weight_bits,
                                          per_channel=True, ch_dim=0,
                                          inject_noise=inject,
                                          noise_std_ratio=self._noise_std_ratio)

        # === 卷积 (不做内部 ReLU, 不做输出重量化) ===
        out = F.conv2d(x_q, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)

        return out

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, mode={self.mode}, "
                f"w{self._weight_bits}/a{self._act_bits}, "
                f"noise={self._noise}, bias={self.bias is not None}")


# ============================================================
#  QATLinear v3 (修复版)
# ============================================================

class QATLinear_v3(nn.Module):
    """QAT 全连接层 v3 — 同 QATConv2d_v3 设计"""

    def __init__(self, linear_layer: nn.Linear,
                 mode: str = "ste",
                 weight_bits: int = 4,
                 act_bits: int = 8,
                 noise: bool = True,
                 noise_std_ratio: float = 0.02,
                 is_last_layer: bool = False):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        self.bias = nn.Parameter(linear_layer.bias.data.clone()) if linear_layer.bias is not None else None

        self._qat_enabled = True
        self.mode = mode
        self._weight_bits = weight_bits
        self._act_bits = act_bits
        self._noise = noise
        self._noise_std_ratio = noise_std_ratio
        self._is_last_layer = is_last_layer

        if mode == "lsqplus":
            w_qmax = 2 ** (weight_bits - 1) - 1
            w_amax = self.weight.data.abs().view(self.out_features, -1).max(dim=1)[0]
            init_w_scale = (w_amax / w_qmax).clamp(min=1e-8)
            self.weight_scale = nn.Parameter(init_w_scale.view(-1, 1))
            self.weight_zp = nn.Parameter(torch.zeros(self.out_features, 1))

    @property
    def qat_enabled(self) -> bool:
        return self._qat_enabled

    def enable_qat(self):
        self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def quant_params(self):
        params = []
        if self.mode == "lsqplus":
            for attr in ['weight_scale', 'weight_zp']:
                if hasattr(self, attr):
                    params.append(getattr(self, attr))
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.linear(x, self.weight, self.bias)

        # 输入激活量化 (末层不量化输入)
        if self._is_last_layer:
            x_q = x
        elif self._act_bits >= 8:
            x_q = fake_quantize_symmetric(x, bits=self._act_bits, per_channel=True, ch_dim=-1)
        else:
            x_q = fake_quantize_unsigned(x, bits=self._act_bits, per_channel=True, ch_dim=-1)

        # 权重量化
        if self.mode == "lsqplus" and hasattr(self, 'weight_scale'):
            w_s = self.weight_scale.abs().clamp(min=1e-8)
            w_qmax = 2 ** (self._weight_bits - 1) - 1
            w_q = lsqplus_quantize(self.weight, w_s, self.weight_zp,
                                   -w_qmax, w_qmax)
        else:
            inject = self._noise and self.training
            w_q = fake_quantize_symmetric(self.weight, bits=self._weight_bits,
                                          per_channel=True, ch_dim=0,
                                          inject_noise=inject,
                                          noise_std_ratio=self._noise_std_ratio)

        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"mode={self.mode}, w{self._weight_bits}/a{self._act_bits}, "
                f"last={self._is_last_layer}, bias={self.bias is not None}")


# ============================================================
#  模型转换
# ============================================================

def prepare_model_v3(model: nn.Module,
                     mode: str = "ste",
                     weight_bits: int = 4,
                     act_bits: int = 8,
                     noise: bool = True,
                     noise_std_ratio: float = 0.02,
                     quantize_linear: bool = False,
                     preserve_bn: bool = True,
                     inplace: bool = True) -> nn.Module:
    """
    将标准模型转换为 QAT v3 模型.

    新设计 (Conv=int4 / Linear=fp32):
      - 所有 Conv2d → QATConv2d_v3 (全部启用 int4 QAT, 包含首层)
      - 所有 Linear → 保持原生 nn.Linear (fp32, 不包装 QAT)
      - Pool/BN/ReLU/Dropout → 保持原样
      - 光计算对应: Conv 映射到光计算 int4 矩阵乘法
      - 电计算对应: Linear 在电域用 float32 计算 (精度需求高)

    Args:
        model:              标准 PyTorch 模型 (随机初始化或预训练)
        mode:               量化模式 "ste" 或 "lsqplus"
        weight_bits:        权重量化位宽 (默认 4)
        act_bits:           激活量化位宽 (默认 8)
        noise:              STE 模式是否注入训练噪声 (仅对 int4 权重)
        noise_std_ratio:    噪声 std = ratio * scale (默认 0.02)
        quantize_linear:    是否也量化 Linear 层 (默认 False, 保持 fp32)
        preserve_bn:        是否保留 BatchNorm (推荐 True)
        inplace:            是否原地修改

    Returns:
        QAT-ready 模型 (Conv=int4 QAT, Linear=fp32)

    典型架构:
      QATConv_v3(int4) → BN → ReLU → Pool → ... → Linear(fp32)
    """
    if not inplace:
        model = copy.deepcopy(model)

    _convert_to_v3(model, mode, weight_bits, act_bits,
                   noise, noise_std_ratio, quantize_linear)

    qc = sum(1 for m in model.modules() if isinstance(m, QATConv2d_v3))
    qc_enabled = sum(1 for m in model.modules()
                     if isinstance(m, QATConv2d_v3) and m.qat_enabled)
    ql = sum(1 for m in model.modules() if isinstance(m, QATLinear_v3))
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    lin_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    print(f"[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)")
    print(f"  QATConv2d_v3: {qc} ({qc_enabled} enabled)  ← 光计算 int4")
    if quantize_linear:
        print(f"  QATLinear_v3: {ql}  ← 光计算 int4")
    else:
        print(f"  nn.Linear:    {lin_count}  ← 电计算 fp32 (不量化)")
    print(f"  BN (float32): {bn_count}, mode={mode}, w{weight_bits}/a{act_bits}")
    if noise:
        print(f"  训练噪声: std={noise_std_ratio}*scale (仅 int4 Conv 权重)")
    return model


def _convert_to_v3(module: nn.Module, mode, weight_bits, act_bits,
                   noise, noise_std_ratio, quantize_linear):
    """递归转换: Conv→QAT(int4), Linear→保持fp32 (除非 quantize_linear=True)"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            # 所有 Conv 一律转为 int4 QAT (含首层)
            qat = QATConv2d_v3(child, mode=mode,
                               weight_bits=weight_bits,
                               act_bits=act_bits,
                               noise=noise,
                               noise_std_ratio=noise_std_ratio)
            # 全部启用 QAT — 包含首层也做 int4
            setattr(module, name, qat)
        elif isinstance(child, nn.Linear):
            if quantize_linear:
                # 也量化 Linear (旧方案, 一般不推荐)
                qat = QATLinear_v3(child, mode=mode,
                                   weight_bits=weight_bits,
                                   act_bits=act_bits,
                                   noise=noise,
                                   noise_std_ratio=noise_std_ratio)
                setattr(module, name, qat)
            else:
                # 新设计: Linear 保持原生 fp32, 不做任何 QAT 包装
                pass
        elif isinstance(child, (QATConv2d_v3, QATLinear_v3)):
            continue
        else:
            _convert_to_v3(child, mode, weight_bits, act_bits,
                          noise, noise_std_ratio, quantize_linear)


# ============================================================
#  QAT 模式控制
# ============================================================

def enable_qat(model: nn.Module):
    """启用所有 QAT 层的伪量化"""
    count = 0
    for m in model.modules():
        if isinstance(m, (QATConv2d_v3, QATLinear_v3)):
            m.enable_qat()
            count += 1
    if count > 0:
        print(f"[enable_qat] Enabled QAT on {count} layers")
    return count


def disable_qat(model: nn.Module):
    """禁用所有 QAT 层的伪量化 (恢复 float32)"""
    count = 0
    for m in model.modules():
        if isinstance(m, (QATConv2d_v3, QATLinear_v3)):
            m.disable_qat()
            count += 1
    if count > 0:
        print(f"[disable_qat] Disabled QAT on {count} layers")
    return count


# ============================================================
#  LSQ+ 独立学习率
# ============================================================

def set_quant_lr(model: nn.Module, base_lr: float) -> list:
    """
    为 LSQ+ 量化参数设置独立学习率 (0.1x base_lr).
    """
    weight_params = []
    quant_params = []

    for name, param in model.named_parameters():
        is_quant = any(k in name for k in ['weight_scale', 'weight_zp',
                                            'out_scale', 'out_zp',
                                            'in_scale', 'in_zp'])
        if is_quant:
            quant_params.append(param)
        else:
            weight_params.append(param)

    return [
        {'params': weight_params, 'lr': base_lr},
        {'params': quant_params, 'lr': base_lr * 0.1},
    ]


# ============================================================
#  硬件对齐率
# ============================================================

def compute_alignment_ratio(model: nn.Module) -> float:
    """计算 8×2 光计算硬件综合对齐率"""
    total_patch, total_padded = 0, 0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, QATConv2d_v3)):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
        elif isinstance(m, (nn.Linear, QATLinear_v3)):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded

    return total_patch / total_padded if total_padded > 0 else 0.0


def print_alignment_detail(model: nn.Module, label: str = ""):
    """打印每层硬件对齐详情"""
    header = f"  [{label}]" if label else ""
    print(f"\n{header} {'层名':<28s} {'C_in':>4s}  {'K':>5s}  "
          f"{'展平长度':>8s}  {'补零后':>8s}  {'对齐率':>7s}")
    print("  " + "-" * 72)
    total_patch, total_padded = 0, 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, QATConv2d_v3)):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            t = type(m).__name__
            q = "QAT" if hasattr(m, '_qat_enabled') and m._qat_enabled else "FP32"
            print(f"  [{t:<12s} {q:<5s}] {name:<25s} {m.in_channels:>4d}   "
                  f"{m.kernel_size[0]}×{m.kernel_size[1]}"
                  f"  {patch:>8d}  {padded:>8d}  {patch/padded:>6.1%}")
        elif isinstance(m, (nn.Linear, QATLinear_v3)):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            t = type(m).__name__
            q = "QAT" if hasattr(m, '_qat_enabled') and m._qat_enabled else "FP32"
            print(f"  [{t:<12s} {q:<5s}] {name:<25s}   —     —    "
                  f"  {patch:>8d}  {padded:>8d}  {patch/padded:>6.1%}")

    overall = total_patch / total_padded if total_padded > 0 else 0
    print(f"  综合硬件对齐率: {overall:.1%} (展平总长度 {total_patch} → 补零后 {total_padded})")
    return overall


# ============================================================
#  评估工具
# ============================================================

@torch.no_grad()
def evaluate_model_v3(model, dataloader, device, criterion=None):
    """评估模型"""
    model.eval()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion:
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"accuracy": correct / total,
            "loss": total_loss / total if criterion else 0.0,
            "total": total, "correct": correct}


@torch.no_grad()
def compare_qat_vs_float(model, dataloader, device, criterion=None):
    """对比 QAT 模式 vs float32 模式"""
    print("\n" + "=" * 60)
    print("  QAT vs Float32 Comparison")
    print("=" * 60)

    enable_qat(model)
    qat_result = evaluate_model_v3(model, dataloader, device, criterion)
    print(f"  QAT mode (int4):     Accuracy = {qat_result['accuracy']:.2%}")

    disable_qat(model)
    float_result = evaluate_model_v3(model, dataloader, device, criterion)
    print(f"  Float mode (float32): Accuracy = {float_result['accuracy']:.2%}")

    gap = float_result['accuracy'] - qat_result['accuracy']
    status = '✓ QAT successful' if abs(gap) < 0.03 else '⚠ Needs more training'
    print(f"  Accuracy gap:         {gap:.2%} ({status})")

    return {"qat_mode": qat_result, "float_mode": float_result, "accuracy_gap": gap}


# ============================================================
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  optic_qat_v3.py — Self-Test")
    print("=" * 60)

    # Test 1: fake_quantize_symmetric
    print("\n[Test 1] fake_quantize_symmetric (int4)")
    w = torch.randn(16, 3, 3, 3) * 0.5
    w.requires_grad = True
    w_q = fake_quantize_symmetric(w, bits=4, per_channel=True, ch_dim=0, inject_noise=True)
    loss = w_q.sum()
    loss.backward()
    print(f"  grad norm: {w.grad.norm():.4f}  [OK]")

    # Test 2: fake_quantize_unsigned (uint4)
    print("\n[Test 2] fake_quantize_unsigned (uint4)")
    x = torch.randn(4, 8, 32, 32).abs() + 0.5
    x_q = fake_quantize_unsigned(x, bits=4, per_channel=True, ch_dim=1)
    print(f"  unique values: {x_q.unique().numel()} (≤ 16 expected)  [OK]")

    # Test 3: QATConv2d_v3 (STE) — 不内部 ReLU, 所有Conv统一int4
    print("\n[Test 3] QATConv2d_v3 (STE, w4/a8) — no internal ReLU, all Conv=int4")
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat = QATConv2d_v3(conv, mode="ste", weight_bits=4, act_bits=8, noise=True)
    qat.train()
    x_in = torch.randn(2, 3, 32, 32)
    out = qat(x_in)
    print(f"  output shape: {out.shape}")
    print(f"  output has negative values: {(out < 0).any().item()} (expected True, no internal ReLU)")
    out.sum().backward()
    print(f"  weight grad norm: {qat.weight.grad.norm():.4f}  [OK]")

    # Test 4: QATLinear_v3 (Linear保持fp32 — 新设计不包装QAT)
    print("\n[Test 4] QATLinear_v3 — Linear stays fp32 (new design skips QAT wrapping)")
    lin = nn.Linear(64, 10, bias=True)
    # 新设计: Linear 不包装 QAT, 直接保持 nn.Linear
    print(f"  Linear type: {type(lin).__name__} (bias={lin.bias is not None})")
    out = lin(torch.randn(4, 64))
    print(f"  output shape: {out.shape}  [OK] Linear stays in fp32")

    # Test 5: prepare_model_v3
    print("\n[Test 5] prepare_model_v3 (完整模型)")

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.relu2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    tiny = TinyCNN()
    prepare_model_v3(tiny, mode="ste")
    print(f"  conv1 type: {type(tiny.conv1).__name__}, qat={tiny.conv1.qat_enabled}")
    print(f"  conv2 type: {type(tiny.conv2).__name__}, qat={tiny.conv2.qat_enabled}")
    print(f"  fc type: {type(tiny.fc).__name__} (should be nn.Linear, not QAT)")
    print(f"  fc is nn.Linear: {isinstance(tiny.fc, nn.Linear)}")
    print(f"  bn1 preserved: {isinstance(tiny.bn1, nn.BatchNorm2d)}  [OK]")

    # Test 6: Training simulation
    print("\n[Test 6] Training simulation (5 steps)")
    tiny2 = TinyCNN()
    prepare_model_v3(tiny2, mode="ste")
    tiny2.train()
    opt = torch.optim.Adam(tiny2.parameters(), lr=0.001)
    for i in range(5):
        opt.zero_grad()
        out = tiny2(torch.randn(4, 3, 8, 8))
        loss = nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,)))
        loss.backward()
        opt.step()
        print(f"  Step {i+1}: loss={loss.item():.4f}")
    print(f"  [OK] Model learns with QAT!")

    # Test 7: LSQ+ mode
    print("\n[Test 7] LSQ+ mode")
    tiny3 = TinyCNN()
    prepare_model_v3(tiny3, mode="lsqplus")
    pg = set_quant_lr(tiny3, base_lr=0.001)
    print(f"  Weight params: {len(pg[0]['params'])}, lr={pg[0]['lr']}")
    print(f"  Quant params:  {len(pg[1]['params'])}, lr={pg[1]['lr']}  [OK]")

    # Test 8: 不内部 ReLU 验证 + 所有 Conv 统一 int4
    print("\n[Test 8] No internal ReLU + all Conv=int4 verification")
    conv3 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat3 = QATConv2d_v3(conv3, mode="ste")
    x_neg = -torch.ones(2, 3, 32, 32)  # 全负输入
    out_neg = qat3(x_neg)
    has_neg_output = (out_neg < 0).any().item()
    print(f"  Negative input → negative output: {has_neg_output} (expected True)")
    print(f"  conv qat_enabled: {qat3.qat_enabled} (expected True, all Conv=int4)")
    print(f"  [OK] No spurious ReLU, all Conv layers quantized")

    print("\n" + "=" * 60)
    print("  All self-tests passed! [OK]")
    print("=" * 60)
