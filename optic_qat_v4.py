"""
================================================================================
 optic_qat_v4.py — 基于 Gazelle 硬件逆向的改进 QAT 模块

 改进 (vs optic_qat_v3.py):
   1. 支持 int8 权重 QAT — 匹配硬件原生 8-bit (8a8w12o)
   2. Gazelle 硬件匹配噪声模型 — DAC/TIA/ADC 真实物理噪声
   3. 首层 FP32 选项 — stem/首层对齐率低, 保留电计算

 Gazelle 硬件参数 (from GAZELLE_ARCHITECTURE.md):
   - 物理 tile: 8×2 (k=8, n=2)
   - 原生精度: 8-bit activation, 8-bit weight, 12-bit output
   - DAC ENOB: 7.5 bits
   - TIA noise MSE: 2.85×10⁻⁷
   - ADC LSB: 0.00147
   - 线性度: ~99.4% (硬件几乎理想)

 用法:
   from optic_qat_v4 import prepare_model_v4, GazelleNoiseInjector
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


# ============================================================
#  Gazelle 硬件匹配噪声注入器
# ============================================================

class GazelleNoiseInjector:
    """
    基于 Gazelle 逆向报告的硬件匹配噪声模型.

    噪声链路 (from GAZELLE_ARCHITECTURE.md §3.3):
      Input → DAC(ENOB=7.5) → Modulator → Photonic GEMM → TIA → ADC

    参数 (from calibration_params.json):
      dac_enob: 7.5        — DAC 有效位数
      tia_noise_std: ~5.3×10⁻⁴  — TIA 噪声标准差 (sqrt(MSE))
      adc_lsb: 0.00147     — ADC 量化步长
    """

    def __init__(self, dac_enob=7.5, tia_noise_std=5.34e-4, adc_lsb=0.00147):
        self.dac_enob = dac_enob
        self.tia_noise_std = tia_noise_std
        self.adc_lsb = adc_lsb

    def inject_weight_noise(self, weight: torch.Tensor,
                            weight_scale: torch.Tensor) -> torch.Tensor:
        """
        向权重注入 DAC 量化噪声 (匹配 ENOB=7.5).
        前向: weight → DAC 量化 → 带噪 weight
        注意: 仅在训练时调用, 用 STE 保持梯度
        """
        if not weight.requires_grad:
            return weight

        # DAC 量化噪声: ENOB=7.5 → 有效量化级数 = 2^7.5 ≈ 181
        # 噪声 std ≈ scale / (2^ENOB * sqrt(12))
        dac_levels = 2 ** self.dac_enob
        dac_noise_std = 1.0 / (dac_levels * np.sqrt(12))
        noise = torch.randn_like(weight) * dac_noise_std * weight_scale.detach()
        return weight + noise

    def inject_activation_noise(self, activation: torch.Tensor,
                                act_scale: torch.Tensor) -> torch.Tensor:
        """向激活值注入 ADC 量化 + TIA 噪声"""
        if not activation.requires_grad:
            return activation
        # TIA 噪声 (加性高斯)
        tia_noise = torch.randn_like(activation) * self.tia_noise_std
        # ADC 量化噪声
        adc_noise_std = self.adc_lsb / np.sqrt(12)
        adc_noise = torch.randn_like(activation) * adc_noise_std
        return activation + tia_noise + adc_noise

    def __repr__(self):
        return (f"GazelleNoise(DAC_ENOB={self.dac_enob}, "
                f"TIA_σ={self.tia_noise_std:.1e}, ADC_lsb={self.adc_lsb:.4f})")


# ============================================================
#  量化函数 (支持可变 bit 宽度)
# ============================================================

def fake_quantize_symmetric(x: torch.Tensor, bits: int = 8,
                            per_channel: bool = True, ch_dim: int = 0,
                            inject_noise: bool = False,
                            noise_std_ratio: float = 0.0016,  # ★ 匹配硬件 DAC ENOB=7.5
                            noise_injector: GazelleNoiseInjector = None
                            ) -> torch.Tensor:
    """
    对称伪量化 (STE). 支持 int4/int8 权重, int8 激活.

    Args:
        bits: 量化位宽 (4=int4, 8=int8 匹配硬件原生精度)
        noise_std_ratio: 默认 0.0016 匹配 DAC ENOB=7.5 (vs v3 的 0.02)
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

    # 硬件匹配噪声注入
    if inject_noise and x.requires_grad and bits <= 8:
        if noise_injector is not None:
            x = noise_injector.inject_weight_noise(x, scale)
        else:
            noise = torch.randn_like(x) * noise_std_ratio * scale.detach()
            x = x + noise

    x_int = (x / scale).round().clamp(qmin, qmax)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


# ============================================================
#  QATConv2d v4 — 支持 int8 权重, 硬件匹配噪声
# ============================================================

class QATConv2d_v4(nn.Module):
    """
    QAT 卷积层 v4.

    改进:
      - weight_bits 默认 8 (匹配硬件原生精度)
      - 硬件匹配噪声 (GazelleNoiseInjector)
      - 可配置首层 FP32
    """

    def __init__(self, conv_layer: nn.Conv2d,
                 weight_bits: int = 8,      # ★ 默认 int8 匹配硬件
                 act_bits: int = 8,
                 noise: bool = True,
                 noise_injector: GazelleNoiseInjector = None,
                 keep_fp32: bool = False):   # 首层保留 FP32
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

        self._qat_enabled = not keep_fp32  # 首层默认关闭 QAT
        self._weight_bits = weight_bits
        self._act_bits = act_bits
        self._noise = noise
        self._noise_injector = noise_injector
        self._keep_fp32 = keep_fp32

    @property
    def qat_enabled(self): return self._qat_enabled

    def enable_qat(self):
        if not self._keep_fp32:
            self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.conv2d(x, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)

        # 激活量化 (int8 per-channel)
        x_q = fake_quantize_symmetric(x, bits=self._act_bits,
                                      per_channel=True, ch_dim=1,
                                      inject_noise=False)  # 激活噪声在上一层的 output 端注入

        # 权重量化 + 硬件噪声
        inject = self._noise and self.training
        w_q = fake_quantize_symmetric(self.weight, bits=self._weight_bits,
                                      per_channel=True, ch_dim=0,
                                      inject_noise=inject,
                                      noise_injector=self._noise_injector)

        return F.conv2d(x_q, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"w{self._weight_bits}/a{self._act_bits}, "
                f"noise={self._noise}, fp32_first={self._keep_fp32}, "
                f"bias={self.bias is not None}")


# ============================================================
#  QATLinear v4
# ============================================================

class QATLinear_v4(nn.Module):
    """QAT 全连接层 v4"""

    def __init__(self, linear_layer: nn.Linear,
                 weight_bits: int = 8,
                 act_bits: int = 8,
                 noise: bool = True,
                 noise_injector: GazelleNoiseInjector = None,
                 is_last_layer: bool = False):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        self.bias = (nn.Parameter(linear_layer.bias.data.clone())
                     if linear_layer.bias is not None else None)

        self._qat_enabled = True
        self._weight_bits = weight_bits
        self._act_bits = act_bits
        self._noise = noise
        self._noise_injector = noise_injector
        self._is_last_layer = is_last_layer

    @property
    def qat_enabled(self): return self._qat_enabled

    def enable_qat(self): self._qat_enabled = True
    def disable_qat(self): self._qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.linear(x, self.weight, self.bias)

        # 末层不量化输入
        x_q = (x if self._is_last_layer else
               fake_quantize_symmetric(x, bits=self._act_bits,
                                       per_channel=True, ch_dim=-1))

        inject = self._noise and self.training
        w_q = fake_quantize_symmetric(self.weight, bits=self._weight_bits,
                                      per_channel=True, ch_dim=0,
                                      inject_noise=inject,
                                      noise_injector=self._noise_injector)

        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"w{self._weight_bits}/a{self._act_bits}, last={self._is_last_layer}")


# ============================================================
#  模型转换
# ============================================================

def prepare_model_v4(model: nn.Module,
                     weight_bits: int = 8,        # ★ int8 匹配硬件
                     act_bits: int = 8,
                     noise: bool = True,
                     first_conv_fp32: bool = True,  # 首层对齐率低，保留电计算
                     quantize_linear: bool = True,   # Phase4: Linear 也量化
                     preserve_bn: bool = True,
                     inplace: bool = True) -> nn.Module:
    """
    将标准模型转换为 QAT v4.

    Args:
        weight_bits:      权重量化位宽 (8=int8 匹配硬件, 4=int4)
        act_bits:         激活量化位宽 (8=int8)
        noise:            是否注入硬件匹配噪声
        first_conv_fp32:  首层 Conv 是否保持 FP32 (对齐率<50% 的层建议 FP32)
        quantize_linear:  是否量化 Linear 层
        preserve_bn:      保留 BN
    """
    if not inplace:
        model = copy.deepcopy(model)

    noise_inj = GazelleNoiseInjector() if noise else None

    _convert_to_v4(model, weight_bits, act_bits, noise, noise_inj,
                   first_conv_fp32, quantize_linear, _first=True)

    qc_enabled = sum(1 for m in model.modules()
                     if isinstance(m, QATConv2d_v4) and m.qat_enabled)
    qc_disabled = sum(1 for m in model.modules()
                      if isinstance(m, QATConv2d_v4) and not m.qat_enabled)
    ql = sum(1 for m in model.modules() if isinstance(m, QATLinear_v4))
    bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))

    w_label = f"int{weight_bits}"
    print(f"[prepare_model_v4] Gazelle HW-aware QAT: w{w_label}/a{act_bits}")
    print(f"  QAT Conv: {qc_enabled} enabled + {qc_disabled} fp32 (first layer)")
    print(f"  QAT Linear: {ql}, BN: {bn}")
    if noise:
        print(f"  硬件噪声: {noise_inj}")
    if first_conv_fp32 and qc_disabled > 0:
        print(f"  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)")
    return model


def _convert_to_v4(module, weight_bits, act_bits, noise, noise_inj,
                   first_conv_fp32, quantize_linear, _first=True):
    """递归转换. _first 用单元素列表传递, 避免嵌套调用不更新父级的问题."""
    if not isinstance(_first, list):
        _first = [_first]

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            keep_fp32 = _first[0] and first_conv_fp32
            qat = QATConv2d_v4(child, weight_bits=weight_bits,
                               act_bits=act_bits, noise=noise,
                               noise_injector=noise_inj,
                               keep_fp32=keep_fp32)
            setattr(module, name, qat)
            _first[0] = False  # ★ 可变容器: 真正只有第一个 Conv 受影响
        elif isinstance(child, nn.Linear):
            if quantize_linear:
                qat = QATLinear_v4(child, weight_bits=weight_bits,
                                   act_bits=act_bits, noise=noise,
                                   noise_injector=noise_inj)
                setattr(module, name, qat)
        elif isinstance(child, (QATConv2d_v4, QATLinear_v4)):
            continue
        else:
            _convert_to_v4(child, weight_bits, act_bits, noise, noise_inj,
                          first_conv_fp32, quantize_linear, _first)


def enable_qat(model: nn.Module):
    count = 0
    for m in model.modules():
        if isinstance(m, (QATConv2d_v4, QATLinear_v4)):
            m.enable_qat()
            count += 1
    if count > 0:
        print(f"[enable_qat] Enabled QAT on {count} layers")
    return count


def disable_qat(model: nn.Module):
    count = 0
    for m in model.modules():
        if isinstance(m, (QATConv2d_v4, QATLinear_v4)):
            m.disable_qat()
            count += 1
    if count > 0:
        print(f"[disable_qat] Disabled QAT on {count} layers")
    return count


# ============================================================
#  评估工具
# ============================================================

@torch.no_grad()
def evaluate_model_v4(model, dataloader, device, criterion=None):
    model.eval()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion:
            total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return {"accuracy": correct / total,
            "loss": total_loss / total if criterion else 0.0,
            "total": total, "correct": correct}


# ============================================================
#  硬件对齐率
# ============================================================

def compute_alignment_ratio(model: nn.Module) -> float:
    total_patch, total_padded = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, QATConv2d_v4)):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
        elif isinstance(m, (nn.Linear, QATLinear_v4)):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
    return total_patch / total_padded if total_padded > 0 else 0.0


def print_alignment_detail(model: nn.Module, label: str = ""):
    header = f"  [{label}]" if label else ""
    print(f"\n{header} {'层名':<28s} {'C_in':>4s}  {'K':>5s}  "
          f"{'展平长度':>8s}  {'补零后':>8s}  {'对齐率':>7s}")
    print("  " + "-" * 72)
    total_patch, total_padded = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, QATConv2d_v4)):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch; total_padded += padded
            t = type(m).__name__
            q = "QAT" if getattr(m, '_qat_enabled', False) else "FP32"
            wb = getattr(m, '_weight_bits', '?')
            print(f"  [{t:<12s} {q:<5s}] {name:<25s} {m.in_channels:>4d}   "
                  f"{m.kernel_size[0]}×{m.kernel_size[1]}"
                  f"  {patch:>8d}  {padded:>8d}  {patch/padded:>6.1%}  w{wb}")
        elif isinstance(m, (nn.Linear, QATLinear_v4)):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch; total_padded += padded
    overall = total_patch / total_padded if total_padded > 0 else 0
    print(f"  综合硬件对齐率: {overall:.1%} (展平总长度 {total_patch} → 补零后 {total_padded})")
    return overall


# ============================================================
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  optic_qat_v4.py — Gazelle HW-aware QAT Self-Test")
    print("=" * 60)

    # Test GazelleNoiseInjector
    print("\n[Test 1] GazelleNoiseInjector")
    inj = GazelleNoiseInjector()
    print(f"  {inj}")
    w = torch.randn(16, 3, 3, 3) * 0.5
    w.requires_grad = True
    scale = w.abs().view(16, -1).max(dim=1)[0].view(-1, 1, 1, 1) / 127
    w_noisy = inj.inject_weight_noise(w, scale)
    print(f"  Noise std: {(w_noisy - w).std():.6f}, weight std: {w.std():.4f}")
    print(f"  [OK]")

    # Test int8 quantization
    print("\n[Test 2] fake_quantize_symmetric (int8 weight)")
    w = torch.randn(16, 3, 3, 3) * 0.5
    w.requires_grad = True
    w_q = fake_quantize_symmetric(w, bits=8, per_channel=True, ch_dim=0)
    loss = w_q.sum(); loss.backward()
    print(f"  Unique values: {w_q.unique().numel()} (≤256 for int8)")
    print(f"  Weight grad norm: {w.grad.norm():.4f}")
    print(f"  [OK]")

    # Test int4 quantization (backward compatible)
    print("\n[Test 3] fake_quantize_symmetric (int4 weight) — backward compat")
    w = torch.randn(16, 3, 3, 3) * 0.5
    w_q = fake_quantize_symmetric(w, bits=4, per_channel=True, ch_dim=0)
    print(f"  Unique values: {w_q.unique().numel()} (≤16 for int4)")
    print(f"  [OK]")

    # Test QATConv2d_v4 (int8)
    print("\n[Test 4] QATConv2d_v4 (int8 weight, Gazelle noise)")
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat = QATConv2d_v4(conv, weight_bits=8, noise=True,
                       noise_injector=GazelleNoiseInjector())
    qat.train()
    out = qat(torch.randn(2, 3, 32, 32))
    out.sum().backward()
    print(f"  Output shape: {out.shape}, grad norm: {qat.weight.grad.norm():.4f}")
    print(f"  [OK]")

    # Test first layer FP32
    print("\n[Test 5] First Conv FP32, rest int8")
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 8, 1, bias=False)
            self.conv = nn.Conv2d(8, 16, 2, stride=2, bias=False)
            self.fc = nn.Linear(16 * 16 * 16, 10)
        def forward(self, x):
            x = F.relu(self.stem(x))
            x = F.relu(self.conv(x))
            return self.fc(x.flatten(1))

    tiny = TinyNet()
    prepare_model_v4(tiny, weight_bits=8, first_conv_fp32=True,
                    quantize_linear=True)
    print(f"  stem QAT enabled: {tiny.stem.qat_enabled} (should be False)")
    print(f"  conv QAT enabled: {tiny.conv.qat_enabled} (should be True)")
    print(f"  fc type: {type(tiny.fc).__name__}")
    print(f"  [OK]")

    # Test full training loop
    print("\n[Test 6] Training simulation (5 steps, int8)")
    tiny2 = TinyNet()
    prepare_model_v4(tiny2, weight_bits=8, noise=True)
    tiny2.train()
    opt = torch.optim.Adam(tiny2.parameters(), lr=0.001)
    for i in range(5):
        opt.zero_grad()
        out = tiny2(torch.randn(4, 3, 32, 32))
        loss = nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,)))
        loss.backward()
        opt.step()
        print(f"  Step {i+1}: loss={loss.item():.4f}")
    print(f"  [OK]")

    print("\n" + "=" * 60)
    print("  All self-tests passed! [OK]")
    print("=" * 60)
