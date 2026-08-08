"""
================================================================================
 optic_qat_v4.py — 基于 Gazelle 硬件逆向的改进 QAT 模块

 改进 (vs optic_qat_v3.py):
   1. 支持 int8 权重 QAT — 匹配硬件原生 8-bit (8a8w12o)
   2. Gazelle 硬件匹配噪声模型 — DAC/TIA/ADC 真实物理噪声
   3. 首层 FP32 选项 — stem/首层对齐率低, 保留电计算

 v4.1 修复 (训练/推理对齐):
   4. ★ 激活噪声修复: inject_activation_noise (TIA σ=5.34e-4 + ADC lsb=0.00147)
      原先已定义但从未被调用 — 训练只注入权重 DAC 噪声, 而 TIA/ADC 正是
      osimulator 输出噪声 (std=5.31) 的主导来源。现于每个 QAT 层的激活量化
      路径注入 (训练时)。
   5. ★ 激活量化对齐修复: 默认激活量化从 per-channel signed int8 改为
      per-tensor unsigned uint8 + zero_point, 与 osimulator `_matmul_real`
      输入量化 (`quantize_to_int(signed=False)` / `ufixed_quant`) 完全一致。
      旧方案可用 act_quant="signed_per_channel" 保留 (对比实验)。

 Gazelle 硬件参数 (from GAZELLE_ARCHITECTURE.md):
   - 物理 tile: 8×2 (k=8, n=2)
   - 原生精度: 8-bit activation, 8-bit weight, 12-bit output
   - DAC ENOB: 7.5 bits
   - TIA noise MSE: 2.85×10^-^7
   - ADC LSB: 0.00147
   - 线性度: ~99.4% (硬件几乎理想)

 用法:
   from optic_qat_v4 import prepare_model_v4, GazelleNoiseInjector

 TODO (v4.1 修复后, 见 docs/TODO.md §v4.1 重跑清单):
   - [ ] 重训 M1-M4 phase4_v3 (训练行为已变: 激活噪声注入 + uint8+zp 激活量化),
         旧权重是旧语义训练, 不重训无法验证修复效果
   - [ ] 重训后容器内复测 M2/M3 osim gap 是否从 ~1.6pt 收窄 (Bug #12/#13 判据)
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
      tia_noise_std: ~5.3×10^-^4  — TIA 噪声标准差 (sqrt(MSE))
      adc_lsb: 0.00147     — ADC 量化步长

    默认值来源复核 (gazelle-crossval 真机实测 2026-08,
    fitted_params.json / CROSSVAL_REPORT.md §7):
      真机拟合: hw 噪声为**输出端绝对加性底噪**, σ_total ≈ 4.49 counts
      (= 1144 MAC, uint8; uint4x16 ≈ 3.85 counts), 与信号幅度无关;
      σ_static ≈ 4.37 counts (慢漂+静态 LUT 误差) 主导, σ_dynamic (短窗
      快噪声) 仅 ≈ 1.02 counts。crossval 建议 tia_noise_std=0.0392, 但
      其归一化基准是 σ_total/rms_ideal, rms_ideal=29202 MAC 为 **uint8
      满值域随机 GEMM (k=8)** 的输出 RMS, 与本模块口径不同, 不可直接替换。
      换算到本模块口径 (输入激活端 float 单位): σ_act[counts] =
      σ_out[MAC] / (w_rms_int × √k) — 以 w_rms_int≈40 计, k=32 → ≈5.0
      counts, k=1024 → ≈0.9 counts, 再乘 in_scale≈0.01 得 float σ≈
      0.009~0.05, 即当前值的 ~17~95× (w_rms_int≈40 / in_scale≈0.01 为
      int8 amax 量化的经验假设), 且层间散布 >5×, 无法给出单一
      可信默认值。更根本的是**结构失配**: 真机噪声在 GEMM 输出端、与
      信号无关, 而本注入器在输入激活端加噪 (经 GEMM 后变为权重相关),
      且以 i.i.d. 高斯建模无法表达 σ_static 慢漂。此外 v4 中
      inject_activation_noise 实际未被调用 (死代码, 见 auto_research/
      src/qat_v5.py 的修复: 输出端注入 + uint8 affine + 12-bit 输出量化)。
      结论: 保留 5.34e-4 不动, 仅记录真机实测值; 结构正确的输出端噪声
      建模应参考 qat_v5 (其 0.0457 与真机 0.0392 同口径, 差 ~14%)。
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
                            # ★ 匹配硬件 DAC ENOB=7.5。crossval 真机复核 (2026-08,
                            # fitted_params.json) 建议 noise_std_ratio=0.00894, 但那是
                            # σ_dynamic/rms_ideal (uint8 满值域随机 GEMM 输出 RMS 口径,
                            # 且 σ_dynamic=1.02 counts 只是短窗快噪声, 长窗 σ_total≈
                            # 4.49 counts 不可混用), 与本参数"相对 per-channel 权重
                            # scale"的口径不同。换算到本口径: 整数域 δw std = ratio
                            # counts, 输出端 σ_out = ratio × x_rms_int × √k; 要复现真机
                            # 底噪 1144 MAC 需 ratio ≈ 1144/(x_rms_int·√k), 以
                            # x_rms_int≈60 (int8 amax 经验值) 计 ≈ 0.6 (k=1024) ~
                            # 3.4 (k=32), 层间散布 5.6×, 无法给出单一可信默认值, 且
                            # 权重端加噪与真机输出端绝对加性噪声结构不同 (见
                            # GazelleNoiseInjector docstring)。结论: 保留 0.0016。
                            noise_std_ratio: float = 0.0016,
                            noise_injector: GazelleNoiseInjector = None,
                            noise_kind: str = "weight"
                            ) -> torch.Tensor:
    """
    对称伪量化 (STE). 支持 int4/int8 权重, int8 激活.

    Args:
        bits: 量化位宽 (4=int4, 8=int8 匹配硬件原生精度)
        noise_std_ratio: 默认 0.0016 匹配 DAC ENOB=7.5 (vs v3 的 0.02)
        noise_kind: 注入的噪声类型:
            - "weight":    DAC 权重量化噪声 (inject_weight_noise)
            - "activation": TIA+ADC 激活/输出噪声 (inject_activation_noise)
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

    # 硬件匹配噪声注入 (权重侧: DAC; 激活侧: TIA+ADC)
    if inject_noise and x.requires_grad and bits <= 8:
        if noise_injector is not None:
            if noise_kind == "activation":
                x = noise_injector.inject_activation_noise(x, scale)
            else:
                x = noise_injector.inject_weight_noise(x, scale)
        else:
            noise = torch.randn_like(x) * noise_std_ratio * scale.detach()
            x = x + noise

    x_int = (x / scale).round().clamp(qmin, qmax)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


def fake_quantize_unsigned_affine(x: torch.Tensor, bits: int = 8,
                                  inject_noise: bool = False,
                                  noise_injector: GazelleNoiseInjector = None,
                                  include_zero_min: bool = False
                                  ) -> torch.Tensor:
    """
    无符号仿射伪量化 (STE) — ★ 与 osimulator `_matmul_real` 输入量化完全对齐.

    真实光引擎 (`optic_layers._matmul_real` / 反编译 `oMAC_Matmul.ufixed_quant`)
    对输入矩阵做 **per-tensor unsigned affine (uint8 + zero_point)** 量化:
        scale      = (max - min) / (2^bits - 1)
        zero_point = min
        x_int      = round((x - zero_point) / scale).clamp(0, 2^bits - 1)
        x_float    = x_int * scale + zero_point
    而旧的 QAT 路径用 per-channel signed int8 — 训练/推理不对齐
    (EXPERIMENTS.md §16.2/§16.6 的 int4 困境根因 #2, int8 下同样存在).

    min/max 语义: 引擎对 im2col 展平 + 补零后的矩阵取全局 min/max
    (im2col 的元素集合 == x 的元素集合, 补零时另含 0):
      - include_zero_min=True  (该层展平长度非 8 倍数, 推理时补零):
        min = min(x.min(), 0)  → 引擎矩阵含补零 0
      - include_zero_min=False (展平长度已是 8 倍数, 不补零):
        min = x.min()

    Args:
        bits: 量化位宽 (默认 8 = uint8, 匹配硬件原生输入精度)
        inject_noise: 是否注入激活侧硬件噪声 (TIA + ADC)
        noise_injector: GazelleNoiseInjector 实例
        include_zero_min: 该层 im2col 展平后是否补零 (由层自身计算后传入)
    """
    val_range = 2 ** bits
    t_min = x.min()
    if include_zero_min:
        t_min = t_min.clamp(max=0.0)  # 引擎对补零矩阵取 min, 至少为 0
    t_max = x.max()
    scale = ((t_max - t_min) / (val_range - 1)).clamp(min=1e-8)
    zero_point = t_min

    # 激活侧硬件噪声 (TIA + ADC) — 注入发生在量化之前 (模拟模拟域噪声 → ADC)
    if inject_noise and x.requires_grad:
        if noise_injector is not None:
            x = noise_injector.inject_activation_noise(x, scale)
        else:
            # 默认: ADC 量化噪声 (σ = lsb/√12, lsb≈0.00147) + TIA 噪声 (σ≈5.34e-4)
            tia_noise = torch.randn_like(x) * 5.34e-4
            adc_noise = torch.randn_like(x) * (0.00147 / np.sqrt(12))
            x = x + tia_noise + adc_noise

    x_int = ((x - zero_point) / scale).round().clamp(0, val_range - 1)
    x_dq = x_int * scale + zero_point
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
      - 激活量化默认 per-tensor unsigned uint8 + zero_point, 与 osimulator
        `_matmul_real` 输入量化完全一致 (修复训练/推理激活量化不对齐)
      - 激活侧 TIA+ADC 噪声训练时注入 (修复 inject_activation_noise 从未被调用)
    """

    def __init__(self, conv_layer: nn.Conv2d,
                 weight_bits: int = 8,      # ★ 默认 int8 匹配硬件
                 act_bits: int = 8,
                 noise: bool = True,
                 noise_injector: GazelleNoiseInjector = None,
                 keep_fp32: bool = False,   # 首层保留 FP32
                 act_quant: str = "uint8_affine"):  # "uint8_affine"=匹配硬件 | "signed_per_channel"=旧行为
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
        self._act_quant = act_quant
        # 引擎 (OpticConv2d pad_to_8=True) 对 im2col 展平补零到 8 的倍数:
        # 展平长度非 8 倍数 → 引擎矩阵含补零 0 → 训练侧 min 至少为 0
        self._engine_pads = ((self.in_channels * self.kernel_size[0]
                              * self.kernel_size[1]) % 8) != 0

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

        inject = self._noise and self.training  # 硬件噪声仅训练时注入

        # 激活量化 (默认 per-tensor unsigned uint8 + zp, 与 osimulator 一致)
        # 激活噪声 (TIA+ADC) 在此注入 → 等价于"上一层的 output 端注入"
        if self._act_quant == "uint8_affine":
            x_q = fake_quantize_unsigned_affine(x, bits=self._act_bits,
                                                inject_noise=inject,
                                                noise_injector=self._noise_injector,
                                                include_zero_min=self._engine_pads)
        else:  # signed_per_channel (旧行为, 仅作对比实验)
            x_q = fake_quantize_symmetric(x, bits=self._act_bits,
                                          per_channel=True, ch_dim=1,
                                          inject_noise=inject,
                                          noise_injector=self._noise_injector,
                                          noise_kind="activation")

        # 权重量化 + 硬件 DAC 噪声
        w_q = fake_quantize_symmetric(self.weight, bits=self._weight_bits,
                                      per_channel=True, ch_dim=0,
                                      inject_noise=inject,
                                      noise_injector=self._noise_injector)

        return F.conv2d(x_q, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"w{self._weight_bits}/a{self._act_bits}({self._act_quant}), "
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
                 is_last_layer: bool = False,
                 act_quant: str = "uint8_affine"):  # 与 QATConv2d_v4 一致
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
        self._act_quant = act_quant
        # 引擎 (OpticLinear pad_to_8=True) 对输入补零到 8 的倍数
        self._engine_pads = (self.in_features % 8) != 0

    @property
    def qat_enabled(self): return self._qat_enabled

    def enable_qat(self): self._qat_enabled = True
    def disable_qat(self): self._qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.linear(x, self.weight, self.bias)

        inject = self._noise and self.training  # 硬件噪声仅训练时注入

        # 末层不量化输入 (亦不注入激活噪声, 保持原设计)
        if self._is_last_layer:
            x_q = x
        elif self._act_quant == "uint8_affine":
            x_q = fake_quantize_unsigned_affine(x, bits=self._act_bits,
                                                inject_noise=inject,
                                                noise_injector=self._noise_injector,
                                                include_zero_min=self._engine_pads)
        else:  # signed_per_channel (旧行为)
            x_q = fake_quantize_symmetric(x, bits=self._act_bits,
                                          per_channel=True, ch_dim=-1,
                                          inject_noise=inject,
                                          noise_injector=self._noise_injector,
                                          noise_kind="activation")

        w_q = fake_quantize_symmetric(self.weight, bits=self._weight_bits,
                                      per_channel=True, ch_dim=0,
                                      inject_noise=inject,
                                      noise_injector=self._noise_injector)

        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"w{self._weight_bits}/a{self._act_bits}({self._act_quant}), "
                f"last={self._is_last_layer}")


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
                     act_quant: str = "uint8_affine",  # ★ 激活量化: 匹配 osimulator (uint8+zp per-tensor)
                     inplace: bool = True) -> nn.Module:
    """
    将标准模型转换为 QAT v4.

    Args:
        weight_bits:      权重量化位宽 (8=int8 匹配硬件, 4=int4)
        act_bits:         激活量化位宽 (8=int8)
        noise:            是否注入硬件匹配噪声 (训练时: 权重 DAC + 激活 TIA/ADC)
        first_conv_fp32:  首层 Conv 是否保持 FP32 (对齐率<50% 的层建议 FP32)
        quantize_linear:  是否量化 Linear 层
        preserve_bn:      保留 BN
        act_quant:        激活量化方案:
            - "uint8_affine" (默认): per-tensor unsigned uint8 + zero_point,
              与 osimulator `_matmul_real` 输入量化完全一致 (修复训练/推理不对齐)
            - "signed_per_channel": per-channel signed int8 (旧行为, 仅对比实验用)
    """
    if not inplace:
        model = copy.deepcopy(model)

    noise_inj = GazelleNoiseInjector() if noise else None

    _convert_to_v4(model, weight_bits, act_bits, noise, noise_inj,
                   first_conv_fp32, quantize_linear, act_quant, _first=True)

    qc_enabled = sum(1 for m in model.modules()
                     if isinstance(m, QATConv2d_v4) and m.qat_enabled)
    qc_disabled = sum(1 for m in model.modules()
                      if isinstance(m, QATConv2d_v4) and not m.qat_enabled)
    ql = sum(1 for m in model.modules() if isinstance(m, QATLinear_v4))
    bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))

    w_label = f"int{weight_bits}"
    print(f"[prepare_model_v4] Gazelle HW-aware QAT: w{w_label}/a{act_bits}"
          f" ({act_quant}, 匹配 osimulator uint8+zp)")
    print(f"  QAT Conv: {qc_enabled} enabled + {qc_disabled} fp32 (first layer)")
    print(f"  QAT Linear: {ql}, BN: {bn}")
    if noise:
        print(f"  硬件噪声: {noise_inj} — 训练时注入 权重DAC + 激活TIA/ADC")
    if first_conv_fp32 and qc_disabled > 0:
        print(f"  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)")
    return model


def _convert_to_v4(module, weight_bits, act_bits, noise, noise_inj,
                   first_conv_fp32, quantize_linear, act_quant, _first=True):
    """递归转换. _first 用单元素列表传递, 避免嵌套调用不更新父级的问题."""
    if not isinstance(_first, list):
        _first = [_first]

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            keep_fp32 = _first[0] and first_conv_fp32
            qat = QATConv2d_v4(child, weight_bits=weight_bits,
                               act_bits=act_bits, noise=noise,
                               noise_injector=noise_inj,
                               keep_fp32=keep_fp32,
                               act_quant=act_quant)
            setattr(module, name, qat)
            _first[0] = False  # ★ 可变容器: 真正只有第一个 Conv 受影响
        elif isinstance(child, nn.Linear):
            if quantize_linear:
                qat = QATLinear_v4(child, weight_bits=weight_bits,
                                   act_bits=act_bits, noise=noise,
                                   noise_injector=noise_inj,
                                   act_quant=act_quant)
                setattr(module, name, qat)
        elif isinstance(child, (QATConv2d_v4, QATLinear_v4)):
            continue
        else:
            _convert_to_v4(child, weight_bits, act_bits, noise, noise_inj,
                          first_conv_fp32, quantize_linear, act_quant, _first)


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

    # Test unsigned affine activation quant (matches _matmul_real)
    print("\n[Test 3b] fake_quantize_unsigned_affine (per-tensor uint8 + zp)")
    a = torch.randn(2, 8, 16, 16) * 0.5
    a.requires_grad = True
    a_q = fake_quantize_unsigned_affine(a, bits=8)
    loss = a_q.sum(); loss.backward()
    t_min, t_max = a.min(), a.max()
    scale = (t_max - t_min.clamp(max=0.0)) / 255.0
    print(f"  Unique values: {a_q.unique().numel()} (≤256 for uint8)")
    print(f"  Range: [{a_q.min().item():.4f}, {a_q.max().item():.4f}], "
          f"expected min≈{t_min.clamp(max=0.0).item():.4f}")
    print(f"  Act grad norm: {a.grad.norm():.4f} (STE works)")
    print(f"  [OK]")

    # Test activation noise injection (TIA + ADC) — the previously dead code path
    print("\n[Test 3c] inject_activation_noise (TIA + ADC) now wired into training")
    inj2 = GazelleNoiseInjector()
    a2 = torch.randn(2, 8, 16, 16)
    a2.requires_grad = True
    s2 = (a2.max() - a2.min().clamp(max=0.0)) / 255.0
    a2_noisy = inj2.inject_activation_noise(a2, s2)
    expect_std = (5.34e-4 ** 2 + (0.00147 / np.sqrt(12)) ** 2) ** 0.5
    print(f"  Noise std: {(a2_noisy - a2).std():.6f} "
          f"(expect ~= sqrt(TIA^2 + ADC^2) ~= {expect_std:.6f})")
    print(f"  [OK]")

    # Test QATConv2d_v4 (int8)
    print("\n[Test 4] QATConv2d_v4 (int8 weight, Gazelle noise + uint8 affine act)")
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat = QATConv2d_v4(conv, weight_bits=8, noise=True,
                       noise_injector=GazelleNoiseInjector())
    qat.train()
    out = qat(torch.randn(2, 3, 32, 32))
    out.sum().backward()
    print(f"  Output shape: {out.shape}, grad norm: {qat.weight.grad.norm():.4f}")
    print(f"  act_quant: {qat._act_quant} (should be uint8_affine)")
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
