"""
================================================================================
 optic_layers.py — 光计算推理核心库
================================================================================
 提供:
   - OpticalEngine:  光计算模拟器抽象层 (自动检测真实/模拟模式)
   - OpticConv2d:    光计算卷积层 (im2col → 光学矩阵乘法 → col2im)
   - OpticLinear:    光计算全连接层 (光学矩阵乘法)
   - build_optical_model(): 将标准模型转换为光计算版本
   - 噪声注入器:     GaussianReadoutNoise / PhaseNoise / ShotNoise / CrosstalkNoise

 参考 API (example_load_gazelle_model.py):
   from osimulator.api import load_gazelle_model
   model = load_gazelle_model()
   result = model(input_tensors, wght_tensors, inputType="uint4")
   # input_tensors: (b, m, k) int32, wght_tensors: (b, k, n) int32
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import time

# ============================================================
#  尝试加载真实光计算模拟器
# ============================================================
_HAS_REAL_OPTICAL = False
_load_gazelle_model = None
_entrance = None

try:
    import osimulator
    from osimulator.api import load_gazelle_model as _load_gazelle
    import entrance as _ent
    _load_gazelle_model = _load_gazelle
    _entrance = _ent
    _HAS_REAL_OPTICAL = True
    print("[optic_layers] [OK] Real optical simulator loaded (osimulator)")
except ImportError:
    print("[optic_layers] [WARN] osimulator not installed, using FakeOpticalEngine")


# ============================================================
#  量化工具函数
# ============================================================
def quantize_int4(tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    将浮点张量量化为 int4 (-8~7)，再反量化回浮点。
    模拟光计算的 4-bit 精度限制 (int4 是光计算通用精度标准)。

    Args:
        tensor: 浮点张量
        dim: 沿哪个维度做 per-{dim} 量化。
             None = per-tensor
             对 Conv 输入: dim=1 表示 per-channel 量化
             对 Conv 权重: dim=0 表示 per-output-channel 量化
             对 Linear 输入: dim=-1 表示 per-row 量化
             对 Linear 权重: dim=0 表示 per-output-channel 量化
    Returns:
        量化-反量化后的浮点张量
    """
    if dim is None:
        # Per-tensor: 对称量化到 [-8, 7]
        abs_max = max(tensor.abs().max(), 1e-8)
        scale = abs_max / 7.0
        q = (tensor / scale).round().clamp(-8, 7)
        return q * scale
    else:
        # Per-dimension quantization
        reduce_dims = [d for d in range(tensor.dim()) if d != dim]
        abs_max = tensor.abs()
        for d in sorted(reduce_dims, reverse=True):
            abs_max = abs_max.max(dim=d, keepdim=True)[0]
        abs_max = torch.where(abs_max < 1e-8, torch.ones_like(abs_max), abs_max)
        scale = abs_max / 7.0
        q = (tensor / scale).round().clamp(-8, 7)
        return q * scale


# quantize_uint4 保留向后兼容，但默认推荐 int4
def quantize_uint4(tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    [已废弃，推荐使用 quantize_int4]
    将浮点张量量化为 uint4 (0~15)，再反量化回浮点。
    """
    if dim is None:
        t_min = tensor.min()
        t_max = tensor.max()
        if t_max - t_min < 1e-8:
            return tensor
        scale = (t_max - t_min) / 15.0
        zero_point = t_min
        q = ((tensor - zero_point) / scale).round().clamp(0, 15)
        return q * scale + zero_point
    else:
        shape = tensor.shape
        reduce_dims = [d for d in range(tensor.dim()) if d != dim]
        t_min = tensor
        t_max = tensor
        for d in sorted(reduce_dims, reverse=True):
            t_min = t_min.min(dim=d, keepdim=True)[0]
            t_max = t_max.max(dim=d, keepdim=True)[0]
        scale = (t_max - t_min) / 15.0
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        zero_point = t_min
        q = ((tensor - zero_point) / scale).round().clamp(0, 15)
        return q * scale + zero_point


def quantize_to_int(tensor: torch.Tensor, bit: int, signed: bool) -> torch.Tensor:
    """
    将浮点张量化为指定 bit 宽度的整数值 (不反量化)。
    返回 int32 类型的量化值，可直接送入光计算模拟器。

    Args:
        tensor: 浮点张量
        bit: bit 宽度
        signed: True=int, False=uint
    Returns:
        (量化后的 int32 张量, scale, zero_point)
    """
    if signed:
        val_range = 2 ** (bit - 1)
        v_min, v_max = -val_range, val_range - 1
        abs_max = max(tensor.abs().max().item(), 1e-8)
        scale = abs_max / (val_range - 1)
        q = (tensor / scale).round().clamp(v_min, v_max).to(torch.int32)
        return q, scale, 0.0
    else:
        val_range = 2 ** bit
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        if t_max - t_min < 1e-8:
            return torch.zeros_like(tensor, dtype=torch.int32), 1.0, t_min
        scale = (t_max - t_min) / (val_range - 1)
        zero_point = t_min
        q = ((tensor - zero_point) / scale).round().clamp(0, val_range - 1).to(torch.int32)
        return q, scale, zero_point


# ============================================================
#  噪声注入器基类与实现
# ============================================================
class NoiseInjector(ABC):
    """噪声注入器抽象基类"""

    def __init__(self, level: float = 0.0):
        self.level = level

    def set_level(self, level: float):
        self.level = level

    @abstractmethod
    def inject(self, output: torch.Tensor,
               input_raw: torch.Tensor = None,
               weight_raw: torch.Tensor = None) -> torch.Tensor:
        """注入噪声并返回带噪输出"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(level={self.level:.4f})"


class GaussianReadoutNoise(NoiseInjector):
    """
    高斯读出噪声: 在光计算输出端叠加加性高斯白噪声。
    output_noisy = output + level * std(output) * N(0, 1)

    模拟: 光子探测器读出电路的热噪声 / ADC 量化前噪声
    """

    def inject(self, output: torch.Tensor,
               input_raw: torch.Tensor = None,
               weight_raw: torch.Tensor = None) -> torch.Tensor:
        if self.level <= 0:
            return output
        noise_std = self.level * output.std()
        noise = torch.randn_like(output) * noise_std
        return output + noise


class PhaseNoise(NoiseInjector):
    """
    相位噪声: 在权重矩阵上叠加随机扰动，模拟 MZI 相位误差。
    W_noisy = W + level * max_abs(W) * U(-1, 1)

    这是光计算中最主要的物理噪声源之一。
    注意: 此噪声在 matmul 之前注入到权重上。
    """

    def inject(self, output: torch.Tensor,
               input_raw: torch.Tensor = None,
               weight_raw: torch.Tensor = None) -> torch.Tensor:
        if self.level <= 0 or weight_raw is None:
            return output
        # 在权重上注入噪声后重新计算
        w_abs_max = weight_raw.abs().max()
        noise = (torch.rand_like(weight_raw) * 2 - 1) * self.level * w_abs_max
        w_noisy = weight_raw + noise
        # 重新做 matmul
        return input_raw @ w_noisy if input_raw is not None else output


class ShotNoise(NoiseInjector):
    """
    散粒噪声 (光子计数统计):
    将输出视为光子计数期望值，用 Poisson 分布采样。
    level 控制有效光子数的衰减程度。

    photon_count = output_scaled * (1 - level)
    output_noisy = Poisson(photon_count) / scale
    """

    def inject(self, output: torch.Tensor,
               input_raw: torch.Tensor = None,
               weight_raw: torch.Tensor = None) -> torch.Tensor:
        if self.level <= 0:
            return output

        # 将输出偏移到非负区域
        out_min = output.min()
        output_shifted = output - out_min + 1e-8

        # 光子数缩放: level 越高，光子越少，噪声越大
        max_photons = 1000.0
        effective_photons = max_photons * (1.0 - self.level) + 10.0 * self.level
        scale = effective_photons / output_shifted.mean()

        expected = output_shifted * scale
        expected = expected.clamp(min=0)

        # Poisson 采样
        noisy = torch.poisson(expected).float() / scale
        return noisy + out_min - 1e-8


class CrosstalkNoise(NoiseInjector):
    """
    通道串扰: 相邻输出通道之间信号泄漏。
    output[i] += level * (output[i-1] + output[i+1]) / 2

    模拟: 波导阵列中的 evanescent coupling / 电学互连串扰
    """

    def inject(self, output: torch.Tensor,
               input_raw: torch.Tensor = None,
               weight_raw: torch.Tensor = None) -> torch.Tensor:
        if self.level <= 0:
            return output

        # output shape: (..., channels) — 在最后一维做串扰
        result = output.clone()
        n = output.shape[-1]

        # 左邻居贡献
        left = torch.zeros_like(output)
        left[..., 1:] = output[..., :n-1]

        # 右邻居贡献
        right = torch.zeros_like(output)
        right[..., :n-1] = output[..., 1:]

        crosstalk = (left + right) / 2.0
        result = result + self.level * crosstalk

        return result


# ============================================================
#  光计算引擎
# ============================================================
class OpticalEngine:
    """
    光计算矩阵乘法引擎。

    自动检测 Ltsimulator 可用性:
      - 可用:  调用真实 osimulator API
      - 不可用: 使用 FakeOpticalEngine (量化+浮点 matmul)

    API 约定 (与 Ltsimulator 一致):
      matmul(input, weight) 其中:
        input:  (b, m, k) — 输入矩阵，将被量化为 uint4
        weight: (k, n)   — 权重矩阵，将被量化为 int4
        output: (b, m, n) — 输出矩阵
    """

    def __init__(self, use_real: bool = True):
        self.use_real = use_real and _HAS_REAL_OPTICAL
        self._real_model = None
        self.noise_injector: NoiseInjector = None
        self.stats = {"calls": 0, "total_time": 0.0, "total_ops": 0}

        if self.use_real:
            self._real_model = _load_gazelle_model()
            print("[OpticalEngine] Using real optical simulator")
        else:
            if not _HAS_REAL_OPTICAL and use_real:
                print("[OpticalEngine] Real simulator unavailable, fallback to fake mode")
            else:
                print("[OpticalEngine] Using FakeOptical mode")

    def matmul(self,
               input_matrix: torch.Tensor,    # (m, k) or (b, m, k)
               weight_matrix: torch.Tensor,   # (k, n)
               input_bit: int = 4,
               weight_bit: int = 4,
               output_bit: int = 12,
               quantize_inputs: bool = True) -> torch.Tensor:
        """
        光学矩阵乘法。

        Args:
            input_matrix:  (m, k) 或 (b, m, k) 浮点输入
            weight_matrix: (k, n) 浮点权重
            input_bit:     输入量化 bit 数 (默认 4 = uint4)
            weight_bit:    权重量化 bit 数 (默认 4 = int4)
            output_bit:    输出 bit 数 (默认 12 = int12, 用于统计)
            quantize_inputs: 是否在此方法内量化。如果调用方已量化，设为 False。
        Returns:
            (m, n) 或 (b, m, n) 浮点输出
        """
        t0 = time.time()

        # 处理维度: 统一为 (b, m, k) @ (k, n) -> (b, m, n)
        squeeze_batch = False
        if input_matrix.dim() == 2:
            input_matrix = input_matrix.unsqueeze(0)  # (1, m, k)
            squeeze_batch = True

        b, m, k_in = input_matrix.shape
        k_w, n = weight_matrix.shape

        if k_in != k_w:
            raise ValueError(f"Matrix dimension mismatch: input k={k_in}, weight k={k_w}")

        total_ops = b * m * k_in * n
        self.stats["total_ops"] += total_ops

        if self.use_real:
            result = self._matmul_real(input_matrix, weight_matrix,
                                       input_bit, weight_bit)
        else:
            result = self._matmul_fake(input_matrix, weight_matrix,
                                       input_bit, weight_bit,
                                       quantize_inputs=quantize_inputs)

        # Apply noise injector (post-processing types)
        if self.noise_injector is not None:
            if not isinstance(self.noise_injector, PhaseNoise):
                result = self.noise_injector.inject(
                    result,
                    input_raw=input_matrix,
                    weight_raw=weight_matrix
                )

        if squeeze_batch:
            result = result.squeeze(0)

        elapsed = time.time() - t0
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed

        return result

    def _matmul_fake(self, input_matrix: torch.Tensor,
                     weight_matrix: torch.Tensor,
                     input_bit: int, weight_bit: int,
                     quantize_inputs: bool = True) -> torch.Tensor:
        """Simulate optical computing: (optional) quantize + float matmul"""
        b, m, k = input_matrix.shape
        k_w, n = weight_matrix.shape

        if quantize_inputs:
            # Quantize inputs and weights to int4 (optical computing standard)
            input_q = quantize_int4(input_matrix.reshape(-1, k)).reshape(b, m, k)
            weight_q = quantize_int4(weight_matrix)
        else:
            # Data already quantized by the calling layer
            input_q = input_matrix
            weight_q = weight_matrix

        # Phase noise: inject before matmul (perturbs weights)
        if self.noise_injector is not None and isinstance(self.noise_injector, PhaseNoise):
            w_abs_max = weight_q.abs().max()
            if w_abs_max > 0:
                weight_q = weight_q + self.noise_injector.level * w_abs_max * \
                           (torch.rand_like(weight_q) * 2 - 1)

        # Execute matrix multiplication
        result = torch.bmm(input_q, weight_q.unsqueeze(0).expand(b, k, n))

        return result

    def _matmul_real(self, input_matrix: torch.Tensor,
                     weight_matrix: torch.Tensor,
                     input_bit: int, weight_bit: int) -> torch.Tensor:
        """调用真实 Ltsimulator"""
        b, m, k = input_matrix.shape
        k_w, n = weight_matrix.shape

        # 量化到整数
        input_int, in_scale, in_zp = quantize_to_int(
            input_matrix.reshape(-1, k), input_bit, signed=False
        )
        input_int = input_int.reshape(b, m, k)

        weight_int, w_scale, _ = quantize_to_int(
            weight_matrix, weight_bit, signed=True
        )

        # 调用光模拟器
        input_np = input_int.cpu().numpy().astype(np.int32)
        weight_np = weight_int.cpu().numpy().astype(np.int32)
        # 权重需要 batch 维度: (1, k, n) → (b, k, n)
        weight_np = np.tile(weight_np[None, :, :], (b, 1, 1))

        input_type = f"int{input_bit}"
        result_np = self._real_model(input_np, weight_np, inputType=input_type)

        # 反量化
        result = torch.from_numpy(result_np).float()
        result = result * (in_scale * w_scale)

        return result

    def set_noise(self, injector: NoiseInjector):
        """设置噪声注入器"""
        self.noise_injector = injector

    def clear_noise(self):
        """清除噪声注入器"""
        self.noise_injector = None

    def print_stats(self):
        """打印统计信息"""
        print(f"  [OpticalEngine 统计] 调用: {self.stats['calls']}, "
              f"总耗时: {self.stats['total_time']:.3f}s, "
              f"总运算量: {self.stats['total_ops']:.2e} MACs")

    def reset_stats(self):
        """重置统计"""
        self.stats = {"calls": 0, "total_time": 0.0, "total_ops": 0}


# ============================================================
#  光计算卷积层
# ============================================================
class OpticConv2d(nn.Module):
    """
    光计算 2D 卷积层。

    将标准卷积的矩阵乘法替换为光计算:
      1. im2col (F.unfold): 将滑窗展开为大矩阵
      2. 光学矩阵乘法:  (patches) @ (flattened_weights)
      3. col2im:        将结果折叠回特征图

    硬件对齐:
      - 展平长度 C_in × k_h × k_w 需要被 8 整除
      - 不能整除时自动补零 (模拟光计算硬件上的 waste)
    """

    def __init__(self, conv_layer: nn.Conv2d, engine: OpticalEngine,
                 pad_to_8: bool = True):
        super().__init__()

        # 复制卷积参数
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        if self.groups > 1:
            raise NotImplementedError(
                "OpticConv2d 暂不支持分组卷积 (groups > 1)"
            )

        # 复制权重和 bias
        self.weight = nn.Parameter(conv_layer.weight.data.clone())
        if conv_layer.bias is not None:
            self.bias = nn.Parameter(conv_layer.bias.data.clone())
        else:
            self.bias = None

        self.engine = engine
        self.pad_to_8 = pad_to_8

        # 计算硬件对齐信息
        self._patch_len = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self._padded_len = ((self._patch_len + 7) // 8) * 8 if pad_to_8 else self._patch_len
        self._alignment_ratio = self._patch_len / self._padded_len if self._padded_len > 0 else 1.0

    @property
    def alignment_ratio(self) -> float:
        """该层的硬件对齐率"""
        return self._alignment_ratio

    @property
    def patch_len(self) -> int:
        return self._patch_len

    @property
    def padded_len(self) -> int:
        return self._padded_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C_in, H, W)
        returns: (N, C_out, OH, OW)
        """
        N, C, H, W = x.shape
        kh, kw = self.kernel_size

        # 计算输出空间尺寸
        OH = (H + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        OW = (W + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        L = OH * OW

        # === 量化输入特征图 (per-channel int4) ===
        # 光计算通用精度标准: int4 (对称量化, [-8, 7])
        x_q = quantize_int4(x, dim=1)  # per-channel: (N, C, H, W)

        # === 量化权重 (per-output-channel int4) ===
        w_q = quantize_int4(self.weight, dim=0)  # per-OC: (OC, C, kh, kw)

        # 1. im2col: (N, C, H, W) -> (N, C*kh*kw, OH*OW)
        x_unfold = F.unfold(
            x_q, kernel_size=(kh, kw),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        # 2. 转换为 matmul 格式: (N, C*k*k, L) -> (N, L, C*k*k) -> (N*L, C*k*k)
        x_mat = x_unfold.transpose(1, 2).reshape(N * L, C * kh * kw)

        # 3. 权重重塑: (C_out, C_in, kh, kw) -> (C_in*kh*kw, C_out)
        w_mat = w_q.reshape(self.out_channels, -1).t()

        # 4. 硬件对齐补零
        if self.pad_to_8 and self._padded_len > self._patch_len:
            pad_amount = self._padded_len - self._patch_len
            x_mat = F.pad(x_mat, (0, pad_amount), value=0.0)
            w_mat = F.pad(w_mat, (0, 0, 0, pad_amount), value=0.0)

        # 5. 光学矩阵乘法 (数据已量化，直接浮点 matmul)
        # 使用 b=1, m=N*L 格式
        result = self.engine.matmul(x_mat, w_mat, quantize_inputs=False)
        # result: (N*L, C_out)

        # 6. col2im: (N*L, C_out) -> (N, L, C_out) -> (N, C_out, OH, OW)
        result = result.reshape(N, L, self.out_channels)
        result = result.transpose(1, 2).reshape(N, self.out_channels, OH, OW)

        # 7. 加 bias
        if self.bias is not None:
            result = result + self.bias.view(1, -1, 1, 1)

        return result

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, "
                f"alignment={self._alignment_ratio:.1%} "
                f"({self._patch_len}→{self._padded_len})")


# ============================================================
#  光计算全连接层
# ============================================================
class OpticLinear(nn.Module):
    """
    光计算全连接层。

    将标准 Linear 的矩阵乘法替换为光计算:
      input:  (N, in_features)
      weight: (in_features, out_features)
      output = optical_matmul(input, weight)
    """

    def __init__(self, linear_layer: nn.Linear, engine: OpticalEngine,
                 pad_to_8: bool = True):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # 复制权重和 bias
        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.bias = None

        self.engine = engine
        self.pad_to_8 = pad_to_8

        # 对齐信息
        self._patch_len = self.in_features
        self._padded_len = ((self._patch_len + 7) // 8) * 8 if pad_to_8 else self._patch_len
        self._alignment_ratio = self._patch_len / self._padded_len if self._padded_len > 0 else 1.0

    @property
    def alignment_ratio(self) -> float:
        return self._alignment_ratio

    @property
    def patch_len(self) -> int:
        return self._patch_len

    @property
    def padded_len(self) -> int:
        return self._padded_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features)
        returns: (N, out_features)
        """
        N = x.shape[0]

        # === 量化输入 (per-sample/row int4) ===
        x_q = quantize_int4(x, dim=-1)  # per-row: (N, in_features)

        # === 量化权重 (per-output-channel int4) ===
        w_q = quantize_int4(self.weight, dim=0)  # per-OC: (out_features, in_features)

        # 1. 输入: (N, in_features) -> (N, 1, in_features)  [b=N, m=1, k=in_features]
        x_mat = x_q.unsqueeze(1)

        # 2. 权重: (out_features, in_features) -> t() -> (in_features, out_features)
        w_mat = w_q.t()

        # 3. 硬件对齐补零
        if self.pad_to_8 and self._padded_len > self._patch_len:
            pad_amount = self._padded_len - self._patch_len
            x_mat = F.pad(x_mat, (0, pad_amount), value=0.0)
            w_mat = F.pad(w_mat, (0, 0, 0, pad_amount), value=0.0)

        # 4. 光学矩阵乘法 (数据已量化)
        result = self.engine.matmul(x_mat, w_mat, quantize_inputs=False)
        # result: (N, 1, out_features)

        # 5. 还原形状
        result = result.squeeze(1)  # (N, out_features)

        # 6. 加 bias
        if self.bias is not None:
            result = result + self.bias

        return result

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"alignment={self._alignment_ratio:.1%} "
                f"({self._patch_len}→{self._padded_len})")


# ============================================================
#  模型转换工厂
# ============================================================
def build_optical_model(original_model: nn.Module, engine: OpticalEngine,
                        pad_to_8: bool = True) -> nn.Module:
    """
    递归遍历模型，将所有 nn.Conv2d 替换为 OpticConv2d，
    将所有 nn.Linear 替换为 OpticLinear。

    BatchNorm, ReLU, MaxPool, Dropout, Flatten 等保持原样。

    Args:
        original_model: 已加载训练权重的原始模型
        engine:         光计算引擎
        pad_to_8:       是否补零到 8 的倍数
    Returns:
        光计算版本的模型 (原地修改)
    """
    replacements = {
        nn.Conv2d: lambda m: OpticConv2d(m, engine, pad_to_8=pad_to_8),
        nn.Linear: lambda m: OpticLinear(m, engine, pad_to_8=pad_to_8),
    }

    _replace_modules(original_model, replacements)
    return original_model


def _replace_modules(module: nn.Module, replacements: dict,
                     _prefix: str = ""):
    """递归替换模块"""
    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name

        # 检查是否需要替换
        replaced = False
        for src_type, factory in replacements.items():
            if isinstance(child, src_type):
                setattr(module, name, factory(child))
                replaced = True
                break

        # 递归处理子模块 (包括新替换的)
        if not replaced:
            _replace_modules(child, replacements, full_name)


# ============================================================
#  对齐率计算工具
# ============================================================
def compute_alignment_ratio(model: nn.Module) -> float:
    """
    计算模型在 8×2 光计算硬件上的综合对齐率。
    对 OpticConv2d / OpticLinear: 直接读取 alignment_ratio
    对普通 Conv2d / Linear: 手动计算
    """
    total_patch = 0
    total_padded = 0

    for m in model.modules():
        if isinstance(m, (OpticConv2d, OpticLinear)):
            total_patch += m.patch_len
            total_padded += m.padded_len
        elif isinstance(m, nn.Conv2d):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
        elif isinstance(m, nn.Linear):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded

    return total_patch / total_padded if total_padded > 0 else 0.0


def print_alignment_detail(model: nn.Module, label: str = ""):
    """打印每层对齐详情"""
    header = f"  {label}" if label else ""
    print(f"\n{header} 层名                          C_in   K      展平长度  补零后  对齐率")
    print("  " + "-" * 72)
    total_patch, total_padded = 0, 0

    for name, m in model.named_modules():
        if isinstance(m, OpticConv2d):
            patch = m.patch_len
            padded = m.padded_len
            total_patch += patch
            total_padded += padded
            print(f"  [OpticConv2d] {name:<25s} {m.in_channels:>4d}   {m.kernel_size[0]}×{m.kernel_size[1]}"
                  f"   {patch:>8d}   {padded:>8d}   {m.alignment_ratio:.1%}")
        elif isinstance(m, OpticLinear):
            patch = m.patch_len
            padded = m.padded_len
            total_patch += patch
            total_padded += padded
            print(f"  [OpticLinear] {name:<25s}  —     —    "
                  f"   {patch:>8d}   {padded:>8d}   {m.alignment_ratio:.1%}")
        elif isinstance(m, nn.Conv2d):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            print(f"  [Conv2d]      {name:<25s} {m.in_channels:>4d}   {m.kernel_size[0]}×{m.kernel_size[1]}"
                  f"   {patch:>8d}   {padded:>8d}   {patch/padded:.1%}")
        elif isinstance(m, nn.Linear):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            print(f"  [Linear]      {name:<25s}  —     —    "
                  f"   {patch:>8d}   {padded:>8d}   {patch/padded:.1%}")

    overall = total_patch / total_padded if total_padded > 0 else 0
    print(f"  综合硬件对齐率: {overall:.1%} (总展平 {total_patch} → 补零后 {total_padded})")
    return overall


# ============================================================
#  评估工具
# ============================================================
@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader, device: torch.device,
                   criterion=None) -> dict:
    """
    评估模型准确率和损失。

    Returns:
        {"accuracy": float, "loss": float, "total_samples": int, "correct": int}
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        if criterion is not None:
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {
        "accuracy": correct / total,
        "loss": total_loss / total if criterion else 0.0,
        "total_samples": total,
        "correct": correct,
    }


if __name__ == "__main__":
    # 快速自测
    print("=" * 60)
    print("  optic_layers.py - Module Self-Test")
    print("=" * 60)

    # 测试引擎
    engine = OpticalEngine()

    # 测试基本 matmul
    x = torch.randn(1, 16, 32)
    w = torch.randn(32, 8)
    y = engine.matmul(x, w)
    print(f"  Basic matmul: {x.shape} @ {w.shape} -> {y.shape} [OK]")

    # 测试 OpticLinear
    linear = nn.Linear(64, 10)
    optic_linear = OpticLinear(linear, engine)
    test_input = torch.randn(4, 64)
    out_native = linear(test_input)
    out_optic = optic_linear(test_input)
    print(f"  OpticLinear: {test_input.shape} -> {out_optic.shape} [OK]")
    print(f"    Native vs Optic diff: {(out_native - out_optic).abs().mean():.6f}")

    # 测试 OpticConv2d
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    optic_conv = OpticConv2d(conv, engine)
    test_img = torch.randn(2, 3, 32, 32)
    out_native = conv(test_img)
    out_optic = optic_conv(test_img)
    print(f"  OpticConv2d (3x3): {test_img.shape} -> {out_optic.shape} [OK]")
    print(f"    Alignment: {optic_conv.alignment_ratio:.1%} ({optic_conv.patch_len}->{optic_conv.padded_len})")
    print(f"    Native vs Optic diff: {(out_native - out_optic).abs().mean():.6f}")

    # 测试硬件对齐的 conv
    conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
    optic_conv2 = OpticConv2d(conv2, engine)
    test_img2 = torch.randn(2, 8, 64, 64)
    out_native2 = conv2(test_img2)
    out_optic2 = optic_conv2(test_img2)
    print(f"  OpticConv2d (2x2, aligned): {test_img2.shape} -> {out_optic2.shape} [OK]")
    print(f"    Alignment: {optic_conv2.alignment_ratio:.1%} ({optic_conv2.patch_len}->{optic_conv2.padded_len})")

    # 测试噪声注入
    engine.set_noise(GaussianReadoutNoise(level=0.1))
    y_noisy = engine.matmul(x, w)
    print(f"  Gaussian noise matmul: y_std={y.std():.4f}, y_noisy_std={y_noisy.std():.4f} [OK]")
    engine.clear_noise()

    # 测试模型转换
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(8 * 32 * 32, 10)

        def forward(self, x):
            x = self.relu(self.bn(self.conv(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    dummy = DummyModel()
    build_optical_model(dummy, engine)
    print(f"  After conversion: conv={type(dummy.conv).__name__}, "
          f"bn={type(dummy.bn).__name__}, fc={type(dummy.fc).__name__} [OK]")

    print_alignment_detail(dummy, "DummyModel")
    print("\n  All self-tests passed! [OK]")
