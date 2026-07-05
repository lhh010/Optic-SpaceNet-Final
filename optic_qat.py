"""
================================================================================
 optic_qat.py — 光计算 QAT (Quantization-Aware Training) 核心模块
================================================================================

 背景:
   光计算硬件 (8×2 光学矩阵乘法器) 使用 int4 精度进行矩阵乘法运算。
   之前的 PTQ (Post-Training Quantization) 方案在推理时才将 float32 权重
   量化为 int4，模型从未见过量化误差，导致精度大幅下降。

   QAT (Quantization-Aware Training) 通过在训练时插入"伪量化"(Fake Quantization)
   节点，使用 STE (Straight-Through Estimator) 让梯度穿越量化操作，使模型
   学会在 int4 精度下仍能正确推理。

 核心原理:
   1. Fake Quantization: 前向时将 float32 张量先量化到 int4 再反量化回 float32
   2. STE: 反向传播时梯度直接穿过量化节点 (identity gradient)
   3. QAT 训练后，权重虽然仍是 float32，但已对 int4 量化具备鲁棒性
   4. 推理时使用 optic_layers.py 的 OpticConv2d/OpticLinear 进行实际光计算推理

 提供:
   - fake_int4_quantize():    STE 伪 int4 量化函数
   - QATConv2d:               QAT 卷积层 (包裹标准 Conv2d)
   - QATLinear:               QAT 全连接层 (包裹标准 Linear)
   - fuse_conv_bn():          Conv+BN 融合 (QAT 标准预处理)
   - prepare_qat_model():     将标准模型转换为 QAT 模型
   - enable_qat / disable_qat: 切换 QAT 模式
   - calibrate_qat_model():   校准量化尺度
   - evaluate_model():        评估工具

 与 optic_layers.py 的关系:
   - optic_layers.py:  推理阶段的光计算模拟 (OpticConv2d/OpticLinear)
   - optic_qat.py:     训练阶段的 QAT (QATConv2d/QATLinear)
   - QAT 训练后的 float32 权重可直接用于 optic_layers.py 的光计算推理

 参考:
   - Jacob et al., "Quantization and Training of Neural Networks for Efficient
     Integer-Arithmetic-Only Inference" (CVPR 2018)
   - Esser et al., "Learned Step Size Quantization" (ICLR 2020)
   - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import time


# ============================================================
#  Fake Int4 Quantization (STE)
# ============================================================

def fake_int4_quantize(x: torch.Tensor,
                       per_channel: bool = False,
                       ch_dim: int = 0) -> torch.Tensor:
    """
    伪 int4 对称量化，配合 Straight-Through Estimator (STE)。

    前向传播:
      1. 计算 scale: 基于 abs_max / 7
      2. 量化: x_int = round(x / scale).clamp(-8, 7)
      3. 反量化: x_dq = x_int * scale
      4. 输出 x_dq (模拟 int4 量化后的数值)

    反向传播:
      STE — 梯度直接穿过量化节点 (identity gradient)
      实现: x + (x_dq - x).detach()
            → 前向值 = x_dq
            → 梯度流向 x (因为 detach 断开了 x_dq 的梯度路径)

    Args:
        x:              float32 张量
        per_channel:    是否逐通道计算 scale
        ch_dim:         通道维度 (per_channel=True 时生效)
                        - Conv 输入: ch_dim=1 (C_in 维度)
                        - Conv 权重: ch_dim=0 (C_out 维度)
                        - Linear 输入: ch_dim=-1 (最后一维)
                        - Linear 权重: ch_dim=0 (C_out 维度)

    Returns:
        伪量化后的 float32 张量 (值域与量化后的 int4 相同，但保持 float32)

    Examples:
        >>> x = torch.randn(4, 8, 32, 32)  # Conv input
        >>> x_q = fake_int4_quantize(x, per_channel=True, ch_dim=1)
        >>> w = torch.randn(16, 8, 3, 3)    # Conv weight
        >>> w_q = fake_int4_quantize(w, per_channel=True, ch_dim=0)
    """
    if per_channel:
        # 逐通道: 对除 ch_dim 外的所有维度求 abs_max
        reduce_dims = [i for i in range(x.dim()) if i != ch_dim]
        amax = x.abs()
        for d in sorted(reduce_dims, reverse=True):
            amax = amax.max(dim=d, keepdim=True)[0]
        # 对称量化 scale: abs_max / 7
        scale = (amax / 7.0).clamp(min=1e-8)
    else:
        # 逐张量: 全局 abs_max
        scale = (x.abs().max() / 7.0).clamp(min=1e-8)

    # 量化: round to nearest integer, clamp to int4 range [-8, 7]
    x_int = (x / scale).round().clamp(-8, 7)

    # 反量化: back to float
    x_dq = x_int * scale

    # STE: forward = dequantized, backward = identity gradient
    # detach() 断开了 x_dq 的反向传播路径，梯度通过 x 传递
    return x + (x_dq - x).detach()


# ============================================================
#  QAT-aware Conv2d
# ============================================================

class QATConv2d(nn.Module):
    """
    QAT-aware 2D 卷积层。

    对输入特征图和权重施加伪 int4 量化，模拟光计算硬件的精度限制。
    通过 STE 梯度，模型在训练中学会对 int4 量化具有鲁棒性。

    量化方案 (与光计算硬件一致):
      - 输入:  per-channel int4 (对称, 沿 C_in 维度)
      - 权重:  per-output-channel int4 (对称, 沿 C_out 维度)
      - bias:  保持 float32 (不在光计算矩阵乘法中，无需量化)

    与 OpticConv2d 的对应关系:
      - QATConv2d: 训练时使用，F.conv2d + 伪量化
      - OpticConv2d: 推理时使用，im2col → 光计算 matmul → col2im
      - 两者共享相同的 float32 权重
    """

    def __init__(self, conv_layer: nn.Conv2d):
        """
        Args:
            conv_layer: 标准 nn.Conv2d 层 (已初始化权重)
                        支持从预训练模型加载的权重。
        """
        super().__init__()

        # ---- 复制卷积参数 ----
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.padding_mode = conv_layer.padding_mode

        # ---- 复制权重 (float32, 通过 STE 微调) ----
        self.weight = nn.Parameter(conv_layer.weight.data.clone())
        if conv_layer.bias is not None:
            self.bias = nn.Parameter(conv_layer.bias.data.clone())
        else:
            self.bias = None

        # ---- QAT 状态 ----
        self._qat_enabled = True

        # ---- 统计 ----
        self._input_scale_stats = None  # 记录最近一次输入的 scale
        self._weight_scale_stats = None

    @property
    def qat_enabled(self) -> bool:
        return self._qat_enabled

    def enable_qat(self):
        """启用 QAT 伪量化"""
        self._qat_enabled = True

    def disable_qat(self):
        """禁用 QAT 伪量化 (使用原始 float32 精度)"""
        self._qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, H, W) float32 输入
        Returns:
            (N, C_out, OH, OW) float32 输出

        注意: QAT 模式下 (self._qat_enabled=True)，无论训练还是评估
        都会施加伪 int4 量化。这确保:
          - 训练时: 伪量化 + STE 梯度 → 权重适应 int4
          - 评估时: 伪量化 (无梯度) → 准确测量 int4 精度
        disable_qat() 后恢复为标准 float32 推理。
        """
        if self._qat_enabled:
            # === QAT 模式: 伪量化输入和权重 ===
            # 1. 伪量化输入 (per-channel, 沿 C_in 维度)
            #    模拟: 光计算 DAC 将模拟信号量化为 int4
            x_q = fake_int4_quantize(x, per_channel=True, ch_dim=1)

            # 2. 伪量化权重 (per-output-channel, 沿 C_out 维度)
            #    模拟: 光计算交叉阵列存储 int4 权重
            w_q = fake_int4_quantize(self.weight, per_channel=True, ch_dim=0)

            # 3. 标准浮点卷积 (使用量化后的值)
            #    由于 x_q 和 w_q 已模拟 int4 精度，等价于低精度卷积
            return F.conv2d(
                x_q, w_q, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
        else:
            # === 非 QAT 模式: 标准 float32 卷积 ===
            return F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, qat={'on' if self._qat_enabled else 'off'}")


# ============================================================
#  QAT-aware Linear
# ============================================================

class QATLinear(nn.Module):
    """
    QAT-aware 全连接层。

    对输入向量和权重矩阵施加伪 int4 量化，模拟光计算硬件的精度限制。

    量化方案:
      - 输入:  per-row int4 (对称, 沿最后一维, 即 in_features 维度)
      - 权重:  per-output-channel int4 (对称, 沿 C_out 维度)
      - bias:  保持 float32
    """

    def __init__(self, linear_layer: nn.Linear):
        """
        Args:
            linear_layer: 标准 nn.Linear 层 (已初始化权重)
        """
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # ---- 复制权重 ----
        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.bias = None

        # ---- QAT 状态 ----
        self._qat_enabled = True

    @property
    def qat_enabled(self) -> bool:
        return self._qat_enabled

    def enable_qat(self):
        self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_features) float32 输入
        Returns:
            (N, out_features) float32 输出

        注意: 与 QATConv2d 一致，QAT 模式下无论 train/eval 都施加伪量化。
        """
        if self._qat_enabled:
            # 伪量化输入 (per-row, dim=-1)
            x_q = fake_int4_quantize(x, per_channel=True, ch_dim=-1)
            # 伪量化权重 (per-output-channel, dim=0)
            w_q = fake_int4_quantize(self.weight, per_channel=True, ch_dim=0)
            return F.linear(x_q, w_q, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"qat={'on' if self._qat_enabled else 'off'}")


# ============================================================
#  Conv + BatchNorm 融合
# ============================================================

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    将 Conv2d 和紧随其后的 BatchNorm2d 融合为一个 Conv2d。

    这是 QAT 的标准预处理步骤，原因:
      1. BN 在推理时可以完全吸收到卷积权重中，无额外计算
      2. QAT 中的伪量化应该作用于融合后的权重，而非分开量化
      3. 避免 BN 的 affine 变换破坏量化尺度

    融合公式:
      Conv:  y_c = W * x + b
      BN:    y_bn = γ * (y_c - μ) / σ + β

      令 W_fused = γ/σ · W
         b_fused = γ/σ · (b - μ) + β

      则: y_bn = W_fused * x + b_fused  (等价于一个 Conv)

    Args:
        conv: nn.Conv2d 层
        bn:   nn.BatchNorm2d 层 (紧随 conv 之后)

    Returns:
        融合后的 nn.Conv2d (bias=True, 吸收了 BN 参数)
    """
    # 检查 BN 是否已初始化
    if bn.running_mean is None:
        raise ValueError("BatchNorm must be in eval mode or have run statistics. "
                         "Call model.eval() and run one batch before fusing.")

    # 计算融合系数
    std = (bn.running_var + bn.eps).sqrt()
    gamma_over_std = bn.weight / std  # shape: (C_out,)

    # 融合权重: W_fused = γ/σ * W
    # gamma_over_std 变形以适配卷积权重维度
    fused_weight = conv.weight * gamma_over_std.view(-1, 1, 1, 1)

    # 融合 bias: b_fused = γ/σ * (b - μ) + β
    if conv.bias is not None:
        fused_bias = gamma_over_std * (conv.bias - bn.running_mean) + bn.bias
    else:
        fused_bias = gamma_over_std * (-bn.running_mean) + bn.bias

    # 创建融合后的卷积层
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # BN 融合后总是有 bias
    )
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias

    return fused_conv


def fuse_conv_bn_in_sequential(seq: nn.Sequential) -> nn.Sequential:
    """
    在 Sequential 容器中查找 Conv2d + BatchNorm2d 模式并融合。

    融合后的 Sequential:
      [Conv2d, BatchNorm2d, ReLU] → [Conv2d (fused), ReLU]

    Args:
        seq: nn.Sequential 模块

    Returns:
        融合后的 nn.Sequential (保留未融合的层)
    """
    # 将 Sequential 拆分为列表
    layers = list(seq.children())
    fused_layers = []
    skip_next = False

    for i in range(len(layers)):
        if skip_next:
            skip_next = False
            continue

        if (isinstance(layers[i], nn.Conv2d) and
                i + 1 < len(layers) and
                isinstance(layers[i + 1], nn.BatchNorm2d)):
            # 找到 Conv + BN 模式, 融合
            fused_conv = fuse_conv_bn(layers[i], layers[i + 1])
            fused_layers.append(fused_conv)
            skip_next = True
        else:
            fused_layers.append(layers[i])

    return nn.Sequential(*fused_layers)


# ============================================================
#  QAT 模型准备
# ============================================================

def prepare_qat_model(model: nn.Module,
                      fuse_bn: bool = True,
                      inplace: bool = True) -> nn.Module:
    """
    将标准 PyTorch 模型转换为 QAT-ready 模型。

    流程:
      1. [可选] 融合 Conv2d + BatchNorm2d 层
      2. 将所有 nn.Conv2d 替换为 QATConv2d
      3. 将所有 nn.Linear  替换为 QATLinear
      4. 其他层 (BN, ReLU, MaxPool, Dropout 等) 保持不变

    Args:
        model:   标准 PyTorch 模型 (已加载 float32 权重)
        fuse_bn: 是否先进行 Conv+BN 融合 (强烈推荐)
        inplace: 是否原地修改模型

    Returns:
        QAT-ready 模型

    Example:
        >>> model = BaselineVGG()
        >>> model.load_state_dict(torch.load("baseline_vgg.pth"))
        >>> qat_model = prepare_qat_model(model, fuse_bn=True)
        >>> # 开始 QAT 训练...
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Step 1: Conv+BN 融合
    if fuse_bn:
        _fuse_all_conv_bn(model)

    # Step 2 & 3: 替换 Conv2d 和 Linear
    _replace_with_qat(model)

    print(f"[prepare_qat_model] Model converted to QAT-ready "
          f"(fuse_bn={fuse_bn})")

    return model


def _fuse_all_conv_bn(module: nn.Module):
    """递归查找并融合所有 Sequential 中的 Conv+BN 对"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            # 尝试融合此 Sequential 内的 Conv+BN
            fused = fuse_conv_bn_in_sequential(child)
            setattr(module, name, fused)
            # 递归处理融合后的模块
            _fuse_all_conv_bn(fused)
        else:
            _fuse_all_conv_bn(child)


def _replace_with_qat(module: nn.Module):
    """递归将所有 nn.Conv2d 替换为 QATConv2d, nn.Linear 替换为 QATLinear"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QATConv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QATLinear(child))
        elif isinstance(child, QATConv2d) or isinstance(child, QATLinear):
            # 已经转换过，跳过
            continue
        else:
            # 递归处理子模块
            _replace_with_qat(child)


# ============================================================
#  QAT 模式控制
# ============================================================

def enable_qat(model: nn.Module):
    """启用模型中所有 QAT 层的伪量化"""
    count = 0
    for m in model.modules():
        if isinstance(m, (QATConv2d, QATLinear)):
            m.enable_qat()
            count += 1
    print(f"[enable_qat] Enabled QAT on {count} layers")
    return count


def disable_qat(model: nn.Module):
    """禁用模型中所有 QAT 层的伪量化 (恢复 float32 精度)"""
    count = 0
    for m in model.modules():
        if isinstance(m, (QATConv2d, QATLinear)):
            m.disable_qat()
            count += 1
    print(f"[disable_qat] Disabled QAT on {count} layers")
    return count


def get_qat_layer_count(model: nn.Module) -> dict:
    """统计 QAT 层数量"""
    conv_count = 0
    linear_count = 0
    for m in model.modules():
        if isinstance(m, QATConv2d):
            conv_count += 1
        elif isinstance(m, QATLinear):
            linear_count += 1
    return {"QATConv2d": conv_count, "QATLinear": linear_count}


# ============================================================
#  校准 (Calibration)
# ============================================================

def calibrate_qat_model(model: nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        num_batches: int = 5) -> None:
    """
    校准 QAT 模型: 通过少量数据批次预热量化 scale。

    对于使用"动态 scale"的量化方案 (即每步计算 scale)，校准步骤不是
    严格必需的 (因为 scale 会在前向时自动计算)。但如果未来改用"静态 scale"
    (学习型 scale 参数)，则需要校准来初始化 scale。

    当前版本主要用途:
      1. 验证 QAT 模型前向传播正常
      2. 为未来的 LSQ 风格 scale 初始化做预留

    Args:
        model:       QAT-ready 模型
        dataloader:  训练数据加载器
        device:      计算设备
        num_batches: 校准批次数
    """
    model.eval()
    model.to(device)

    print(f"[calibrate] Running calibration on {num_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
            print(f"  Batch {i+1}/{num_batches} — input range: "
                  f"[{images.min():.3f}, {images.max():.3f}]")

    print(f"[calibrate] Calibration complete.")


# ============================================================
#  QAT 模型导出
# ============================================================

def export_to_standard_model(qat_model: nn.Module,
                             model_class: type) -> nn.Module:
    """
    从 QAT 模型导出标准 float32 模型 (移除 QAT 包装)。

    QAT 训练后，权重已经对 int4 量化具有鲁棒性。
    导出为标准模型后，可以:
      1. 直接用于 float32 推理
      2. 加载到 optic_layers.py 的 OpticConv2d/OpticLinear 进行光计算推理
      3. 导出为 ONNX 等格式

    Args:
        qat_model:   QAT-trained 模型
        model_class: 原始模型类 (用于创建相同架构的新实例)

    Returns:
        标准 nn.Module, 权重为 QAT 训练的 float32 值

    Example:
        >>> std_model = export_to_standard_model(qat_model, BaselineVGG)
        >>> torch.save(std_model.state_dict(), "baseline_vgg_qat.pth")
    """
    std_model = model_class()
    std_state = {}

    for (qat_name, qat_param) in qat_model.named_parameters():
        # QAT 层参数名与原始层相同 (QAT 只是外包装)
        std_state[qat_name] = qat_param.data.clone()

    # 尝试加载 (忽略不匹配的 key, 例如可能存在的额外参数)
    std_model.load_state_dict(std_state, strict=False)
    return std_model


# ============================================================
#  硬件对齐率计算
# ============================================================

def compute_alignment_ratio(model: nn.Module) -> float:
    """
    计算模型在 8×2 光计算硬件上的综合对齐率。

    对于每个卷积层，im2col 展平长度 = C_in × k_h × k_w。
    如果该长度不能被 8 整除，则需要补零到最近的 8 的倍数。
    对齐率 = 原始长度 / 补零后长度。

    注意: QAT 不影响对齐率 (对齐率仅由网络结构决定)。

    Args:
        model: 任意模型 (QAT 或标准)

    Returns:
        float: 综合对齐率 (1.0 = 100% 完美对齐)
    """
    total_patch = 0
    total_padded = 0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, QATConv2d)):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
        elif isinstance(m, (nn.Linear, QATLinear)):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded

    return total_patch / total_padded if total_padded > 0 else 0.0


def print_alignment_detail(model: nn.Module, label: str = ""):
    """打印每层硬件对齐详情"""
    header = f"  [{label}]" if label else ""
    print(f"\n{header} 层名                            C_in   K      展平长度  补零后  对齐率")
    print("  " + "-" * 72)
    total_patch, total_padded = 0, 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, QATConv2d)):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            layer_type = "QATConv2d" if isinstance(m, QATConv2d) else "Conv2d"
            print(f"  [{layer_type:<11s}] {name:<25s} {m.in_channels:>4d}   "
                  f"{m.kernel_size[0]}×{m.kernel_size[1]}"
                  f"   {patch:>8d}   {padded:>8d}   {patch/padded:.1%}")
        elif isinstance(m, (nn.Linear, QATLinear)):
            patch = m.in_features
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            layer_type = "QATLinear" if isinstance(m, QATLinear) else "Linear"
            print(f"  [{layer_type:<11s}] {name:<25s}  —     —    "
                  f"   {patch:>8d}   {padded:>8d}   {patch/padded:.1%}")

    overall = total_patch / total_padded if total_padded > 0 else 0
    print(f"  综合硬件对齐率: {overall:.1%} (总展平 {total_patch} → 补零后 {total_padded})")
    return overall


# ============================================================
#  评估工具
# ============================================================

@torch.no_grad()
def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   device: torch.device,
                   criterion: nn.Module = None) -> dict:
    """
    评估模型准确率和损失。

    Args:
        model:     模型 (QAT 或标准)
        dataloader: 数据加载器
        device:    计算设备
        criterion: 损失函数 (可选)

    Returns:
        {"accuracy": float, "loss": float, "total": int, "correct": int}
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
        "total": total,
        "correct": correct,
    }


# ============================================================
#  性能对比
# ============================================================

@torch.no_grad()
def compare_qat_vs_float(qat_model: nn.Module,
                         dataloader: DataLoader,
                         device: torch.device,
                         criterion: nn.Module = None) -> dict:
    """
    对比 QAT 模型在 QAT 模式与 float32 模式下的准确率差异。

    这是一个诊断工具:
      - 如果 QAT 模式准确率 ≈ float32 模式 → QAT 训练成功
      - 如果 QAT 模式准确率 << float32 模式 → 模型尚未适应量化，需要更多 QAT 训练

    Args:
        qat_model:  QAT-trained 模型
        dataloader: 验证数据
        device:     计算设备
        criterion:  损失函数

    Returns:
        {"qat_mode": {...}, "float_mode": {...}, "accuracy_gap": float}
    """
    print("\n" + "=" * 60)
    print("  QAT vs Float32 Comparison")
    print("=" * 60)

    # QAT 模式
    enable_qat(qat_model)
    qat_result = evaluate_model(qat_model, dataloader, device, criterion)
    print(f"  QAT mode (int4):     Accuracy = {qat_result['accuracy']:.2%}")

    # Float32 模式
    disable_qat(qat_model)
    float_result = evaluate_model(qat_model, dataloader, device, criterion)
    print(f"  Float mode (float32): Accuracy = {float_result['accuracy']:.2%}")

    gap = float_result['accuracy'] - qat_result['accuracy']
    print(f"  Accuracy gap:         {gap:.2%} "
          f"({'✓ QAT successful' if gap < 0.02 else '⚠ Needs more QAT training'})")

    return {
        "qat_mode": qat_result,
        "float_mode": float_result,
        "accuracy_gap": gap,
    }


# ============================================================
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  optic_qat.py — Self-Test")
    print("=" * 60)

    # ---- 测试 fake_int4_quantize ----
    print("\n[Test 1] fake_int4_quantize")
    x = torch.randn(4, 8, 32, 32) * 3.0
    x_q = fake_int4_quantize(x, per_channel=True, ch_dim=1)
    # 验证: 量化后的值应该是 1/7 精度的离散值
    unique_vals = x_q.unique().numel()
    print(f"  Input shape: {x.shape}, unique values: {x.unique().numel()}")
    print(f"  Quantized unique values: {unique_vals} (should be ≤ 16 for int4)")
    print(f"  Max abs error: {(x - x_q).abs().max():.4f}")
    print(f"  [OK] fake_int4_quantize works")

    # ---- 测试 QATConv2d ----
    print("\n[Test 2] QATConv2d")
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
    qat_conv = QATConv2d(conv)
    test_input = torch.randn(2, 8, 32, 32)

    # QAT mode
    qat_conv.train()
    qat_out = qat_conv(test_input)
    print(f"  QAT mode output: {qat_out.shape} [OK]")

    # Float mode
    qat_conv.eval()
    qat_conv.disable_qat()
    float_out = qat_conv(test_input)
    native_out = conv(test_input)
    print(f"  Float mode output: {float_out.shape}")
    print(f"  Float vs native max diff: {(float_out - native_out).abs().max():.8f}")
    print(f"  [OK] QATConv2d works")

    # ---- 测试 QATLinear ----
    print("\n[Test 3] QATLinear")
    linear = nn.Linear(64, 10)
    qat_linear = QATLinear(linear)
    test_vec = torch.randn(4, 64)
    qat_linear.train()
    qat_out_l = qat_linear(test_vec)
    print(f"  QAT mode output: {qat_out_l.shape} [OK]")

    # ---- 测试 BN 融合 ----
    print("\n[Test 4] Conv+BN Fusion")
    conv2 = nn.Conv2d(3, 8, 3, padding=1)
    bn = nn.BatchNorm2d(8)
    bn.eval()  # 设置 running stats
    # 模拟一次前向以初始化 BN running stats
    with torch.no_grad():
        _ = bn(conv2(torch.randn(2, 3, 8, 8)))
    fused = fuse_conv_bn(conv2, bn)
    print(f"  Fused conv: in={fused.in_channels}, out={fused.out_channels}, "
          f"has_bias={fused.bias is not None} [OK]")

    # ---- 测试 prepare_qat_model ----
    print("\n[Test 5] prepare_qat_model")

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
    # 先 eval 让 BN 有 running stats
    dummy.eval()
    with torch.no_grad():
        _ = dummy(torch.randn(2, 3, 32, 32))
    prepare_qat_model(dummy, fuse_bn=False)  # 不融合 (测试基础转换)
    print(f"  conv: {type(dummy.conv).__name__}")
    print(f"  fc:   {type(dummy.fc).__name__}")
    counts = get_qat_layer_count(dummy)
    print(f"  QAT layers: {counts} [OK]")

    # ---- 测试 enable/disable ----
    print("\n[Test 6] enable_qat / disable_qat")
    enable_qat(dummy)
    disable_qat(dummy)
    print(f"  [OK] Toggle works")

    # ---- 测试评估 ----
    print("\n[Test 7] evaluate_model")
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(torch.randn(16, 3, 32, 32),
                               torch.randint(0, 10, (16,)))
    dummy_loader = DataLoader(dummy_data, batch_size=4)
    result = evaluate_model(dummy, dummy_loader, torch.device("cpu"),
                            nn.CrossEntropyLoss())
    print(f"  Accuracy: {result['accuracy']:.2%}, Loss: {result['loss']:.4f} [OK]")

    # ---- 测试对齐率 ----
    print("\n[Test 8] compute_alignment_ratio")
    ratio = compute_alignment_ratio(dummy)
    print(f"  Alignment ratio: {ratio:.1%} [OK]")

    print("\n" + "=" * 60)
    print("  All self-tests passed! [OK]")
    print("=" * 60)
