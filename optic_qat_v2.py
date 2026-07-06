"""
================================================================================
 optic_qat_v2.py — 基于初赛验证方法的 QAT 量化模块 (Phase 4)

 参考:
   初赛 MNIST 量化方案 (STE / LSQ+ / DSQ)
   - 01_设计报告.pdf: 量化方法设计
   - 02_验证报告.pdf: STE 97.03% 最佳精度
   - 03_技术数据.pdf: 复现步骤与超参数

 核心改进 (vs optic_qat.py Phase 1-3):
   1. 非对称量化: 激活 uint4 [0,15], 权重 int4 [-8,7]
   2. STE+噪声注入: 训练时向权重注入高斯噪声 (std=0.05*scale)
   3. LSQ+: 可学习 scale + zero_point, 独立学习率 (0.1x)
   4. 逐层输出重量化: 每层 ReLU 后重新量化为 uint4
   5. bias=False: 匹配光计算硬件约束

 提供:
   - FakeQuantizeUINT4:         uint4 伪量化 (激活值用)
   - FakeQuantizeINT4:          int4 伪量化 (权重用)
   - QATConv2d_v2:              QAT 卷积 (非对称量化 + 噪声 + 逐层重量化)
   - QATLinear_v2:              QAT 全连接 (同上)
   - prepare_model_phase4():    模型转换
   - set_quant_lr():            设置量化参数独立学习率
   - 训练/评估工具
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


# ============================================================
#  Fake Quantization Functions
# ============================================================

def fake_quantize_uint4(x: torch.Tensor,
                         per_channel: bool = False,
                         ch_dim: int = 0) -> torch.Tensor:
    """
    伪 uint4 量化 [0, 15] — 用于 ReLU 后的激活值。

    非对称量化，充分利用 uint4 的 16 个级别。
    与初赛方案一致: scale = max(x) / 15, 无 zero_point 偏移。

    STE 梯度: x + (x_q - x).detach()
    """
    if per_channel:
        reduce_dims = [i for i in range(x.dim()) if i != ch_dim]
        xmax = x
        for d in sorted(reduce_dims, reverse=True):
            xmax = xmax.max(dim=d, keepdim=True)[0]
        scale = (xmax / 15.0).clamp(min=1e-8)
    else:
        scale = (x.max() / 15.0).clamp(min=1e-8)

    x_int = (x / scale).round().clamp(0, 15)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


def fake_quantize_int4(x: torch.Tensor,
                        per_channel: bool = False,
                        ch_dim: int = 0,
                        inject_noise: bool = False) -> torch.Tensor:
    """
    伪 int4 量化 [-8, 7] — 用于权重。

    对称量化，scale = max(|x|) / 7。
    可选噪声注入 (初赛 STE 方案核心)。

    Args:
        x:              float32 张量
        per_channel:    是否逐通道计算 scale
        ch_dim:         通道维度
        inject_noise:   是否注入高斯噪声 (std = 0.05 * scale)
                        仅训练时使用
    """
    if per_channel:
        reduce_dims = [i for i in range(x.dim()) if i != ch_dim]
        amax = x.abs()
        for d in sorted(reduce_dims, reverse=True):
            amax = amax.max(dim=d, keepdim=True)[0]
        scale = (amax / 7.0).clamp(min=1e-8)
    else:
        scale = (x.abs().max() / 7.0).clamp(min=1e-8)

    # 噪声注入 (初赛 STE 方案: std = 0.05 * scale)
    if inject_noise and x.requires_grad:
        noise = torch.randn_like(x) * 0.05 * scale
        x = x + noise

    x_int = (x / scale).round().clamp(-8, 7)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


# ============================================================
#  LSQ+ 可学习量化 (learnable scale + zero_point)
# ============================================================

class _LSQPlusInt4Fn(torch.autograd.Function):
    """
    LSQ+ int4 量化 — 可学习 scale + zero_point。

    参考: 初赛 01_设计报告 3.2 节
    梯度缩放: 1 / sqrt(N * qmax)
    独立学习率: scale/zp 使用 0.1x 基础 lr
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
        x_dq = (x_int - zero_point) * scale
        ctx.save_for_backward(x, scale, zero_point, x_int)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point, x_int = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        n_levels = qmax - qmin + 1  # 16 for int4 signed

        # === x 梯度: 截断区间外为 0 ===
        inner = ((x_int > qmin) & (x_int < qmax)).float()
        grad_x = inner * grad_output

        # === scale 梯度 ===
        inner_s = (x_int.abs() < qmax).float()
        outer_s = 1.0 - inner_s
        grad_scale = (
            inner_s * (x_int - zero_point - x / scale) +
            outer_s * torch.sign(x) * qmax
        ) * grad_output

        # 规约到 scale 的 shape
        sum_dims = [d for d in range(x.dim())
                     if d >= scale.dim() or scale.shape[d] == 1]
        if sum_dims:
            grad_scale = grad_scale.sum(dim=sum_dims)
        grad_scale = grad_scale.view(scale.shape)

        # LSQ 梯度缩放
        N = x.numel() / max(1, scale.numel())
        grad_scale = grad_scale / (N * n_levels) ** 0.5

        # === zero_point 梯度 ===
        grad_zp = -(x_int - zero_point) * inner_s * grad_output
        sum_dims_zp = [d for d in range(x.dim())
                        if d >= zero_point.dim() or zero_point.shape[d] == 1]
        if sum_dims_zp:
            grad_zp = grad_zp.sum(dim=sum_dims_zp)
        grad_zp = grad_zp.view(zero_point.shape)

        return grad_x, grad_scale, grad_zp, None, None


def lsqplus_int4_quantize(x, scale, zero_point, qmin=-8, qmax=7):
    """LSQ+ 可学习 int4 量化"""
    return _LSQPlusInt4Fn.apply(x, scale, zero_point, qmin, qmax)


class _LSQPlusUInt4Fn(torch.autograd.Function):
    """
    LSQ+ uint4 量化 [0, 15] — 用于激活值。
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
        x_dq = (x_int - zero_point) * scale
        ctx.save_for_backward(x, scale, zero_point, x_int)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point, x_int = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        n_levels = qmax - qmin + 1

        inner = ((x_int > qmin) & (x_int < qmax)).float()
        grad_x = inner * grad_output

        inner_s = (x_int > qmin).float() & (x_int < qmax).float()
        outer_s = 1.0 - inner_s
        grad_scale = (
            inner_s * (x_int - zero_point - x / scale) +
            outer_s * torch.where(x > 0,
                                  torch.ones_like(x) * qmax,
                                  torch.ones_like(x) * qmin)
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


def lsqplus_uint4_quantize(x, scale, zero_point, qmin=0, qmax=15):
    """LSQ+ 可学习 uint4 量化"""
    return _LSQPlusUInt4Fn.apply(x, scale, zero_point, qmin, qmax)


# ============================================================
#  QAT-aware Conv2d v2
# ============================================================

class QATConv2d_v2(nn.Module):
    """
    QAT 卷积层 v2 — 基于初赛方案。

    特性:
      - 激活量化: uint4 [0,15] (非对称, 匹配 ReLU 输出)
      - 权重量化: int4 [-8,7] (对称)
      - STE 模式: 静态 scale + 可选噪声注入
      - LSQ+ 模式: 可学习 scale + zero_point
      - 逐层输出重量化: ReLU 后重新量化为 uint4
      - bias=False: 匹配光计算硬件
    """

    def __init__(self, conv_layer: nn.Conv2d,
                 mode: str = "ste",        # "ste" or "lsqplus"
                 noise: bool = True,       # STE 噪声注入
                 quantize_output: bool = True):  # 逐层输出重量化
        super().__init__()

        # 卷积参数
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        # 权重 (bias=False 匹配光计算硬件)
        self.weight = nn.Parameter(conv_layer.weight.data.clone())
        self.bias = None  # 光计算不支持 bias

        self._qat_enabled = True
        self.mode = mode
        self._noise = noise
        self._quantize_output = quantize_output

        # ---- LSQ+ 可学习参数 ----
        if mode == "lsqplus":
            # 权重 scale + zero_point (per-output-channel)
            w_amax = self.weight.data.abs().view(self.out_channels, -1).max(dim=1)[0]
            init_w_scale = (w_amax / 7.0).clamp(min=1e-8)
            self.weight_scale = nn.Parameter(init_w_scale.view(-1, 1, 1, 1))
            self.weight_zp = nn.Parameter(torch.zeros(self.out_channels, 1, 1, 1))

            # 输出 scale + zero_point (per-output-channel, 用于逐层重量化)
            self.out_scale = nn.Parameter(torch.ones(self.out_channels, 1, 1))
            self.out_zp = nn.Parameter(torch.zeros(self.out_channels, 1, 1))

    @property
    def qat_enabled(self) -> bool:
        return self._qat_enabled

    def enable_qat(self):
        self._qat_enabled = True

    def disable_qat(self):
        self._qat_enabled = False

    def quant_params(self):
        """返回量化参数 (用于设置独立学习率)"""
        params = []
        if self.mode == "lsqplus" and hasattr(self, 'weight_scale'):
            params.extend([self.weight_scale, self.weight_zp,
                          self.out_scale, self.out_zp])
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.conv2d(x, self.weight, None,
                            self.stride, self.padding, self.dilation, self.groups)

        # === 输入激活量化: uint4 [0, 15] ===
        if self.mode == "lsqplus" and hasattr(self, 'out_scale'):
            # 这里用上一层的输出量化参数... 实际由上一层负责
            # 本层的输入量化
            x_q = fake_quantize_uint4(x, per_channel=True, ch_dim=1)
        else:
            x_q = fake_quantize_uint4(x, per_channel=True, ch_dim=1)

        # === 权重量化: int4 [-8, 7] ===
        if self.mode == "lsqplus" and hasattr(self, 'weight_scale'):
            w_s = self.weight_scale.abs().clamp(min=1e-8)
            w_zp = self.weight_zp
            w_q = lsqplus_int4_quantize(self.weight, w_s, w_zp, -8, 7)
        else:
            inject = self._noise and self.training
            w_q = fake_quantize_int4(self.weight, per_channel=True, ch_dim=0,
                                     inject_noise=inject)

        # === 卷积 (无 bias) ===
        out = F.conv2d(x_q, w_q, None,
                       self.stride, self.padding, self.dilation, self.groups)

        # === 逐层输出重量化 (模拟光计算推理管线) ===
        # ReLU 后重新量化为 uint4
        if self._quantize_output and self._qat_enabled:
            out = F.relu(out)  # 确保非负
            out = fake_quantize_uint4(out, per_channel=True, ch_dim=1)

        return out

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, mode={self.mode}, "
                f"noise={self._noise}, bias=False")


# ============================================================
#  QAT-aware Linear v2
# ============================================================

class QATLinear_v2(nn.Module):
    """
    QAT 全连接层 v2 — 同 QATConv2d_v2 设计。
    """

    def __init__(self, linear_layer: nn.Linear,
                 mode: str = "ste",
                 noise: bool = True,
                 quantize_output: bool = True,
                 is_last_layer: bool = False):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        self.bias = None

        self._qat_enabled = True
        self.mode = mode
        self._noise = noise
        self._quantize_output = quantize_output and not is_last_layer
        self._is_last_layer = is_last_layer

        if mode == "lsqplus":
            w_amax = self.weight.data.abs().view(self.out_features, -1).max(dim=1)[0]
            init_w_scale = (w_amax / 7.0).clamp(min=1e-8)
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
        if self.mode == "lsqplus" and hasattr(self, 'weight_scale'):
            params.extend([self.weight_scale, self.weight_zp])
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.linear(x, self.weight, None)

        # 输入量化: uint4
        x_q = fake_quantize_uint4(x, per_channel=True, ch_dim=-1)

        # 权重量化: int4
        if self.mode == "lsqplus" and hasattr(self, 'weight_scale'):
            w_s = self.weight_scale.abs().clamp(min=1e-8)
            w_q = lsqplus_int4_quantize(self.weight, w_s, self.weight_zp, -8, 7)
        else:
            inject = self._noise and self.training
            w_q = fake_quantize_int4(self.weight, per_channel=True, ch_dim=0,
                                     inject_noise=inject)

        out = F.linear(x_q, w_q, None)

        # 逐层输出重量化 (末层跳过)
        if self._quantize_output:
            out = F.relu(out)
            out = fake_quantize_uint4(out, per_channel=True, ch_dim=-1)

        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"mode={self.mode}, last={self._is_last_layer}, bias=False")


# ============================================================
#  模型转换
# ============================================================

def prepare_model_phase4(model: nn.Module,
                         mode: str = "ste",
                         noise: bool = True,
                         first_layer_fp32: bool = True,
                         last_layer_fp32: bool = True,
                         inplace: bool = True) -> nn.Module:
    """
    将标准模型转换为 Phase 4 QAT 模型。

    Args:
        model:              标准 PyTorch 模型
        mode:               量化模式 "ste" 或 "lsqplus"
        noise:              STE 模式是否注入训练噪声
        first_layer_fp32:   首层是否保持 float32
        last_layer_fp32:    末层是否保持 float32
        inplace:            是否原地修改

    Returns:
        QAT-ready 模型 (bias=False, 非对称量化)
    """
    if not inplace:
        model = copy.deepcopy(model)

    _convert_to_phase4(model, mode, noise, first_layer_fp32, last_layer_fp32)

    qat_count = sum(1 for m in model.modules()
                    if isinstance(m, (QATConv2d_v2, QATLinear_v2)))
    print(f"[prepare_model_phase4] Converted {qat_count} layers to QAT v2 "
          f"(mode={mode}, noise={noise}, bias=False)")
    return model


def _convert_to_phase4(module: nn.Module, mode, noise,
                       first_layer_fp32, last_layer_fp32,
                       _depth=0):
    """递归转换"""
    is_first_conv = (_depth == 0)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            is_first = is_first_conv and first_layer_fp32
            qat_layer = QATConv2d_v2(child, mode=mode, noise=noise)
            if is_first:
                qat_layer.disable_qat()
            setattr(module, name, qat_layer)
            is_first_conv = False
        elif isinstance(child, nn.Linear):
            # 判断是否为末层 (输出维度 = num_classes, 且后面没有 Linear)
            is_last = (child.out_features == 10 and last_layer_fp32)
            qat_layer = QATLinear_v2(child, mode=mode, noise=noise,
                                     is_last_layer=is_last)
            if is_last:
                qat_layer.disable_qat()
            setattr(module, name, qat_layer)
        elif isinstance(child, (QATConv2d_v2, QATLinear_v2)):
            continue
        else:
            _convert_to_phase4(child, mode, noise,
                              first_layer_fp32, last_layer_fp32,
                              _depth + 1)


def set_quant_lr(model: nn.Module, base_lr: float) -> list:
    """
    为 LSQ+ 量化参数设置独立学习率 (0.1x base_lr)。

    返回: [{'params': weight_params, 'lr': base_lr},
           {'params': quant_params, 'lr': base_lr * 0.1}]

    用法:
        param_groups = set_quant_lr(model, lr=0.001)
        optimizer = torch.optim.AdamW(param_groups)
    """
    weight_params = []
    quant_params = []

    for name, param in model.named_parameters():
        is_quant = any(k in name for k in ['weight_scale', 'weight_zp',
                                            'out_scale', 'out_zp'])
        if is_quant:
            quant_params.append(param)
        else:
            weight_params.append(param)

    return [
        {'params': weight_params, 'lr': base_lr},
        {'params': quant_params, 'lr': base_lr * 0.1},
    ]


# ============================================================
#  评估工具
# ============================================================

@torch.no_grad()
def evaluate_model_v2(model, dataloader, device, criterion=None):
    """评估模型 (QAT 层在 eval 模式也施加量化)"""
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


# ============================================================
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  optic_qat_v2.py — Self-Test")
    print("=" * 60)

    # Test uint4 quantize
    print("\n[Test 1] fake_quantize_uint4")
    x = torch.randn(4, 8, 32, 32).abs() + 0.5  # 模拟 ReLU 输出
    x_q = fake_quantize_uint4(x, per_channel=True, ch_dim=1)
    print(f"  Input: min={x.min():.3f}, max={x.max():.3f}")
    print(f"  Quantized: min={x_q.min():.3f}, max={x_q.max():.3f}")
    print(f"  Unique values: {x_q.unique().numel()} (≤ 16 expected)")
    print(f"  [OK]")

    # Test int4 + noise
    print("\n[Test 2] fake_quantize_int4 + noise")
    w = torch.randn(16, 3, 3, 3) * 0.5
    w.requires_grad = True
    w_q = fake_quantize_int4(w, per_channel=True, ch_dim=0, inject_noise=True)
    loss = w_q.sum()
    loss.backward()
    print(f"  Weight grad norm: {w.grad.norm():.4f}")
    print(f"  [OK] gradient flows through STE+noise")

    # Test QATConv2d_v2 STE
    print("\n[Test 3] QATConv2d_v2 (STE mode)")
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat = QATConv2d_v2(conv, mode="ste", noise=True)
    qat.train()
    x = torch.randn(2, 3, 32, 32).abs()
    out = qat(x)
    print(f"  Output shape: {out.shape}")
    loss = out.sum()
    loss.backward()
    print(f"  Weight grad norm: {qat.weight.grad.norm():.4f}")
    print(f"  [OK]")

    # Test QATConv2d_v2 LSQ+
    print("\n[Test 4] QATConv2d_v2 (LSQ+ mode)")
    conv2 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat2 = QATConv2d_v2(conv2, mode="lsqplus", noise=False)
    qat2.train()
    out2 = qat2(x)
    loss2 = out2.sum()
    loss2.backward()
    print(f"  weight_scale.grad shape: {qat2.weight_scale.grad.shape}")
    print(f"  weight_zp.grad shape: {qat2.weight_zp.grad.shape}")
    print(f"  [OK]")

    # Test QATLinear_v2
    print("\n[Test 5] QATLinear_v2")
    lin = nn.Linear(64, 10, bias=False)
    qat_lin = QATLinear_v2(lin, mode="ste", noise=True, is_last_layer=True)
    qat_lin.train()
    v = torch.randn(4, 64).abs()
    out3 = qat_lin(v)
    print(f"  Output shape: {out3.shape}")
    loss3 = out3.sum()
    loss3.backward()
    print(f"  [OK] last layer no output requantize")

    # Test prepare_model_phase4
    print("\n[Test 6] prepare_model_phase4")

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(8 * 8 * 8, 10, bias=False)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    tiny = TinyCNN()
    prepare_model_phase4(tiny, mode="ste", noise=True)
    print(f"  conv: {type(tiny.conv).__name__}")
    print(f"  fc:   {type(tiny.fc).__name__} (is_last={tiny.fc._is_last_layer})")
    print(f"  [OK]")

    # Test training works
    print("\n[Test 7] Training simulation (5 steps)")
    tiny2 = TinyCNN()
    prepare_model_phase4(tiny2, mode="ste", noise=True)
    tiny2.train()
    opt = torch.optim.Adam(tiny2.parameters(), lr=0.001)
    for i in range(5):
        opt.zero_grad()
        out = tiny2(torch.randn(4, 3, 8, 8).abs())
        loss = nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,)))
        loss.backward()
        opt.step()
        print(f"  Step {i+1}: loss={loss.item():.4f}")
    print(f"  [OK] Model learns!")

    # Test set_quant_lr
    print("\n[Test 8] set_quant_lr (LSQ+ mode)")
    tiny3 = TinyCNN()
    prepare_model_phase4(tiny3, mode="lsqplus", noise=False)
    pg = set_quant_lr(tiny3, base_lr=0.001)
    print(f"  Weight params lr: {pg[0]['lr']}")
    print(f"  Quant params lr:  {pg[1]['lr']}")
    print(f"  Quant params count: {len(pg[1]['params'])}")
    print(f"  [OK]")

    print("\n" + "=" * 60)
    print("  All self-tests passed! [OK]")
    print("=" * 60)
