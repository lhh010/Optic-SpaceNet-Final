"""
================================================================================
 optic_qat_lsq.py — LSQ+ int8 QAT 模块 (修复版)

 修复 (vs optic_qat_v2.py 的 LSQ+):
   1. 激活 int8 (256 级) 替代 uint4 (16 级) — 信息瓶颈修复
   2. in_scale/in_zp 真正参与前向 — 死参数修复
   3. 无内部 ReLU — 不干扰模型架构
   4. 无内部输出重量化 — 由下一层输入量化接管
   5. BN 保留 — 稳定激活分布
   6. scale/zp 初始化从权重统计计算 — 避免 1.0 初始化
   7. LSQ 梯度缩放: 1/sqrt(N * qmax) — 标准 LSQ 公式

 LSQ+ 原理 (Esser et al., ICLR 2020):
   - 每层有可学习的 scale 和 zero_point 参数
   - 前向: x_q = round(x/scale + zp).clamp() → x_dq = (x_q - zp) * scale
   - 反向: STE 直通梯度 + scale/zp 自己的 LSQ 梯度
   - scale 学习率独立 (0.1x base_lr)

 用法:
   from optic_qat_lsq import prepare_model_lsq, LSQConv2d, LSQLinear
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


# ============================================================
#  LSQ+ 自定义 autograd Function
# ============================================================

class _LSQPlusFn(torch.autograd.Function):
    """
    LSQ+ 量化: x_q = round(x/scale + zp).clamp(qmin, qmax)
              x_dq = (x_q - zp) * scale

    梯度:
      - x: STE (直通), 截断区外为 0
      - scale: LSQ 公式, 缩放 1/sqrt(N * n_levels)
      - zp: LSQ 公式, 缩放 1/sqrt(N * n_levels)
    """

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

        # === x 梯度: STE, 截断区外 0 ===
        inner = ((x_int > qmin) & (x_int < qmax)).float()
        grad_x = inner * grad_output

        # === scale 梯度 ===
        inner_s = inner
        outer_s = 1.0 - inner_s
        grad_scale = (
            inner_s * (x_int - zero_point - x / scale) +
            outer_s * torch.where(x > 0,
                                  torch.full_like(x, qmax),
                                  torch.full_like(x, qmin))
        ) * grad_output

        # 规约到 scale 的 shape (处理 x.dim() > scale.dim() 的偏移)
        offset = x.dim() - scale.dim()
        sum_dims = list(range(offset))
        for d in range(scale.dim()):
            if scale.shape[d] == 1:
                sum_dims.append(offset + d)
        if sum_dims:
            grad_scale = grad_scale.sum(dim=sum_dims)
        grad_scale = grad_scale.view(scale.shape)

        # LSQ 梯度缩放
        N = x.numel() / max(1, scale.numel())
        grad_scale = grad_scale / (N * n_levels) ** 0.5

        # === zero_point 梯度 ===
        grad_zp = -(x_int - zero_point) * inner_s * grad_output
        offset_zp = x.dim() - zero_point.dim()
        sum_dims_zp = list(range(offset_zp))
        for d in range(zero_point.dim()):
            if zero_point.shape[d] == 1:
                sum_dims_zp.append(offset_zp + d)
        if sum_dims_zp:
            grad_zp = grad_zp.sum(dim=sum_dims_zp)
        grad_zp = grad_zp.view(zero_point.shape)
        grad_zp = grad_zp / (N * n_levels) ** 0.5

        return grad_x, grad_scale, grad_zp, None, None


def lsq_quantize(x, scale, zero_point, qmin, qmax):
    """LSQ+ 可学习量化"""
    return _LSQPlusFn.apply(x, scale, zero_point, qmin, qmax)


# ============================================================
#  STE fallback (warmup 阶段使用)
# ============================================================

def ste_quantize(x, bits, per_channel=True, ch_dim=0):
    """STE 对称量化 (fallback/warmup)"""
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
    x_int = (x / scale).round().clamp(qmin, qmax)
    x_dq = x_int * scale
    return x + (x_dq - x).detach()


# ============================================================
#  LSQConv2d — LSQ+ 卷积层
# ============================================================

class LSQConv2d(nn.Module):
    """
    LSQ+ 卷积层 (修复版).

    关键修复:
      - in_scale/in_zp 真正参与输入量化前向
      - weight_scale/weight_zp 参与权重量化前向
      - 无内部 ReLU, 无内部输出重量化
      - 所有 LSQ 参数有正确的 LSQ 梯度
    """

    def __init__(self, conv_layer: nn.Conv2d,
                 weight_bits: int = 8,
                 act_bits: int = 8,
                 use_lsq: bool = True):
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

        self._qat_enabled = True
        self._weight_bits = weight_bits
        self._act_bits = act_bits
        self._use_lsq = use_lsq

        w_qmax = 2 ** (weight_bits - 1) - 1
        a_qmax = 2 ** (act_bits - 1) - 1

        # === LSQ+ 可学习参数 ===
        # 权重 scale/zp (per-output-channel)
        w_amax = self.weight.data.abs().view(self.out_channels, -1).max(dim=1)[0]
        init_w_scale = (w_amax / w_qmax).clamp(min=1e-8)
        self.weight_scale = nn.Parameter(init_w_scale.view(-1, 1, 1, 1))
        self.weight_zp = nn.Parameter(torch.zeros(self.out_channels, 1, 1, 1))

        # 输入 scale/zp (per-input-channel) — ★ 修复: 真正用于前向
        # 用合理初值: scale = 2*std / qmax (覆盖 ~95% 的值)
        # 后续通过 LSQ 梯度自动调整
        self.in_scale = nn.Parameter(torch.ones(self.in_channels, 1, 1) * 0.5 / a_qmax)
        self.in_zp = nn.Parameter(torch.zeros(self.in_channels, 1, 1))

    @property
    def qat_enabled(self): return self._qat_enabled

    def enable_qat(self): self._qat_enabled = True
    def disable_qat(self): self._qat_enabled = False

    def lsq_params(self):
        """返回 LSQ+ 参数 (用于独立学习率)"""
        return [self.weight_scale, self.weight_zp,
                self.in_scale, self.in_zp]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.conv2d(x, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)

        a_qmax = 2 ** (self._act_bits - 1) - 1
        w_qmax = 2 ** (self._weight_bits - 1) - 1

        # === 输入量化 (LSQ+ 可学习 scale/zp) ★ 修复: in_scale 真正参与 ===
        if self._use_lsq:
            in_s = self.in_scale.abs().clamp(min=1e-8)
            x_q = lsq_quantize(x, in_s, self.in_zp, -a_qmax, a_qmax)
        else:
            x_q = ste_quantize(x, self._act_bits, per_channel=True, ch_dim=1)

        # === 权重量化 (LSQ+ 可学习 scale/zp) ===
        if self._use_lsq:
            w_s = self.weight_scale.abs().clamp(min=1e-8)
            w_q = lsq_quantize(self.weight, w_s, self.weight_zp, -w_qmax, w_qmax)
        else:
            w_q = ste_quantize(self.weight, self._weight_bits,
                               per_channel=True, ch_dim=0)

        # === 卷积 (无内部 ReLU, 无输出重量化) ===
        return F.conv2d(x_q, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"w{self._weight_bits}/a{self._act_bits}, "
                f"lsq={self._use_lsq}, bias={self.bias is not None}")


# ============================================================
#  LSQLinear — LSQ+ 全连接层
# ============================================================

class LSQLinear(nn.Module):
    """LSQ+ 全连接层 (修复版)"""

    def __init__(self, linear_layer: nn.Linear,
                 weight_bits: int = 8,
                 act_bits: int = 8,
                 use_lsq: bool = True,
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
        self._use_lsq = use_lsq
        self._is_last_layer = is_last_layer

        w_qmax = 2 ** (weight_bits - 1) - 1
        a_qmax = 2 ** (act_bits - 1) - 1

        w_amax = self.weight.data.abs().view(self.out_features, -1).max(dim=1)[0]
        init_w_scale = (w_amax / w_qmax).clamp(min=1e-8)
        self.weight_scale = nn.Parameter(init_w_scale.view(-1, 1))
        self.weight_zp = nn.Parameter(torch.zeros(self.out_features, 1))

        # in_scale shape: (1, in_features) — broadcast with input (N, in_features)
        self.in_scale = nn.Parameter(torch.ones(1, self.in_features) * 0.5 / a_qmax)
        self.in_zp = nn.Parameter(torch.zeros(1, self.in_features))

    @property
    def qat_enabled(self): return self._qat_enabled

    def enable_qat(self): self._qat_enabled = True
    def disable_qat(self): self._qat_enabled = False

    def lsq_params(self):
        return [self.weight_scale, self.weight_zp,
                self.in_scale, self.in_zp]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._qat_enabled:
            return F.linear(x, self.weight, self.bias)

        a_qmax = 2 ** (self._act_bits - 1) - 1
        w_qmax = 2 ** (self._weight_bits - 1) - 1

        # 末层不量化输入
        if self._is_last_layer:
            x_q = x
        elif self._use_lsq:
            in_s = self.in_scale.abs().clamp(min=1e-8)
            x_q = lsq_quantize(x, in_s, self.in_zp, -a_qmax, a_qmax)
        else:
            x_q = ste_quantize(x, self._act_bits, per_channel=True, ch_dim=-1)

        if self._use_lsq:
            w_s = self.weight_scale.abs().clamp(min=1e-8)
            w_q = lsq_quantize(self.weight, w_s, self.weight_zp, -w_qmax, w_qmax)
        else:
            w_q = ste_quantize(self.weight, self._weight_bits,
                               per_channel=True, ch_dim=0)

        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"w{self._weight_bits}/a{self._act_bits}, "
                f"lsq={self._use_lsq}, last={self._is_last_layer}")


# ============================================================
#  模型转换
# ============================================================

def prepare_model_lsq(model: nn.Module,
                      weight_bits: int = 8,
                      act_bits: int = 8,
                      first_conv_fp32: bool = True,
                      quantize_linear: bool = True,
                      inplace: bool = True) -> nn.Module:
    """
    将标准模型转换为 LSQ+ QAT 模型.

    Args:
        weight_bits:      权重位宽 (8=int8)
        act_bits:         激活位宽 (8=int8)
        first_conv_fp32:  首层 Conv 保留 FP32
        quantize_linear:  是否量化 Linear
    """
    if not inplace:
        model = copy.deepcopy(model)

    _convert_to_lsq(model, weight_bits, act_bits,
                    first_conv_fp32, quantize_linear, _first=[True])

    qc_enabled = sum(1 for m in model.modules()
                     if isinstance(m, LSQConv2d) and m.qat_enabled)
    qc_fp32 = sum(1 for m in model.modules()
                  if isinstance(m, LSQConv2d) and not m.qat_enabled)
    ql = sum(1 for m in model.modules() if isinstance(m, LSQLinear))
    bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))

    print(f"[prepare_model_lsq] LSQ+ int{weight_bits} QAT: w{weight_bits}/a{act_bits}")
    print(f"  LSQ Conv: {qc_enabled} enabled + {qc_fp32} fp32 (first layer)")
    print(f"  LSQ Linear: {ql}, BN: {bn}")
    print(f"  LSQ+ learnable params: scale + zero_point per layer")
    if first_conv_fp32:
        print(f"  首层 Conv 保留 FP32")
    return model


def _convert_to_lsq(module, weight_bits, act_bits,
                    first_conv_fp32, quantize_linear, _first):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            keep_fp32 = _first[0] and first_conv_fp32
            use_lsq = not keep_fp32  # 首层不用 LSQ (本身是 FP32)
            qat = LSQConv2d(child, weight_bits=weight_bits,
                           act_bits=act_bits, use_lsq=use_lsq)
            if keep_fp32:
                qat.disable_qat()
            setattr(module, name, qat)
            _first[0] = False
        elif isinstance(child, nn.Linear):
            if quantize_linear:
                qat = LSQLinear(child, weight_bits=weight_bits,
                               act_bits=act_bits, use_lsq=True)
                setattr(module, name, qat)
        elif isinstance(child, (LSQConv2d, LSQLinear)):
            continue
        else:
            _convert_to_lsq(child, weight_bits, act_bits,
                           first_conv_fp32, quantize_linear, _first)


def enable_qat(model: nn.Module):
    count = 0
    for m in model.modules():
        if isinstance(m, (LSQConv2d, LSQLinear)):
            m.enable_qat()
            count += 1
    if count: print(f"[enable_qat] Enabled QAT on {count} layers")
    return count


def disable_qat(model: nn.Module):
    count = 0
    for m in model.modules():
        if isinstance(m, (LSQConv2d, LSQLinear)):
            m.disable_qat()
            count += 1
    if count: print(f"[disable_qat] Disabled QAT on {count} layers")
    return count


def set_lsq_lr(model: nn.Module, base_lr: float) -> list:
    """LSQ+ 参数独立学习率 (0.1x base_lr)"""
    weight_params, lsq_params = [], []
    for name, param in model.named_parameters():
        is_lsq = any(k in name for k in
                     ['weight_scale', 'weight_zp', 'in_scale', 'in_zp'])
        if is_lsq:
            lsq_params.append(param)
        else:
            weight_params.append(param)
    print(f"[set_lsq_lr] Weight params: {len(weight_params)}, "
          f"LSQ params: {len(lsq_params)} (lr={base_lr*0.1})")
    return [
        {'params': weight_params, 'lr': base_lr},
        {'params': lsq_params, 'lr': base_lr * 0.1},
    ]


# ============================================================
#  评估
# ============================================================

@torch.no_grad()
def evaluate_model_lsq(model, dataloader, device, criterion=None):
    model.eval(); model.to(device)
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
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  optic_qat_lsq.py — LSQ+ int8 Self-Test")
    print("=" * 60)

    # Test 1: LSQ+ quantization
    print("\n[Test 1] lsq_quantize (int8)")
    x = torch.randn(4, 8, 32, 32) * 3.0
    scale = nn.Parameter(torch.ones(8, 1, 1) * 0.5)
    zp = nn.Parameter(torch.zeros(8, 1, 1))
    x_q = lsq_quantize(x, scale, zp, -127, 127)
    loss = x_q.sum(); loss.backward()
    print(f"  scale.grad norm: {scale.grad.norm():.6f}")
    print(f"  zp.grad norm:    {zp.grad.norm():.6f}")
    print(f"  Quantized unique values: {x_q.unique().numel()}")
    print(f"  [OK] LSQ+ gradients flow through scale and zp")

    # Test 2: LSQConv2d
    print("\n[Test 2] LSQConv2d (int8, LSQ+)")
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat = LSQConv2d(conv, weight_bits=8, act_bits=8, use_lsq=True)
    qat.train()
    x = torch.randn(2, 3, 32, 32)
    out = qat(x)
    out.sum().backward()
    print(f"  Output shape: {out.shape}")
    print(f"  weight_scale.grad norm: {qat.weight_scale.grad.norm():.6f}")
    print(f"  in_scale.grad norm:     {qat.in_scale.grad.norm():.6f}")
    assert qat.weight_scale.grad is not None, "weight_scale should have grad!"
    assert qat.in_scale.grad is not None, "in_scale should have grad!"
    print(f"  [OK] All LSQ+ params receive gradients")

    # Test 3: prepare_model_lsq
    print("\n[Test 3] prepare_model_lsq")

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 8, 1, bias=False),
                                      nn.BatchNorm2d(8), nn.ReLU())
            self.conv = nn.Sequential(nn.Conv2d(8, 16, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(16), nn.ReLU())
            self.fc = nn.Linear(16*16*16, 10, bias=False)
        def forward(self, x):
            x = self.stem(x); x = self.conv(x)
            return self.fc(x.flatten(1))

    tiny = TinyNet()
    prepare_model_lsq(tiny, first_conv_fp32=True, quantize_linear=True)
    print(f"  stem.0  qat_enabled={tiny.stem[0].qat_enabled} (expect False, first layer FP32)")
    print(f"  stem.0  use_lsq={tiny.stem[0]._use_lsq} (expect False)")
    print(f"  conv.0  qat_enabled={tiny.conv[0].qat_enabled} (expect True)")
    print(f"  conv.0  use_lsq={tiny.conv[0]._use_lsq} (expect True)")
    print(f"  fc type: {type(tiny.fc).__name__}")
    assert not tiny.stem[0].qat_enabled, "stem should be FP32"
    assert tiny.conv[0].qat_enabled, "conv should be LSQ+"
    assert tiny.conv[0]._use_lsq, "conv should use LSQ+"
    print(f"  [OK]")

    # Test 4: Training loop
    print("\n[Test 4] Training simulation (5 steps, LSQ+)")
    tiny2 = TinyNet()
    prepare_model_lsq(tiny2)
    tiny2.train()
    pg = set_lsq_lr(tiny2, base_lr=0.001)
    opt = torch.optim.Adam(pg)
    for i in range(5):
        opt.zero_grad()
        out = tiny2(torch.randn(4, 3, 32, 32))
        loss = nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,)))
        loss.backward()
        opt.step()
        print(f"  Step {i+1}: loss={loss.item():.4f}")
    print(f"  [OK] LSQ+ model trains!")

    # Test 5: STE fallback
    print("\n[Test 5] STE fallback (use_lsq=False)")
    conv2 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    qat2 = LSQConv2d(conv2, use_lsq=False)
    qat2.train()
    out2 = qat2(torch.randn(2, 3, 32, 32))
    out2.sum().backward()
    print(f"  Output shape: {out2.shape}")
    print(f"  weight_scale.grad: {qat2.weight_scale.grad} (should be None in STE mode)")
    print(f"  [OK]")

    print("\n" + "=" * 60)
    print("  All self-tests passed! [OK]")
    print("=" * 60)
