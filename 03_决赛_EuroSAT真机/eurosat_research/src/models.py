"""
===============================================================================
 models.py — 可参数化模型族 (Model 4 系列 + 搜索空间)
===============================================================================
 核心设计原则 (光计算第一性原理):
   - model 尺寸无限, activation compute (总 MACs) 高度受限
   - 权重存光阵列 → params 接近免费
   - 每层成本 = m×k×n MACs → 快速下采样到低分辨率 + 低分辨率宽层最优

 MiniVGG-GAP (Model 4):
   stem(3×3,s2) → 3 stages × 2 convs → GAP → Linear
   channels 可参数化 → 架构搜索直接复用

 扩展设计 (≤2M MACs 搜索空间):
   - ds_schedule: 下采样节奏 [stem_stride, pool次数]
   - head_dims: GAP 后 FC 头尺寸 (params 免费, MACs 近零)
===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_macs(model, input_size=(3, 64, 64)):
    """估算单图 MACs (与 osimulator MOPs 口径一致: conv 只计矩阵乘部分)。"""
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    x = torch.zeros(1, *input_size, device=device)
    total = 0
    hooks = []
    def hook_fn(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            b, c, h, w = inp[0].shape
            kh, kw = m.kernel_size
            oh = (h + 2 * m.padding[0] - m.dilation[0] * (kh - 1) - 1) // m.stride[0] + 1
            ow = (w + 2 * m.padding[1] - m.dilation[1] * (kw - 1) - 1) // m.stride[1] + 1
            total += b * oh * ow * (c * kh * kw) * m.out_channels
        elif isinstance(m, nn.Linear):
            total += inp[0].shape[0] * m.in_features * m.out_features
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return total


class BlurPool(nn.Module):
    """X0: 固定 3×3 二项式核 [1,2,1]⊗[1,2,1]/16 抗混叠下采样。
    stride=1 滤波 (padding=1 保尺寸) + stride=2 子采样, 等价于 stride=2 卷积。
    电侧算子 (不进光计算路径); 核注册为 buffer → 固定不可学, 且随 .pth 落盘可复现。"""

    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        kernel = (k[:, None] * k[None, :]) / 16.0
        self.register_buffer("kernel", kernel.expand(channels, 1, 3, 3).contiguous())
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=2, padding=1, groups=self.channels)


def make_downsample(channels, mode):
    """下采样算子 (R6 Q1 + X0):
       max       — MaxPool2d(2), 零 MACs
       avg       — AvgPool2d(2), 零 MACs
       stride1x1 — 1×1 conv s2 (可学习, 丢 3/4 位置)
       patchify  — PixelUnshuffle(2) + 1×1 mix 4C→C (保全位置 + 通道混合)
       max3      — MaxPool2d(3, s2, p1), 零 MACs, 每步 RF 贡献 (k−1)·j 翻倍 [X0]
       blur      — BlurPool 固定二项式核抗混叠, 电侧, 零 (光侧) MACs [X0]
       conv3s2   — 3×3 conv s2 + BN + ReLU (可学习, 光计算层, 花 MACs) [X0]"""
    if mode == "max":
        return nn.MaxPool2d(2)
    if mode == "avg":
        return nn.AvgPool2d(2)
    if mode == "stride1x1":
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
    if mode == "patchify":
        return nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
    if mode == "max3":
        return nn.MaxPool2d(3, stride=2, padding=1)
    if mode == "blur":
        return BlurPool(channels)
    if mode == "conv3s2":
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
    raise ValueError(f"Unknown pool_mode: {mode}")


class GlobalBypass(nn.Module):
    """R6 Q3: 全局信息旁路。stem 输出 → AvgPool 4× → Linear → ReLU。
    输出 concat 到 GAP 特征后, 给分类头提供全局布局线索。"""

    def __init__(self, in_channels, bypass_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 16, bypass_dim, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(self.pool(x))


class MiniVGG(nn.Module):
    """Model 4 MiniVGG-GAP, channels 可参数化。全 3×3 conv, bias=False, GAP head。

    fast_downsample=True 时 stage1 之后立即 pool (8×8→4×4), 跳过 8×8 上的第二层
    3×3 conv, 大幅压低 MACs — 光计算第一性原理: 低分辨率宽层。

    R6 扩展 (全部后向兼容):
      pool_mode     — 下采样算子 (max/avg/stride1x1/patchify), 见 make_downsample
      stem_kernel   — stem conv 核大小 (RF 臂)
      stage_depths  — (d1,d2,d3) 每 stage conv 层数, 覆盖 fast_downsample 逻辑
      bypass_dim    — >0 时启用 GlobalBypass (stem 输出 → 分类头前 concat)

    X0 扩展 (全部后向兼容):
      pool_mode 新增 max3 / blur / conv3s2 (见 make_downsample)。
        max3/blur 为电侧池化, 不进光计算路径; conv3s2 是 Conv2d, 会被
        prepare_model_v8 自动转为光计算层 (替代的原 1×1 也在光路, 行为一致)。
        注意 conv3s2 只替换 stage 的下采样点, stem 下采样仍用 MaxPool2d(2)
        (stem 保持全 FP32 电计算, 避免 stem.3.0 被无光刻标定地 QAT 化)。
      extra_pool    — True 时 stage3 末尾 (GAP 前) 追加一次 MaxPool2d(2),
        4×4→2×2, j=32, 每位置 RF 贡献 +(k−1)·j; 电侧, 零 MACs
      stem_pool_mode — 独立控制 stem 池化 (默认 None = 沿用 pool_mode 推导逻辑),
        用于组合臂如 pool_mode=conv3s2 + stem_pool_mode=max3
    """

    def __init__(self, channels=(32, 48, 72, 96), num_classes=10,
                 stem_stride=2, head_dims=(96,), bias=False,
                 fast_downsample=False, kernels=(3, 3, 3),
                 pool_mode="max", stem_kernel=3, stage_depths=None,
                 bypass_dim=0, extra_pool=False, stem_pool_mode=None):
        super().__init__()
        C0, C1, C2, C3 = channels
        k1, k2, k3 = kernels
        self.bypass_dim = bypass_dim
        self.extra_pool = extra_pool
        # X0: conv3s2 只替换 stage 下采样点, stem 保持 MaxPool (全 FP32 电计算)
        if stem_pool_mode is None:
            stem_pool_mode = "max" if pool_mode == "conv3s2" else pool_mode

        def stage(cin, cout, k, depth, pool):
            layers = []
            for i in range(depth):
                layers += [
                    nn.Conv2d(cin if i == 0 else cout, cout,
                              kernel_size=k, padding=k // 2, bias=bias),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                ]
            if pool:
                layers.append(make_downsample(cout, pool_mode))
            return nn.Sequential(*layers)

        if stage_depths is None:
            d1 = 1 if fast_downsample else 2
            d2, d3 = 2, 2
        else:
            d1, d2, d3 = stage_depths

        self.stem = nn.Sequential(
            nn.Conv2d(3, C0, kernel_size=stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2, bias=bias),
            nn.BatchNorm2d(C0), nn.ReLU(inplace=True),
            make_downsample(C0, stem_pool_mode),
        )
        self.stage1 = stage(C0, C1, k1, d1, pool=True)
        self.stage2 = stage(C1, C2, k2, d2, pool=True)
        self.stage3 = stage(C2, C3, k3, d3, pool=False)
        if extra_pool:
            # X0: 第 4 次下采样, stage3 末尾 (GAP 前), 电侧零 MACs
            self.stage3_pool = nn.MaxPool2d(2)

        if bypass_dim > 0:
            self.bypass = GlobalBypass(C0, bypass_dim)

        # GAP head: params 免费 (MACs ≈ C*num_classes, 可忽略)
        head_layers = []
        in_dim = C3 + bypass_dim
        for hd in head_dims:
            head_layers.append(nn.Linear(in_dim, hd, bias=True))
            head_layers.append(nn.ReLU(inplace=True))
            in_dim = hd
        head_layers.append(nn.Linear(in_dim, num_classes, bias=True))
        if bypass_dim > 0:
            # bypass 模式: GAP 与 FC 栈分离, forward 中 concat 旁路特征
            self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Sequential(*head_layers)
        else:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      *head_layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        bypass_feat = self.bypass(x) if self.bypass_dim > 0 else None
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if self.extra_pool:
            x = self.stage3_pool(x)
        if self.bypass_dim > 0:
            x = self.head(torch.cat([self.gap(x), bypass_feat], dim=1))
        else:
            x = self.head(x)
        return x


def build_model(cfg):
    """从配置构建模型。"""
    arch = cfg.get("arch", "minivgg_gap")
    if arch in ("minivgg_gap", "search"):
        return MiniVGG(
            channels=tuple(cfg["channels"]),
            num_classes=cfg["num_classes"],
            stem_stride=cfg.get("stem_stride", 2),
            head_dims=tuple(cfg.get("head_dims", [cfg["channels"][-1]])),
            bias=cfg.get("bias", False),
            fast_downsample=cfg.get("fast_downsample", False),
            kernels=tuple(cfg.get("kernels", [3, 3, 3])),
            pool_mode=cfg.get("pool_mode", "max"),
            stem_kernel=cfg.get("stem_kernel", 3),
            stage_depths=tuple(cfg["stage_depths"]) if cfg.get("stage_depths") else None,
            bypass_dim=cfg.get("bypass_dim", 0),
            extra_pool=cfg.get("extra_pool", False),
            stem_pool_mode=cfg.get("stem_pool_mode"),
        )
    raise ValueError(f"Unknown arch: {arch}")
