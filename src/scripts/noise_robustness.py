"""
================================================================================
 noise_robustness.py — 光计算噪声鲁棒性测试 (Part B)
================================================================================
 功能:
   对 3 个已训练模型在光计算推理中注入不同噪声，测试准确率衰减曲线。

 噪声类型 (模拟光计算物理噪声):
   1. Gaussian Readout — 加性高斯噪声 (探测器读出电路热噪声)
   2. Phase Noise      — 权重相位误差 (MZI 相位扰动)
   3. Shot Noise       — 光子散粒噪声 (光子计数统计)
   4. Crosstalk        — 通道间串扰 (波导/电学互连泄漏)
   5. Quantization     — 降低 bit 精度 (DAC/ADC 量化噪声)

 测试流程:
   - 对每种噪声，在 5-7 个强度级别上评估准确率
   - 生成准确率 vs 噪声强度曲线图
   - 输出汇总对比表

 参考: example_load_gazelle_model.py (Ltsimulator API)
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import time
import numpy as np

# 光计算库
from optic_layers import (
    OpticalEngine,
    build_optical_model,
    evaluate_model,
    GaussianReadoutNoise,
    PhaseNoise,
    ShotNoise,
    CrosstalkNoise,
)

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 32
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

print(f"Device: {DEVICE}")

# ============================================================
#  模型架构 (与训练脚本一致)
# ============================================================

class BaselineVGG(nn.Module):
    """Model 1: Mini-VGG, 3x3 convs"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


class OpticSpaceNet(nn.Module):
    """Model 2/3: Hardware-aligned CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x


# ============================================================
#  数据加载
# ============================================================
def load_data():
    """Load EuroSAT validation split"""
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_full = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    n = len(train_full)
    val_size = int(n * VAL_SPLIT)
    indices = list(range(n))
    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_dataset = torch.utils.data.Subset(train_full, train_idx)
    val_dataset = torch.utils.data.Subset(val_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader


# ============================================================
#  构建光计算模型
# ============================================================
def build_optic_model(model_class, weight_path, engine):
    """Build model, load weights, convert to optical version"""
    model = model_class(num_classes=NUM_CLASSES)
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    build_optical_model(model, engine, pad_to_8=True)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
#  噪声鲁棒性测试
# ============================================================
def run_noise_sweep(model, engine, val_loader, noise_class, levels, noise_name):
    """
    对指定噪声类型，在多个强度级别上评估准确率。

    Args:
        model:      光计算模型 (已加载权重)
        engine:     光计算引擎
        val_loader: 验证数据加载器
        noise_class: 噪声注入器类 (或字符串 "quantization")
        levels:     噪声强度列表
        noise_name: 噪声名称 (用于打印)

    Returns:
        {"levels": [...], "accuracies": [...], "name": str}
    """
    print(f"\n  {'='*50}")
    print(f"  Noise: {noise_name}")
    print(f"  {'='*50}")

    accuracies = []

    for level in levels:
        if noise_class == "quantization":
            # 量化噪声通过降低 bit 精度模拟
            # 需要重建 engine 并设置 bit 宽度
            engine.clear_noise()
            # 对于量化噪声，通过修改 input_bit/weight_bit 来模拟
            # 这里使用一个变通方法：临时修改 engine 的行为
            acc = evaluate_quantization_noise(model, val_loader, level)
        else:
            # 标准噪声注入
            injector = noise_class(level=level)
            engine.set_noise(injector)
            result = evaluate_model(model, val_loader, DEVICE)
            acc = result["accuracy"]
            engine.clear_noise()

        accuracies.append(acc)
        print(f"    level={level:>8.4f}  |  Accuracy: {acc:.2%}")

    return {"levels": levels, "accuracies": accuracies, "name": noise_name}


def evaluate_quantization_noise(model, val_loader, bits):
    """
    评估给定位宽下的量化噪声。
    通过临时修改所有 OpticConv2d/OpticLinear 的量化参数来模拟。

    Args:
        model: 光计算模型
        val_loader: 数据加载器
        bits: int, bit 宽度 (2, 3, 4, 5, 6, 8)
    Returns:
        accuracy (float)
    """
    from optic_layers import OpticConv2d, OpticLinear, quantize_int4

    # 保存原始 forward 方法
    original_forwards = {}
    for name, module in model.named_modules():
        if isinstance(module, (OpticConv2d, OpticLinear)):
            original_forwards[name] = module.forward

    # 修改所有光计算层的量化 bit
    def make_optic_conv_forward(original_forward, bit_width):
        def new_forward(self, x):
            N, C, H, W = x.shape
            kh, kw = self.kernel_size
            OH = (H + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
            OW = (W + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
            L = OH * OW

            # 用量化后的 bit_width
            x_q = quantize_nbit(x, dim=1, bits=bit_width, signed=True)
            w_q = quantize_nbit(self.weight, dim=0, bits=bit_width, signed=True)

            x_unfold = torch.nn.functional.unfold(
                x_q, kernel_size=(kh, kw),
                stride=self.stride, padding=self.padding, dilation=self.dilation
            )
            x_mat = x_unfold.transpose(1, 2).reshape(N * L, C * kh * kw)
            w_mat = w_q.reshape(self.out_channels, -1).t()

            if self.pad_to_8 and self._padded_len > self._patch_len:
                pad_amount = self._padded_len - self._patch_len
                x_mat = torch.nn.functional.pad(x_mat, (0, pad_amount), value=0.0)
                w_mat = torch.nn.functional.pad(w_mat, (0, 0, 0, pad_amount), value=0.0)

            result = self.engine.matmul(x_mat, w_mat, quantize_inputs=False)
            result = result.reshape(N, L, self.out_channels)
            result = result.transpose(1, 2).reshape(N, self.out_channels, OH, OW)

            if self.bias is not None:
                result = result + self.bias.view(1, -1, 1, 1)
            return result
        return new_forward

    def make_optic_linear_forward(original_forward, bit_width):
        def new_forward(self, x):
            N = x.shape[0]
            x_q = quantize_nbit(x, dim=-1, bits=bit_width, signed=True)
            w_q = quantize_nbit(self.weight, dim=0, bits=bit_width, signed=True)
            x_mat = x_q.unsqueeze(1)
            w_mat = w_q.t()

            if self.pad_to_8 and self._padded_len > self._patch_len:
                pad_amount = self._padded_len - self._patch_len
                x_mat = torch.nn.functional.pad(x_mat, (0, pad_amount), value=0.0)
                w_mat = torch.nn.functional.pad(w_mat, (0, 0, 0, pad_amount), value=0.0)

            result = self.engine.matmul(x_mat, w_mat, quantize_inputs=False)
            result = result.squeeze(1)
            if self.bias is not None:
                result = result + self.bias
            return result
        return new_forward

    # 替换 forward
    for name, module in model.named_modules():
        if isinstance(module, OpticConv2d):
            bound_method = make_optic_conv_forward(module.forward, bits)
            module.forward = bound_method.__get__(module, OpticConv2d)
        elif isinstance(module, OpticLinear):
            bound_method = make_optic_linear_forward(module.forward, bits)
            module.forward = bound_method.__get__(module, OpticLinear)

    # 评估
    result = evaluate_model(model, val_loader, DEVICE)
    acc = result["accuracy"]

    # 恢复原始 forward
    for name, module in model.named_modules():
        if name in original_forwards:
            module.forward = original_forwards[name]

    return acc


def quantize_nbit(tensor: torch.Tensor, dim: int = None,
                  bits: int = 4, signed: bool = True) -> torch.Tensor:
    """
    将浮点张量量化为指定 bit 宽度，再反量化。

    Args:
        tensor: 浮点张量
        dim:    量化维度 (None=per-tensor)
        bits:   bit 宽度
        signed: True=symmetric signed, False=unsigned
    """
    if signed:
        max_val = 2 ** (bits - 1) - 1  # e.g., 4 bits -> 7
        if dim is None:
            abs_max = max(tensor.abs().max(), 1e-8)
            scale = abs_max / max_val
            q = (tensor / scale).round().clamp(-max_val, max_val)
            return q * scale
        else:
            reduce_dims = [d for d in range(tensor.dim()) if d != dim]
            abs_max = tensor.abs()
            for d in sorted(reduce_dims, reverse=True):
                abs_max = abs_max.max(dim=d, keepdim=True)[0]
            abs_max = torch.where(abs_max < 1e-8, torch.ones_like(abs_max), abs_max)
            scale = abs_max / max_val
            q = (tensor / scale).round().clamp(-max_val, max_val)
            return q * scale
    else:
        max_val = 2 ** bits - 1  # e.g., 4 bits -> 15
        if dim is None:
            t_min = tensor.min()
            t_max = tensor.max()
            if t_max - t_min < 1e-8:
                return tensor
            scale = (t_max - t_min) / max_val
            zero_point = t_min
            q = ((tensor - zero_point) / scale).round().clamp(0, max_val)
            return q * scale + zero_point
        else:
            reduce_dims = [d for d in range(tensor.dim()) if d != dim]
            t_min = tensor
            t_max = tensor
            for d in sorted(reduce_dims, reverse=True):
                t_min = t_min.min(dim=d, keepdim=True)[0]
                t_max = t_max.max(dim=d, keepdim=True)[0]
            scale = (t_max - t_min) / max_val
            scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
            zero_point = t_min
            q = ((tensor - zero_point) / scale).round().clamp(0, max_val)
            return q * scale + zero_point


# ============================================================
#  汇总报告与绘图
# ============================================================
def print_summary_table(all_results: dict):
    """
    打印噪声鲁棒性汇总表。
    all_results: {model_name: [sweep_result, ...]}
    每个 sweep_result: {"name": str, "levels": [...], "accuracies": [...]}
    """
    print("\n\n")
    print("=" * 100)
    print("  NOISE ROBUSTNESS SUMMARY")
    print("=" * 100)

    model_names = list(all_results.keys())
    # 获取噪声名称列表 (从第一个模型的 sweeps)
    first_model_results = all_results[model_names[0]]
    noise_names = [s["name"] for s in first_model_results]

    for noise_idx, noise_name in enumerate(noise_names):
        print(f"\n  --- {noise_name} ---")
        levels = first_model_results[noise_idx]["levels"]

        # 表头
        header = f"  {'Level':>10s} |"
        for mn in model_names:
            header += f" {mn:<28s} |"
        print(header)
        print("  " + "-" * (len(header) - 2))

        # 每个噪声级别的准确率
        for level_idx, level in enumerate(levels):
            row = f"  {level:>10.4f} |"
            for mn in model_names:
                acc = all_results[mn][noise_idx]["accuracies"][level_idx]
                row += f" {acc:>28.2%} |"
            print(row)

    # 关键指标: 准确率降到 80% 时的噪声水平
    print(f"\n  --- Noise Tolerance (accuracy >= 80%) ---")
    print(f"  {'Noise Type':<25s} |", end="")
    for mn in model_names:
        print(f" {mn:<28s} |", end="")
    print()
    print("  " + "-" * 100)

    for noise_idx, noise_name in enumerate(noise_names):
        print(f"  {noise_name:<25s} |", end="")
        for mn in model_names:
            sweep = all_results[mn][noise_idx]
            # Find first level where accuracy drops below 80%
            tolerance = ">max"
            for level, acc in zip(sweep["levels"], sweep["accuracies"]):
                if acc < 0.80:
                    tolerance = f"< {level:.4f}"
                    break
            print(f" {tolerance:<28s} |", end="")
        print()

    print("=" * 100)


def generate_noise_plot(all_results: dict, save_path: str = "docs/figures/noise_robustness.png"):
    """
    Generate noise robustness plots using matplotlib.
    5 subplots (one per noise type), 3 lines per subplot (one per model).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot generation.")
        print(f"       Install with: pip install matplotlib")
        return

    noise_names = [s["name"] for s in all_results[list(all_results.keys())[0]]]
    model_names = list(all_results.keys())
    colors = ['#E63946', '#457B9D', '#2A9D8F']  # Red, Blue, Green
    markers = ['o', 's', '^']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, noise_name in enumerate(noise_names):
        ax = axes[i]
        noise_idx = i

        for j, model_name in enumerate(model_names):
            sweep = all_results[model_name][noise_idx]
            levels = sweep["levels"]
            accuracies = sweep["accuracies"]

            ax.plot(levels, [a * 100 for a in accuracies],
                    color=colors[j], marker=markers[j],
                    linewidth=2, markersize=6,
                    label=model_name)

        ax.set_title(noise_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Noise Level', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    # 隐藏多余的 subplot
    for i in range(len(noise_names), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Optic-SpaceNet: Noise Robustness Analysis (int4 Optical Computing)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {save_path}")


# ============================================================
#  主函数
# ============================================================
def main():
    print("=" * 60)
    print("  Optic-SpaceNet: Noise Robustness Testing (Part B)")
    print("  Optical precision: int4 (signed, [-8, 7])")
    print("=" * 60)

    # 加载数据
    print("\n--- Loading Data ---")
    train_loader, val_loader = load_data()

    # 创建引擎
    print("\n--- Initializing Optical Engine ---")
    engine = OpticalEngine(use_real=True)

    # 定义模型
    models_config = [
        {"class": BaselineVGG,   "weight": "weights/baseline_vgg.pth",          "name": "Model 1 (Baseline VGG)"},
        {"class": OpticSpaceNet, "weight": "weights/spacenet_v1.pth",           "name": "Model 2 (SpaceNet V1)"},
        {"class": OpticSpaceNet, "weight": "weights/spacenet_v2_distilled.pth", "name": "Model 3 (SpaceNet V2 KD)"},
    ]

    # 定义噪声配置
    noise_configs = [
        {
            "class": GaussianReadoutNoise,
            "name": "Gaussian Readout",
            "levels": [0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
            "description": "sigma relative to output std",
        },
        {
            "class": PhaseNoise,
            "name": "Phase Noise (MZI)",
            "levels": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
            "description": "fraction of max|W| perturbed",
        },
        {
            "class": ShotNoise,
            "name": "Photon Shot Noise",
            "levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "description": "normalized photon starvation",
        },
        {
            "class": CrosstalkNoise,
            "name": "Channel Crosstalk",
            "levels": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
            "description": "inter-channel mixing ratio",
        },
        {
            "class": "quantization",
            "name": "Bit Precision (Quantization)",
            "levels": [8, 6, 5, 4, 3, 2],
            "description": "effective bit-width (intN)",
        },
    ]

    # 存储所有结果
    all_results = {}

    # 对每个模型进行噪声鲁棒性测试
    for cfg in models_config:
        model_name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"  Testing: {model_name}")
        print(f"{'='*60}")

        # 构建光计算模型
        print("  Building optical model...")
        model = build_optic_model(cfg["class"], cfg["weight"], engine)
        print(f"  Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

        # 先评估无噪声 baseline
        engine.clear_noise()
        baseline = evaluate_model(model, val_loader, DEVICE)
        print(f"  Baseline (no noise): {baseline['accuracy']:.2%}")

        # 对每种噪声进行扫描
        model_results = []
        for noise_cfg in noise_configs:
            sweep = run_noise_sweep(
                model=model,
                engine=engine,
                val_loader=val_loader,
                noise_class=noise_cfg["class"],
                levels=noise_cfg["levels"],
                noise_name=noise_cfg["name"],
            )
            model_results.append(sweep)

        all_results[model_name] = model_results
        # 清理
        engine.clear_noise()
        engine.reset_stats()

    # 打印汇总
    print_summary_table(all_results)

    # 生成图表
    print("\n--- Generating Noise Robustness Plot ---")
    generate_noise_plot(all_results, save_path="docs/figures/noise_robustness.png")

    print("\n" + "=" * 60)
    print("  Noise robustness testing complete!")
    print("  Results saved to: docs/figures/noise_robustness.png")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
