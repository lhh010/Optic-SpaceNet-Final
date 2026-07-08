"""
================================================================================
 noise_robustness_v2.py — int4 模型噪声鲁棒性测试 (Phase 4+)

 功能:
   对 int4 QAT 训练模型在推理中注入不同噪声，测试准确率衰减曲线。
   同时对比 FP32 baseline 模型在相同噪声下的表现。

 噪声类型 (模拟光计算物理噪声):
   1. Weight Quantization — 权重量化噪声 (int4/int3/int2)
   2. Activation Quantization — 激活量化噪声
   3. Gaussian Weight Noise — 权重高斯噪声 (MZI 相位误差)
   4. Gaussian Activation Noise — 激活读出噪声 (探测器热噪声)
   5. Channel Dropout — 通道随机丢弃 (波导故障/串扰极端情况)

 测试流程:
   - 对每种噪声, 在 5-7 个强度级别上评估准确率
   - 同时测试 QAT (int4) 和 Float32 模式
   - 生成准确率 vs 噪声强度曲线图
   - 输出汇总对比表

 用法:
   python noise_robustness_v2.py
   python noise_robustness_v2.py --model baseline_vgg_phase4_ste.pth --arch vgg
   python noise_robustness_v2.py --all  # 测试全部 3 个模型
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import time
import numpy as np
import copy

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

print(f"Device: {DEVICE}")


# ============================================================
#  模型架构
# ============================================================

class BaselineVGG(nn.Module):
    """Model 1: Mini-VGG + BN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=False)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
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
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    full = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    n = len(full)
    val_size = int(n * VAL_SPLIT)
    indices = list(range(n))
    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)
    val_dataset = torch.utils.data.Subset(full, indices[:val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    print(f"验证集: {len(val_dataset)} 张")
    return val_loader


# ============================================================
#  噪声注入器
# ============================================================

class WeightNoiseInjector:
    """向模型权重注入噪声"""

    def __init__(self, model, noise_type="gaussian", level=0.0,
                 target_bit=None):
        """
        Args:
            model:       模型
            noise_type:  "gaussian" | "quantization" | "dropout"
            level:       噪声强度
                         - gaussian: std = level * weight_std
                         - quantization: bit width = target_bit
                         - dropout: fraction of weights zeroed
            target_bit:  量化目标 bit (用于 quantization 类型)
        """
        self.model = model
        self.noise_type = noise_type
        self.level = level
        self.target_bit = target_bit
        self._saved_weights = {}

    def apply(self):
        """应用噪声到所有 Conv/Linear 权重"""
        self._saved_weights = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._saved_weights[name] = module.weight.data.clone()
                noisy_w = self._add_noise(module.weight.data, name)
                module.weight.data = noisy_w

    def remove(self):
        """恢复原始权重"""
        for name, module in self.model.named_modules():
            if name in self._saved_weights:
                module.weight.data = self._saved_weights[name]
        self._saved_weights = {}

    def _add_noise(self, weight, name):
        w = weight.clone()
        if self.noise_type == "gaussian":
            std = self.level * w.std().item()
            noise = torch.randn_like(w) * std
            return w + noise
        elif self.noise_type == "quantization":
            if self.target_bit is None:
                return w
            return self._quantize_to_bits(w, self.target_bit)
        elif self.noise_type == "dropout":
            mask = torch.rand_like(w) > self.level
            return w * mask.float()
        else:
            return w

    @staticmethod
    def _quantize_to_bits(w, bits):
        """量化张量到指定 bit"""
        qmax = 2 ** (bits - 1) - 1
        scale = (w.abs().max() / qmax).clamp(min=1e-8)
        q = (w / scale).round().clamp(-qmax, qmax)
        return q * scale


class ActivationNoiseInjector:
    """向模型激活注入噪声 (通过 hook)"""

    def __init__(self, noise_type="gaussian", level=0.0):
        self.noise_type = noise_type
        self.level = level
        self._hooks = []

    def apply(self, model):
        """注册 forward hook 到所有 ReLU 层"""
        def make_hook(noise_type, level):
            def hook(module, input, output):
                if noise_type == "gaussian":
                    std = level * output.std().item()
                    noise = torch.randn_like(output) * std
                    return output + noise
                elif noise_type == "quantization":
                    if level <= 0:
                        return output
                    bits = max(2, int(level))
                    qmax = 2 ** (bits - 1) - 1
                    scale = (output.max() / qmax).clamp(min=1e-8)
                    q = (output / scale).round().clamp(0, qmax)
                    return q * scale
                elif noise_type == "dropout":
                    mask = torch.rand_like(output) > level
                    return output * mask.float()
                return output
            return hook

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                h = module.register_forward_hook(
                    make_hook(self.noise_type, self.level))
                self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ============================================================
#  评估
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


# ============================================================
#  噪声扫描
# ============================================================

def run_noise_sweep(model, val_loader, device, noise_config,
                   use_qat=True):
    """
    对指定噪声类型在多个强度级别上评估准确率。

    Args:
        model:        模型
        val_loader:   验证数据
        device:       设备
        noise_config: dict, 噪声配置
        use_qat:      是否启用 QAT 模式 (int4 量化)

    Returns:
        {"name": str, "levels": [...], "accuracies": [...]}
    """
    name = noise_config["name"]
    levels = noise_config["levels"]
    noise_type = noise_config["type"]       # "weight" | "activation"
    noise_mode = noise_config["mode"]       # "gaussian" | "quantization" | "dropout"

    print(f"\n  {'='*50}")
    print(f"  噪声: {name}")
    print(f"  模式: {'QAT int4' if use_qat else 'FP32'}")
    print(f"  {'='*50}")

    # 尝试启用/禁用 QAT
    try:
        from optic_qat_v3 import enable_qat as v3_enable, disable_qat as v3_disable
        from optic_qat_v3 import QATConv2d_v3, QATLinear_v3
        has_qat = any(isinstance(m, (QATConv2d_v3, QATLinear_v3))
                     for m in model.modules())
    except ImportError:
        has_qat = False

    if has_qat:
        if use_qat:
            v3_enable(model)
        else:
            v3_disable(model)

    accuracies = []

    for level in levels:
        if noise_type == "weight":
            kwargs = {"noise_type": noise_mode, "level": level}
            if noise_mode == "quantization":
                kwargs["target_bit"] = int(level)
            injector = WeightNoiseInjector(model, **kwargs)
            injector.apply()
            acc = evaluate(model, val_loader, device)
            injector.remove()
        elif noise_type == "activation":
            injector = ActivationNoiseInjector(noise_type=noise_mode, level=level)
            injector.apply(model)
            acc = evaluate(model, val_loader, device)
            injector.remove()
        else:
            acc = evaluate(model, val_loader, device)

        accuracies.append(acc)
        if isinstance(level, float):
            print(f"    level={level:>8.4f}  |  Accuracy: {acc:.2%}")
        else:
            print(f"    level={level!s:>8s}  |  Accuracy: {acc:.2%}")

    return {"name": name, "levels": levels, "accuracies": accuracies}


# ============================================================
#  汇总与绘图
# ============================================================

def print_summary_table(all_results):
    """打印噪声鲁棒性汇总表"""
    print("\n\n")
    print("=" * 100)
    print("  NOISE ROBUSTNESS SUMMARY — int4 Optical Computing")
    print("=" * 100)

    model_names = list(all_results.keys())
    first_model = all_results[model_names[0]]
    noise_names = [s["name"] for s in first_model]

    for noise_idx, noise_name in enumerate(noise_names):
        print(f"\n  --- {noise_name} ---")
        levels = first_model[noise_idx]["levels"]

        header = f"  {'Level':>10s} |"
        for mn in model_names:
            header += f" {mn:<30s} |"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for level_idx, level in enumerate(levels):
            if isinstance(level, float):
                row = f"  {level:>10.4f} |"
            else:
                row = f"  {str(level):>10s} |"
            for mn in model_names:
                acc = all_results[mn][noise_idx]["accuracies"][level_idx]
                row += f" {acc:>29.2%} |"
            print(row)

    # 噪声容忍度 (准确率 > 80%)
    print(f"\n  --- Noise Tolerance (accuracy >= 80%) ---")
    print(f"  {'Noise Type':<30s} |", end="")
    for mn in model_names:
        print(f" {mn:<30s} |", end="")
    print()
    print("  " + "-" * 100)

    for noise_idx, noise_name in enumerate(noise_names):
        print(f"  {noise_name:<30s} |", end="")
        for mn in model_names:
            sweep = all_results[mn][noise_idx]
            tolerance = ">max"
            for level, acc in zip(sweep["levels"], sweep["accuracies"]):
                if acc < 0.80:
                    tolerance = f"< {level}"
                    break
            print(f" {tolerance:<30s} |", end="")
        print()

    print("=" * 100)


def generate_plot(all_results, save_path="noise_robustness_v2.png"):
    """生成噪声鲁棒性图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    noise_names = [s["name"] for s in all_results[list(all_results.keys())[0]]]
    model_names = list(all_results.keys())
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#E76F51', '#264653']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    n_noises = len(noise_names)
    n_cols = min(3, n_noises)
    n_rows = (n_noises + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, noise_name in enumerate(noise_names):
        ax = axes[i]
        for j, model_name in enumerate(model_names):
            sweep = all_results[model_name][i]
            levels = sweep["levels"]
            accs = sweep["accuracies"]

            # 转换 levels 为数值 (对于 quantization, 使用 bit 数)
            x_labels = levels
            x_values = list(range(len(levels)))

            ax.plot(x_values, [a * 100 for a in accs],
                    color=colors[j % len(colors)],
                    marker=markers[j % len(markers)],
                    linewidth=2, markersize=6, label=model_name)

        ax.set_title(noise_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Noise Level', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels], rotation=30, fontsize=7)
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    for i in range(n_noises, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('int4 QAT Model: Noise Robustness Analysis\n(Optical Computing Simulation)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  图表已保存: {save_path}")


# ============================================================
#  主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Noise Robustness Testing v2 — int4 QAT Models")
    print("=" * 60)

    # 解析参数
    test_all = "--all" in sys.argv
    model_path = None
    arch = "vgg"

    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
        if arg == "--arch" and i + 1 < len(sys.argv):
            arch = sys.argv[i + 1]

    val_loader = load_data()

    # 噪声配置
    noise_configs = [
        {
            "name": "Weight Quantization (bits)",
            "type": "weight",
            "mode": "quantization",
            "levels": [8, 6, 5, 4, 3, 2],
            "desc": "Effective weight bit-width",
        },
        {
            "name": "Weight Gaussian Noise (σ)",
            "type": "weight",
            "mode": "gaussian",
            "levels": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
            "desc": "Gaussian noise std relative to weight std",
        },
        {
            "name": "Activation Gaussian Noise (σ)",
            "type": "activation",
            "mode": "gaussian",
            "levels": [0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
            "desc": "Readout noise on activations",
        },
        {
            "name": "Activation Quantization (bits)",
            "type": "activation",
            "mode": "quantization",
            "levels": [8, 6, 5, 4, 3, 2],
            "desc": "Activation bit precision",
        },
        {
            "name": "Weight Dropout (fraction)",
            "type": "weight",
            "mode": "dropout",
            "levels": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
            "desc": "Random weight zeroing (waveguide failure)",
        },
    ]

    # 模型配置
    if test_all:
        model_configs = [
            {
                "name": "Model 1 (VGG int4)",
                "arch": "vgg",
                "path": "baseline_vgg_phase4_ste.pth",
                "use_qat": True,
            },
            {
                "name": "Model 1 (VGG FP32 ref)",
                "arch": "vgg",
                "path": "baseline_vgg_phase4_ste.pth",
                "use_qat": False,
            },
            {
                "name": "Model 2 (SpaceNet V1 int4)",
                "arch": "spacenet",
                "path": "spacenet_v1_phase4_ste.pth",
                "use_qat": True,
            },
            {
                "name": "Model 3 (SpaceNet V2 KD int4)",
                "arch": "spacenet",
                "path": "spacenet_v2_phase4_ste.pth",
                "use_qat": True,
            },
        ]
    elif model_path:
        model_configs = [
            {
                "name": f"Model ({arch}, int4)",
                "arch": arch,
                "path": model_path,
                "use_qat": True,
            },
            {
                "name": f"Model ({arch}, FP32 ref)",
                "arch": arch,
                "path": model_path,
                "use_qat": False,
            },
        ]
    else:
        # 默认: 测试 Model 1
        print("\n[INFO] 未指定模型, 使用默认 Model 1 (VGG)")
        print("  用法: python noise_robustness_v2.py --all  测试全部 3 个模型")
        print("        python noise_robustness_v2.py --model <path> --arch <vgg|spacenet>")
        model_configs = [
            {
                "name": "Model 1 (VGG int4)",
                "arch": "vgg",
                "path": "baseline_vgg_phase4_ste.pth",
                "use_qat": True,
            },
            {
                "name": "Model 1 (VGG FP32 ref)",
                "arch": "vgg",
                "path": "baseline_vgg_phase4_ste.pth",
                "use_qat": False,
            },
        ]

    all_results = {}

    for cfg in model_configs:
        model_name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"  测试: {model_name}")
        print(f"{'='*60}")

        # 创建模型
        if cfg["arch"] == "vgg":
            model = BaselineVGG(num_classes=NUM_CLASSES)
        else:
            model = OpticSpaceNet(num_classes=NUM_CLASSES)

        # 加载权重
        try:
            state = torch.load(cfg["path"], map_location='cpu')
            # 处理可能的 key 不匹配 (QAT 模型有额外参数)
            model_state = model.state_dict()
            filtered_state = {}
            for k, v in state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
            model.load_state_dict(filtered_state, strict=False)
            missing = sum(1 for k in model_state if k not in filtered_state)
            if missing > 0:
                print(f"  ⚠ {missing} keys missing (QAT params), using random init")
            print(f"  权重加载: {cfg['path']}")
        except FileNotFoundError:
            print(f"  ⚠ 权重文件未找到: {cfg['path']}")
            print(f"  将使用随机初始化 (仅供测试噪声框架)")
            print(f"  请先运行训练脚本生成权重文件")

        model.to(DEVICE)

        # 基线准确率 (无噪声)
        try:
            from optic_qat_v3 import enable_qat as v3_enable, disable_qat as v3_disable
            if cfg["use_qat"]:
                v3_enable(model)
            else:
                v3_disable(model)
        except ImportError:
            pass

        baseline_acc = evaluate(model, val_loader, DEVICE)
        print(f"  基线准确率 (无噪声, {'int4' if cfg['use_qat'] else 'FP32'}): {baseline_acc:.2%}")

        # 噪声扫描
        model_results = []
        for noise_cfg in noise_configs:
            sweep = run_noise_sweep(
                model, val_loader, DEVICE, noise_cfg,
                use_qat=cfg["use_qat"]
            )
            model_results.append(sweep)

        all_results[model_name] = model_results

    # 汇总
    if len(all_results) > 0:
        print_summary_table(all_results)
        generate_plot(all_results)

    print("\n" + "=" * 60)
    print("  噪声鲁棒性测试完成!")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
