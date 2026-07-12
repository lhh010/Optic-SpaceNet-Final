"""
================================================================================
 optic_inference_int8_model1.py — Model 1 INT8 光计算容器内推理 + MOPs 统计

 模型: Model 1 Baseline VGG Phase4 v3 (移植自 Model 2/3 v3 int8 配方)
   训练脚本:  model1_baseline_phase4_v3.py
   权重文件:  baseline_vgg_phase4_v3_int8.pth       (变体 A)
              baseline_vgg_phase4_v3_int8_vB.pth    (变体 B)
   QAT 模块:  optic_qat_v4.py (int8 权重 + int8 激活 + Gazelle 硬件噪声)
   架构:     Flat VGG (6 Conv 3×3 + 2 Linear), BN 保留, 全 bias=False, ~2.39M 参数

 变体 (光计算占比 vs 速度/精度 消融, 用户要求 >50%):
   --variant A (默认): 仅 conv1_1 电计算 → 光计算占比 97.7%
   --variant B:        conv1_1 + conv3_2 电计算 → 光计算占比 73.7%, osimulator 快 ~24%

 训练↔推理参数对齐 (§16.8 checklist):
   - conv1_1:        训练 first_conv_fp32=True ↔ 推理 keep_first_conv_electronic=True
   - 变体 B conv3_2: 训练 _keep_fp32=True      ↔ 推理 OpticConv2d→Conv2d 还原
   - 其余层:          训练 int8 QAT             ↔ 推理 OpticConv2d/OpticLinear (8a8w)
   - 噪声:           训练 GazelleNoise         ↔ 推理 osimulator 物理噪声

 评估模式 (容器内运行, 需要 osimulator):
   - Optic 模式 (默认): build_optical_model + OpticalEngine(use_real=True)
       Conv→OpticConv2d, Linear→OpticLinear, 矩阵乘法走 osimulator
   - QAT 模式 (--qat): PyTorch 伪量化交叉验证
   - MOPs 统计 (--mops-only): 仅打印各层 MOPs 与光计算占比

 ⚠️ Model 1 MACs 是 Model 2 的 ~150 倍 (156.6M vs 1.05M/张),
    全量 5400 张 ~9 天 — 仅用 --quick 抽样验证。

 用法 (在光计算 Docker 容器内):
   python optic_inference_int8_model1.py --variant A --mops-only      # MOPs 预览
   python optic_inference_int8_model1.py --variant A --qat             # QAT 全量(秒级)
   python optic_inference_int8_model1.py --variant A --quick 50        # osimulator 抽样
   python optic_inference_int8_model1.py --variant B --quick 50
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import time
import numpy as np

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED_TRAIN = 42      # 训练时使用的 seed (train/val split 与此一致)
DEFAULT_BATCH = 1    # osimulator 安全值

print(f"Device: {DEVICE}")


# ============================================================
#  模型架构 — Model 1 Baseline VGG (与训练脚本完全一致)
#  flat + BN + bias=False (匹配光计算硬件)
# ============================================================

class BaselineVGG(nn.Module):
    """
    Model 1 Phase4 v3: flat VGG, 全 bias=False, BN 保留。
    与 model1_baseline_phase4_v3.py 的 BaselineVGG 完全一致。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=False)
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
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================
#  变体配置: 哪些层保持电计算 (FP32)
# ============================================================

ELECTRONIC_LAYERS = {
    "A": {"conv1_1"},                # 仅首层
    "B": {"conv1_1", "conv3_2"},     # 首层 + conv3_2 (变体 B)
}


# ============================================================
#  数据加载 (独立测试集, 与 val 零重叠, 复用 Model 2 容器逻辑)
# ============================================================

def load_test_data(batch_size=DEFAULT_BATCH, test_ratio=0.2):
    """
    独立测试集 (与 optic_inference_int8.py 相同逻辑, seed=42):
      val  = indices[:test_size]            (训练时用于模型选择)
      train= indices[test_size:]            (训练时用于梯度更新)
      test = indices[test_size:test_size*2] (取自 train 段, 与 val 零重叠)
    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    n = len(full_dataset)
    test_size = int(n * test_ratio)

    indices = list(range(n))
    rng = np.random.RandomState(SEED_TRAIN)
    rng.shuffle(indices)

    test_indices = indices[test_size:test_size * 2]
    val_indices_set = set(indices[:test_size])
    overlap = len(val_indices_set & set(test_indices))
    assert overlap == 0, f"BUG: test/val overlap={overlap}!"

    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    print(f"Full: {n} | Train(used): {n-test_size} | Val(used): {test_size} "
          f"| Test(now): {len(test_dataset)} | Test/Val overlap: {overlap}")
    return test_loader


# ============================================================
#  评估工具
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None,
             max_batches=None, desc="Evaluating", print_interval=None):
    model.eval()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(dataloader)
    effective_n = min(n_batches, max_batches or n_batches)
    if print_interval is None:
        print_interval = max(1, effective_n // 10)

    for i, (images, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion:
            total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    acc = correct / total if total > 0 else 0
    print(f"  [{desc}] {effective_n} batches — acc={acc:.2%}", flush=True)
    return {"accuracy": acc, "loss": total_loss / total if criterion else 0.0,
            "total": total, "correct": correct}


# ============================================================
#  MOPs 计算 — Model 1 Baseline VGG (6 Conv + 2 Linear)
#  按变体标记 conv1_1 (/conv3_2) 为电计算
# ============================================================

INPUT_C, INPUT_H, INPUT_W = 3, 64, 64

# 每层结构 (不含 is_optical, 由变体决定): (name, type, C_in, C_out, Kh, Kw, stride, pad, pool)
_RAW_SPECS_M1 = [
    ("conv1_1", "Conv", 3,   32,  3, 3, 1, 1, None),
    ("conv1_2", "Conv", 32,  32,  3, 3, 1, 1, "Max2x2"),
    ("conv2_1", "Conv", 32,  64,  3, 3, 1, 1, None),
    ("conv2_2", "Conv", 64,  64,  3, 3, 1, 1, "Max2x2"),
    ("conv3_1", "Conv", 64,  128, 3, 3, 1, 1, None),
    ("conv3_2", "Conv", 128, 128, 3, 3, 1, 1, "Max2x2"),
    ("fc1",     "Linear", 8192, 256, 0, 0, 0, 0, None),
    ("fc2",     "Linear", 256,  10,  0, 0, 0, 0, None),
]


def layer_specs_for(variant):
    electronic = ELECTRONIC_LAYERS[variant]
    return [(n, t, ci, co, kh, kw, s, p, pool, (n not in electronic))
            for (n, t, ci, co, kh, kw, s, p, pool) in _RAW_SPECS_M1]


def _spatial(H, W, Kh, Kw, s, p):
    return (H + 2 * p - Kh) // s + 1, (W + 2 * p - Kw) // s + 1


def _pool(H, W, pool):
    if pool == "Max2x2":
        return H // 2, W // 2
    return H, W


def compute_mops_detail(variant):
    """逐层计算 MOPs, 区分电子计算 vs 光计算 (含补零对齐开销)."""
    specs = layer_specs_for(variant)
    H, W = INPUT_H, INPUT_W
    layers = []
    for name, ltype, ci, co, kh, kw, s, p, pool, is_opt in specs:
        if ltype == "Conv":
            patch_len = ci * kh * kw
            padded_len = ((patch_len + 7) // 8) * 8
            Ho, Wo = _spatial(H, W, kh, kw, s, p)
            raw = co * Ho * Wo * ci * kh * kw
            opt_m = Ho * Wo * padded_len * co
            elec_m = raw
            layers.append({"name": name, "type": ltype, "c_in": ci, "c_out": co,
                           "kernel": f"{kh}x{kw}", "spatial_in": f"{H}x{W}",
                           "spatial_out": f"{Ho}x{Wo}", "pool": pool or "None",
                           "patch_len": patch_len, "padded_len": padded_len,
                           "alignment": patch_len / padded_len if padded_len else 1,
                           "raw_mops": raw / 1e6,
                           "optical_mops": opt_m / 1e6 if is_opt else 0.0,
                           "electronic_mops": elec_m / 1e6 if not is_opt else 0.0,
                           "effective_mops": (opt_m if is_opt else elec_m) / 1e6,
                           "is_optical": is_opt})
            H, W = _pool(Ho, Wo, pool)
        else:  # Linear
            patch_len = ci
            padded_len = ((patch_len + 7) // 8) * 8
            raw = ci * co
            opt_m = padded_len * co
            elec_m = raw
            layers.append({"name": name, "type": ltype, "c_in": ci, "c_out": co,
                           "kernel": "-", "spatial_in": "-", "spatial_out": "-",
                           "pool": "None", "patch_len": patch_len, "padded_len": padded_len,
                           "alignment": patch_len / padded_len if padded_len else 1,
                           "raw_mops": raw / 1e6,
                           "optical_mops": opt_m / 1e6 if is_opt else 0.0,
                           "electronic_mops": elec_m / 1e6 if not is_opt else 0.0,
                           "effective_mops": (opt_m if is_opt else elec_m) / 1e6,
                           "is_optical": is_opt})

    total_raw = sum(l["raw_mops"] for l in layers)
    total_opt = sum(l["optical_mops"] for l in layers)
    total_elec = sum(l["electronic_mops"] for l in layers)
    total_eff = sum(l["effective_mops"] for l in layers)
    return layers, {
        "total_raw_mops": total_raw,
        "total_optical_mops": total_opt,
        "total_electronic_mops": total_elec,
        "total_effective_mops": total_eff,
        "optical_ratio": total_opt / total_eff if total_eff > 0 else 0,
        "optical_waste": total_opt - sum(l["raw_mops"] for l in layers if l["is_optical"]),
    }


def print_mops_report(layers, summary, variant):
    elec = ELECTRONIC_LAYERS[variant]
    print("\n")
    print("=" * 110)
    print(f"  Model 1 INT8 光计算 MOPs 统计 — Baseline VGG Phase4 v3 (变体 {variant})")
    print(f"  Gazelle 硬件: 8×2 tile, 8a8w12o | 电计算层 (FP32): {sorted(elec)}")
    print("=" * 110)
    print(f"\n  {'Layer':<10s} {'Type':<6s} {'C_in':>5s} {'C_out':>5s} "
          f"{'Kernel':>6s} {'Input':>10s} {'ConvOut':>10s} {'Pool':>6s} "
          f"{'Patch':>6s} {'Padded':>6s} {'Align':>7s} "
          f"{'RawMOPs':>10s} {'OptMOPs':>10s} {'ElecMOPs':>10s} {'Compute':>12s}")
    print("  " + "-" * 120)
    for l in layers:
        loc = "[Optical]" if l["is_optical"] else "[Electronic]"
        print(f"  {l['name']:<10s} {l['type']:<6s} {l['c_in']:>5d} {l['c_out']:>5d} "
              f"{l['kernel']:>6s} {l['spatial_in']:>10s} {l['spatial_out']:>10s} {l['pool']:>6s} "
              f"{l['patch_len']:>6d} {l['padded_len']:>6d} {l['alignment']:>6.1%} "
              f"{l['raw_mops']:>9.4f}M {l['optical_mops']:>9.4f}M {l['electronic_mops']:>9.4f}M "
              f"{loc:<12s}")
    print("  " + "-" * 120)
    print(f"  {'Total':<10s} {'':<6s} {'':>5s} {'':>5s} {'':>6s} {'':>10s} {'':>10s} {'':>6s} "
          f"{'':>6s} {'':>6s} {'':>7s} "
          f"{summary['total_raw_mops']:>9.4f}M {summary['total_optical_mops']:>9.4f}M "
          f"{summary['total_electronic_mops']:>9.4f}M")
    print(f"\n  {'-' * 60}")
    print(f"  [MOPs] 光计算占比汇总 (变体 {variant})")
    print(f"  {'-' * 60}")
    print(f"  总原始 MOPs:           {summary['total_raw_mops']:.4f} M")
    print(f"  光计算 MOPs (有效):    {summary['total_optical_mops']:.4f} M")
    print(f"  电子计算 MOPs:         {summary['total_electronic_mops']:.4f} M")
    print(f"  总有效 MOPs:           {summary['total_effective_mops']:.4f} M")
    print(f"  -------------------------------------")
    print(f"  ** 光计算占比:         {summary['optical_ratio']:.2%}  "
          f"({'✓ 达标 (≥50%)' if summary['optical_ratio'] >= 0.50 else '✗ 低于 50%!'})")
    if summary['optical_waste'] > 0:
        print(f"  光计算补零浪费:        {summary['optical_waste']:.4f} M (conv1_1 展平=27→32)")
    else:
        print(f"  光计算补零浪费:        0 (光计算层均对齐 8 的倍数) [OK]")
    print("=" * 110)


# ============================================================
#  QAT 模式评估 (int8 伪量化, 与训练配置严格一致)
# ============================================================

def evaluate_qat_int8(weight_path, test_loader, device, variant,
                      quick_batches=None):
    model_name = f"Model 1 Phase4 v3 INT8 (变体 {variant})"
    print(f"\n{'='*60}\n  {model_name}  [QAT mode: int8]\n{'='*60}")

    print(f"\n  [1/3] Creating model...")
    model = BaselineVGG(num_classes=NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n  [2/3] Converting to QAT v4 (int8, 首层 conv1_1 FP32"
          + (f" + conv3_2 FP32" if variant == "B" else "") + ")...")
    from optic_qat_v4 import prepare_model_v4, enable_qat, disable_qat
    prepare_model_v4(model, weight_bits=8, act_bits=8, noise=False,
                     first_conv_fp32=True, quantize_linear=True, preserve_bn=True)
    # 变体 B: conv3_2 也保持 FP32 (与训练一致)
    if variant == "B":
        from model1_baseline_phase4_v3 import keep_conv_fp32
        keep_conv_fp32(model, "conv3_2")

    print(f"\n  [3/3] Loading INT8 QAT weights: {weight_path}")
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight not found: {weight_path}")
        return None
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)

    # float32 评估
    print(f"\n  --- Native float32 (QAT disabled) ---")
    disable_qat(model)
    t0 = time.time()
    r_fp32 = evaluate(model, test_loader, device, nn.CrossEntropyLoss(),
                      quick_batches, f"{model_name} fp32")
    print(f"  Float32: {r_fp32['accuracy']:.2%} ({time.time()-t0:.1f}s)")

    # int8 QAT 评估
    print(f"\n  --- int8 QAT (光计算模拟) ---")
    enable_qat(model)
    if variant == "B":  # enable_qat 后重新确保 conv3_2 关闭
        keep_conv_fp32(model, "conv3_2")
    t0 = time.time()
    r_int8 = evaluate(model, test_loader, device, nn.CrossEntropyLoss(),
                      quick_batches, f"{model_name} int8")
    print(f"  Int8 QAT: {r_int8['accuracy']:.2%} ({time.time()-t0:.1f}s)")
    print(f"  Quant Loss: {r_fp32['accuracy']-r_int8['accuracy']:+.2%}")

    return {"name": model_name, "params": sum(p.numel() for p in model.parameters()),
            "fp32_acc": r_fp32["accuracy"], "int8_acc": r_int8["accuracy"],
            "quant_loss": r_fp32["accuracy"] - r_int8["accuracy"]}


# ============================================================
#  变体 B 推理后处理: OpticConv2d → 原生 Conv2d (保持电计算)
# ============================================================

def revert_optic_to_conv2d(model, conv_name):
    """
    把变体 B 中应保持电计算的层的 OpticConv2d 还原为原生 nn.Conv2d。
    OpticConv2d 已拷贝 in/out_channels, kernel_size, stride, padding,
    dilation, groups 及 weight/bias (见 optic_layers.py:618-636)。
    """
    from optic_layers import OpticConv2d
    old = getattr(model, conv_name)
    if not isinstance(old, OpticConv2d):
        return  # 已是原生 Conv2d 或未被转换
    native = nn.Conv2d(old.in_channels, old.out_channels,
                      kernel_size=old.kernel_size, stride=old.stride,
                      padding=old.padding, dilation=old.dilation,
                      groups=old.groups, bias=(old.bias is not None))
    native.weight = nn.Parameter(old.weight.data.clone())
    if old.bias is not None:
        native.bias = nn.Parameter(old.bias.data.clone())
    setattr(model, conv_name, native)
    print(f"  [Variant B] {conv_name} OpticConv2d → Conv2d (电计算 FP32): "
          f"{old.in_channels}→{old.out_channels}")


# ============================================================
#  Optic 模式评估 (osimulator 硬件级仿真)
# ============================================================

def evaluate_optic_int8(weight_path, engine, test_loader, device, variant,
                        quick_batches=None, is_quick_mode=False):
    from optic_layers import (build_optical_model, print_alignment_detail,
                              evaluate_model)
    model_name = f"Model 1 Phase4 v3 INT8 (变体 {variant})"
    print(f"\n{'='*60}\n  {model_name}  [Optic mode: osimulator]\n{'='*60}")

    print(f"\n  [1/3] Creating model & loading weights...")
    model = BaselineVGG(num_classes=NUM_CLASSES)
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight not found: {weight_path}")
        return None
    sd = torch.load(weight_path, map_location='cpu')
    ms = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    skipped = len(sd) - len(filtered)
    if skipped:
        print(f"  Skipped {skipped} QAT-specific params (expected)")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print_alignment_detail(model, f"{model_name} (Original FP32)")

    print(f"\n  [2/3] Converting to optical (int8, conv1_1 electronic"
          + (f" + conv3_2 electronic" if variant == "B" else "") + ")...")
    build_optical_model(model, engine, pad_to_8=True,
                        input_bit=8, weight_bit=8,
                        keep_first_conv_electronic=True)
    # 变体 B: conv3_2 还原为电计算 Conv2d
    if variant == "B":
        revert_optic_to_conv2d(model, "conv3_2")
    print_alignment_detail(model, f"{model_name} (Optical)")

    print(f"\n  [3/3] Evaluating via osimulator...")
    total_batches = quick_batches if quick_batches else len(test_loader)
    print_interval = 1 if is_quick_mode else max(1, total_batches // 10)
    t0 = time.time()
    result = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss(),
                            quick_batches, f"{model_name} optic", print_interval)
    t = time.time() - t0
    print(f"  Optical Accuracy: {result['accuracy']:.2%}  Time: {t:.1f}s")
    return {"name": model_name, "params": sum(p.numel() for p in model.parameters()),
            "optic_acc": result["accuracy"], "optic_time": t}


# ============================================================
#  主函数
# ============================================================

def main():
    # ---- 解析参数 ----
    use_qat = "--qat" in sys.argv
    mops_only = "--mops-only" in sys.argv
    quick_batches = None
    batch_size = DEFAULT_BATCH
    variant = "A"
    for i, a in enumerate(sys.argv):
        if a == "--quick":
            quick_batches = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 5
        if a == "--batch":
            batch_size = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else DEFAULT_BATCH
        if a == "--variant":
            variant = sys.argv[i + 1].upper().strip() if i + 1 < len(sys.argv) else "A"
    assert variant in ("A", "B"), f"未知变体: {variant} (应为 A 或 B)"

    weight_path = ("baseline_vgg_phase4_v3_int8.pth" if variant == "A"
                   else "baseline_vgg_phase4_v3_int8_vB.pth")
    mode_str = ("MOPs-only" if mops_only else
                "QAT (pseudo-quant)" if use_qat else "Optic (osimulator)")

    print("=" * 60)
    print("  Optic-SpaceNet Model 1 INT8: In-Container Optical Inference")
    print(f"  Model:  Baseline VGG Phase4 v3 (变体 {variant})")
    print(f"  Weight: {weight_path}")
    print(f"  Mode:   {mode_str}  |  Batch: {batch_size}"
          + (f"  quick={quick_batches}" if quick_batches else "  full"))
    print("=" * 60)

    layers, summary = compute_mops_detail(variant)
    if mops_only:
        print_mops_report(layers, summary, variant)
        return layers, summary

    # Model 1 速度警告
    if not use_qat:
        if quick_batches:
            est = quick_batches * (150 if variant == "A" else 115) / 60
            print(f"  ⚠️ Model 1 是 Model 2 的 ~150x 重 (156.6M vs 1.05M MACs/张)")
            print(f"  估计 {quick_batches} 张耗时: ~{est:.0f} min (变体 {variant})")
            print(f"  全量 5400 张 ~9 天 — 仅用 --quick 抽样")
        else:
            print(f"  ⚠️ 全量 5400 张 ~9 天! 建议: --quick 50")

    print("\n--- Loading Independent Test Set ---")
    test_loader = load_test_data(batch_size=batch_size)

    result_qat = None
    result_optic = None

    if use_qat:
        print("\n[Mode: QAT] PyTorch pseudo-quantization cross-validation...")
        result_qat = evaluate_qat_int8(weight_path, test_loader, DEVICE, variant,
                                       quick_batches=quick_batches)
    else:
        print("\n[Mode: Optic] Initializing Optical Engine (osimulator)...")
        is_quick = quick_batches is not None
        from optic_layers import OpticalEngine
        engine = OpticalEngine(use_real=True, verbose=is_quick)
        engine.reset_stats()
        result_optic = evaluate_optic_int8(weight_path, engine, test_loader, DEVICE,
                                           variant, quick_batches=quick_batches,
                                           is_quick_mode=is_quick)
        print("\n--- Optical Engine Statistics ---")
        engine.print_stats()

    # ---- 综合报告 ----
    print("\n" + "=" * 100)
    print(f"  Model 1 INT8 (变体 {variant}) — Container Verification Report")
    print("=" * 100)
    if result_qat:
        print(f"  QAT float32: {result_qat['fp32_acc']:.2%}  |  "
              f"QAT int8: {result_qat['int8_acc']:.2%}  |  "
              f"Quant Loss: {result_qat['quant_loss']:+.2%}")
    if result_optic:
        print(f"  Optic osimulator: {result_optic['optic_acc']:.2%}  |  "
              f"Time: {result_optic['optic_time']:.0f}s")
    print(f"  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%")
    print_mops_report(layers, summary, variant)
    print("=" * 100)

    return result_qat, result_optic, layers, summary


if __name__ == "__main__":
    main()
