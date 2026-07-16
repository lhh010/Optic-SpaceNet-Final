"""
================================================================================
 optic_inference_phase4.py — Phase 4 光计算推理迁移与对比评估

 评估模式:
   - QAT 模式 (默认): 使用 QAT 层的 enable_qat/disable_qat 评估
     int4 伪量化 = 光计算模拟, 精度与训练日志一致
   - Optic 模式 (--optic): 使用 build_optical_model + osimulator
     硬件级光计算模拟 (含 im2col 展开、补零对齐、物理噪声)

 模型:
   Model 1 (Baseline VGG STE):     weights/baseline_vgg_phase4_ste.pth     训练 98.07%
   Model 2 (SpaceNet V1 STE):      weights/spacenet_v1_phase4_ste.pth      训练 92.87%
   Model 3 (SpaceNet V2 KD STE):   weights/spacenet_v2_phase4_ste.pth      训练 93.22%
   Model 1 (Baseline VGG LSQ+):    weights/baseline_vgg_phase4_lsqplus.pth 训练 61.72%

 用法:
   python optic_inference_phase4.py                  # QAT 模式全量评估
   python optic_inference_phase4.py --quick 10       # 快速测试
   python optic_inference_phase4.py --optic --quick 3  # Optic 模式 + 硬件模拟器

 参考:
   - noise_robustness_v2.py (QAT 评估模式)
   - optic_inference.py (FP32 baseline)
   - optic_layers.py (optical engine)
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

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42
DEFAULT_BATCH = 1  # 真实模拟器安全值

print(f"Device: {DEVICE}")


# ============================================================
#  模型架构 — 标准 PyTorch (训练原始架构)
# ============================================================

# --- Model 1 STE: 扁平架构 + BN, bias=False ---
class BaselineVGG_Flat(nn.Module):
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


# --- Model 1 LSQ+: Sequential 架构, bias=False ---
class BaselineVGG_Seq(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# --- Model 2/3: Sequential 架构 + BN, bias=False ---
class OpticSpaceNet_Phase4(nn.Module):
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
            nn.Linear(16 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
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
def load_data(batch_size=DEFAULT_BATCH):
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
    train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
    val_dataset = torch.utils.data.Subset(val_full, indices[:val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)} imgs, Val: {len(val_dataset)} imgs")
    print(f"Classes: {train_full.classes}")
    return train_loader, val_loader


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
        print_interval = max(1, effective_n // 10)  # 默认每 10% 打印

    for i, (images, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion:
            total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    # 循环结束后统一打印
    acc = correct / total if total > 0 else 0
    print(f"  [{desc}] {effective_n} batches — acc={acc:.2%}", flush=True)
    return {"accuracy": acc, "loss": total_loss / total if criterion else 0.0,
            "total": total, "correct": correct}


# ============================================================
#  QAT 模式评估 (主模式: 精度与训练一致)
# ============================================================
def evaluate_qat(model_class, weight_path, model_name, arch_label,
                 val_loader, device, qat_mode="v3", quick_batches=None):
    """
    QAT 模式: 用 QAT 层的 enable_qat/disable_qat 评估。
    enable_qat → int4 伪量化 = 光计算模拟.
    disable_qat → float32 全精度.
    精度与训练日志一致。
    """
    print(f"\n{'='*60}")
    print(f"  {model_name}  [QAT mode: {qat_mode}]")
    print(f"  Architecture: {arch_label}")
    print(f"{'='*60}")

    # Step 1: 创建标准模型
    print(f"\n  [1/3] Creating standard model...")
    model = model_class(num_classes=NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Step 2: 转换为 QAT 模型 (匹配训练时的 QAT 包装)
    print(f"\n  [2/3] Converting to QAT ({qat_mode})...")
    if qat_mode == "v3":
        from optic_qat_v3 import prepare_model_v3, enable_qat, disable_qat
        prepare_model_v3(model, mode="ste", weight_bits=4, act_bits=8,
                        noise=False, quantize_linear=True, preserve_bn=True)
        enable_fn, disable_fn = enable_qat, disable_qat
    else:  # v2 / lsqplus — preserve lsqplus mode with learnable scale/zp
        from optic_qat_v2 import prepare_model_phase4, QATConv2d_v2, QATLinear_v2
        # LSQ+ 训练用了 mode="lsqplus", 必须保持一致才能加载 scale/zp 参数
        prepare_model_phase4(model, mode="lsqplus", noise=False)
        def enable_fn(m):
            for mod in m.modules():
                if hasattr(mod, 'enable_qat'): mod.enable_qat()
        def disable_fn(m):
            for mod in m.modules():
                if hasattr(mod, 'disable_qat'): mod.disable_qat()

    # Step 3: 加载 QAT 权重
    print(f"\n  [3/3] Loading QAT weights...")
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight file not found: {weight_path}")
        return None
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded from: {weight_path}")

    # ----- float32 评估 -----
    print(f"\n  --- Native float32 evaluation ---")
    disable_fn(model)
    t0 = time.time()
    result_fp32 = evaluate(model, val_loader, device,
                           criterion=nn.CrossEntropyLoss(),
                           max_batches=quick_batches,
                           desc=f"{model_name} float32")
    fp32_time = time.time() - t0
    print(f"  Float32 Accuracy: {result_fp32['accuracy']:.2%}")
    print(f"  Float32 Loss:     {result_fp32['loss']:.4f}")
    print(f"  Float32 Time:     {fp32_time:.2f}s")

    # ----- int4 QAT (光计算模拟) 评估 -----
    print(f"\n  --- int4 QAT (optical computing simulation) evaluation ---")
    enable_fn(model)
    t0 = time.time()
    result_int4 = evaluate(model, val_loader, device,
                          criterion=nn.CrossEntropyLoss(),
                          max_batches=quick_batches,
                          desc=f"{model_name} int4-QAT")
    int4_time = time.time() - t0
    print(f"  Int4 QAT Accuracy: {result_int4['accuracy']:.2%}")
    print(f"  Int4 QAT Loss:     {result_int4['loss']:.4f}")
    print(f"  Int4 QAT Time:     {int4_time:.2f}s")

    return {
        "name": model_name, "arch": arch_label, "mode": f"QAT-{qat_mode}",
        "params": sum(p.numel() for p in model.parameters()),
        "fp32_acc": result_fp32["accuracy"],
        "int4_acc": result_int4["accuracy"],
        "quant_loss": result_fp32["accuracy"] - result_int4["accuracy"],
        "fp32_time": fp32_time, "int4_time": int4_time,
    }


# ============================================================
#  Optic 模式评估 (硬件级光计算模拟, 使用 osimulator)
# ============================================================
def evaluate_optic(model_class, weight_path, model_name, arch_label,
                   engine, val_loader, device, quick_batches=None):
    """
    Optic 模式: build_optical_model + osimulator 硬件级评估.
    """
    from optic_layers import (build_optical_model, compute_alignment_ratio,
                              print_alignment_detail)

    print(f"\n{'='*60}")
    print(f"  {model_name}  [Optic mode: osimulator]")
    print(f"{'='*60}")

    print(f"\n  [1/3] Building optical model...")
    model = model_class(num_classes=NUM_CLASSES)
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight not found: {weight_path}")
        return None
    state_dict = torch.load(weight_path, map_location='cpu')
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    skipped = len(state_dict) - len(filtered)
    if skipped: print(f"  Skipped {skipped} QAT params")
    print(f"  Weights loaded from: {weight_path}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print_alignment_detail(model, f"{model_name} (Original)")

    print(f"\n  [2/3] Converting to optical (OpticConv2d + OpticLinear)...")
    build_optical_model(model, engine, pad_to_8=True)
    print_alignment_detail(model, f"{model_name} (Optical)")

    print(f"\n  [3/3] Evaluating via osimulator...")
    from optic_layers import evaluate_model
    t0 = time.time()
    result = evaluate_model(model, val_loader, device,
                           criterion=nn.CrossEntropyLoss(),
                           max_batches=quick_batches,
                           desc=f"{model_name} optic",
                           print_interval=1)
    optic_time = time.time() - t0
    print(f"  Optical Accuracy: {result['accuracy']:.2%}")
    print(f"  Optical Time:     {optic_time:.2f}s")

    return {
        "name": model_name, "arch": arch_label, "mode": "Optic",
        "params": sum(p.numel() for p in model.parameters()),
        "optic_acc": result["accuracy"],
        "optic_time": optic_time,
    }


# ============================================================
#  报告
# ============================================================
def print_report(results_qat, results_optic):
    print("\n\n")
    print("=" * 100)
    print("  OPTIC-SPACENET PHASE 4: Optical Computing Inference Report")
    print("=" * 100)

    if results_qat:
        print(f"\n  [QAT Mode] int4 pseudo-quantization (matches training)")
        print(f"  {'Model':<30s} {'Params':>8s} {'FP32 Acc':>9s} "
              f"{'Int4 Acc':>9s} {'Quant Loss':>9s}")
        print("  " + "-" * 72)
        for r in results_qat:
            if r is None: continue
            print(f"  {r['name']:<30s} {r['params']:>8,} "
                  f"{r['fp32_acc']:>8.2%} {r['int4_acc']:>8.2%} "
                  f"{r['quant_loss']:>8.2%}")

    if results_optic:
        print(f"\n  [Optic Mode] osimulator hardware-level simulation")
        print(f"  {'Model':<30s} {'Params':>8s} {'Optic Acc':>9s} {'Time':>8s}")
        print("  " + "-" * 62)
        for r in results_optic:
            if r is None: continue
            print(f"  {r['name']:<30s} {r['params']:>8,} "
                  f"{r['optic_acc']:>8.2%} {r['optic_time']:>7.1f}s")

    print(f"\n  Reference (training logs — final enable_qat eval):")
    print(f"    Model 1 STE:  96.46% int4  (weights/baseline_vgg_phase4_ste.pth)")
    print(f"    Model 2 STE:  74.35% int4  (weights/spacenet_v1_phase4_ste.pth)")
    print(f"    Model 3 STE:  78.26% int4  (weights/spacenet_v2_phase4_ste.pth)")
    print(f"    Model 1 LSQ+: 61.72% int4  (weights/baseline_vgg_phase4_lsqplus.pth)")
    print("=" * 100)


# ============================================================
#  主函数
# ============================================================
def main():
    use_optic = "--optic" in sys.argv
    quick_batches = None
    batch_size = DEFAULT_BATCH
    for i, arg in enumerate(sys.argv):
        if arg == "--quick":
            quick_batches = int(sys.argv[i+1]) if i+1 < len(sys.argv) else 5
        if arg == "--batch":
            batch_size = int(sys.argv[i+1]) if i+1 < len(sys.argv) else DEFAULT_BATCH

    print("=" * 60)
    print("  Optic-SpaceNet Phase 4: Optical Inference Migration")
    print(f"  Mode: {'Optic (osimulator)' if use_optic else 'QAT (pseudo-quant)'}")
    print(f"  Batch={batch_size}" +
          (f", quick={quick_batches}" if quick_batches else ""))
    print("=" * 60)

    print("\n--- Loading Data ---")
    train_loader, val_loader = load_data(batch_size=batch_size)

    # 模型配置
    models_ste = [
        {"class": BaselineVGG_Flat, "weight": "weights/baseline_vgg_phase4_ste.pth",
         "name": "Model 1 Phase4 STE (VGG+BN)", "arch": "flat+BN", "qat": "v3"},
        {"class": OpticSpaceNet_Phase4, "weight": "weights/spacenet_v1_phase4_ste.pth",
         "name": "Model 2 Phase4 STE (SpaceNet V1)", "arch": "seq+BN", "qat": "v3"},
        {"class": OpticSpaceNet_Phase4, "weight": "weights/spacenet_v2_phase4_ste.pth",
         "name": "Model 3 Phase4 STE (KD+SpaceNet)", "arch": "seq+BN", "qat": "v3"},
    ]
    models_lsq = [
        {"class": BaselineVGG_Seq, "weight": "weights/baseline_vgg_phase4_lsqplus.pth",
         "name": "Model 1 Phase4 LSQ+ (VGG)", "arch": "seq", "qat": "v2"},
    ]

    results_qat = []
    results_optic = []

    if use_optic:
        # ---- Optic 模式: osimulator 硬件模拟 ----
        print("\n--- Initializing Optical Engine (osimulator) ---")
        from optic_layers import OpticalEngine
        engine = OpticalEngine(use_real=True)
        engine.reset_stats()

        for cfg in models_ste + models_lsq:
            r = evaluate_optic(cfg["class"], cfg["weight"], cfg["name"],
                              cfg["arch"], engine, val_loader, DEVICE,
                              quick_batches=quick_batches)
            results_optic.append(r)
            engine.reset_stats()
        print("\n--- Optical Engine Statistics ---")
        engine.print_stats()
    else:
        # ---- QAT 模式: 伪量化 (匹配训练精度) ----
        all_models = models_ste + models_lsq
        for cfg in all_models:
            r = evaluate_qat(cfg["class"], cfg["weight"], cfg["name"],
                           cfg["arch"], val_loader, DEVICE,
                           qat_mode=cfg["qat"],
                           quick_batches=quick_batches)
            results_qat.append(r)

    print_report(results_qat, results_optic)
    return results_qat, results_optic


if __name__ == "__main__":
    main()
