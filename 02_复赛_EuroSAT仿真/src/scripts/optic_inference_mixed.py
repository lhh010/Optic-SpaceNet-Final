"""
================================================================================
 optic_inference_mixed.py — 混合精度光计算推理迁移与对比评估

 混合精度策略:
   Conv=int4 QAT (光计算) + Linear=fp32 (电计算)

 评估模式:
   - QAT 模式 (默认): QAT 层的 enable_qat/disable_qat
     int4 伪量化 = Conv 光计算模拟, Linear 保持 fp32
   - Optic 模式 (--optic): build_optical_model + osimulator

 模型:
   Model 1 (Baseline VGG Mixed):    weights/baseline_vgg_mixed_ste.pth  训练 98.26%
   Model 2 (SpaceNet V1 Mixed):     weights/spacenet_v1_mixed_ste.pth   训练 91.26%
   Model 3 (SpaceNet V2 KD Mixed):  weights/spacenet_v2_mixed_ste.pth   训练 91.13%

 用法:
   python optic_inference_mixed.py                  # QAT 模式全量
   python optic_inference_mixed.py --quick 10       # 快速测试
   python optic_inference_mixed.py --optic --quick 3  # Optic 硬件模式
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
DATA_DIR = "data/EuroSAT_RGB"
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42
DEFAULT_BATCH = 1

print(f"Device: {DEVICE}")


# ============================================================
#  模型架构 (匹配训练脚本)
# ============================================================

class BaselineVGG_Mixed(nn.Module):
    """Model 1 Mixed: Conv(int4/bias=False) + Linear(fp32/bias=True)"""
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
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=True)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

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


class OpticSpaceNet_Mixed(nn.Module):
    """Model 2/3 Mixed: Conv(int4/bias=False) + Linear(fp32/bias=True)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256, bias=True),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.classifier(x)
        return x


# ============================================================
def load_data(batch_size=DEFAULT_BATCH):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_full = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    n = len(train_full); val_size = int(n * VAL_SPLIT)
    indices = list(range(n))
    rng = np.random.RandomState(SEED); rng.shuffle(indices)
    train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
    val_dataset = torch.utils.data.Subset(val_full, indices[:val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Classes: {train_full.classes}")
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None,
             max_batches=None, desc="Evaluating", print_interval=None):
    model.eval(); model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(dataloader)
    effective_n = min(n_batches, max_batches or n_batches)
    if print_interval is None:
        print_interval = max(1, effective_n // 10)

    for i, (images, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches: break
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
#  QAT 模式: Conv=int4 QAT + Linear=fp32
# ============================================================
def evaluate_qat_mixed(model_class, weight_path, model_name, arch_label,
                       val_loader, device, quick_batches=None):
    print(f"\n{'='*60}")
    print(f"  {model_name}  [QAT Mixed: Conv=int4, Linear=fp32]")
    print(f"{'='*60}")

    print(f"\n  [1/3] Creating standard model...")
    model = model_class(num_classes=NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n  [2/3] Converting to Mixed QAT (Conv=int4, Linear=fp32)...")
    from optic_qat_v3 import prepare_model_v3, enable_qat, disable_qat
    prepare_model_v3(model, mode="ste", weight_bits=4, act_bits=8,
                    noise=False, quantize_linear=False, preserve_bn=True)

    print(f"\n  [3/3] Loading Mixed QAT weights...")
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight not found: {weight_path}"); return None
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded from: {weight_path}")

    # float32
    print(f"\n  --- float32 evaluation ---")
    disable_qat(model)
    t0 = time.time()
    r_fp32 = evaluate(model, val_loader, device, criterion=nn.CrossEntropyLoss(),
                      max_batches=quick_batches, desc=f"{model_name} float32")
    t_fp32 = time.time() - t0
    print(f"  Float32 Acc: {r_fp32['accuracy']:.2%}  Time: {t_fp32:.1f}s")

    # int4 (Conv 量化)
    print(f"\n  --- int4 Mixed (Conv=int4 光计算) evaluation ---")
    enable_qat(model)
    t0 = time.time()
    r_int4 = evaluate(model, val_loader, device, criterion=nn.CrossEntropyLoss(),
                     max_batches=quick_batches, desc=f"{model_name} int4-mixed")
    t_int4 = time.time() - t0
    print(f"  Int4 Mixed Acc: {r_int4['accuracy']:.2%}  Time: {t_int4:.1f}s")

    return {
        "name": model_name, "arch": arch_label, "mode": "Mixed-QAT",
        "params": sum(p.numel() for p in model.parameters()),
        "fp32_acc": r_fp32["accuracy"], "int4_acc": r_int4["accuracy"],
        "quant_loss": r_fp32["accuracy"] - r_int4["accuracy"],
        "fp32_time": t_fp32, "int4_time": t_int4,
    }


# ============================================================
#  Optic 模式
# ============================================================
def evaluate_optic_mixed(model_class, weight_path, model_name, arch_label,
                         engine, val_loader, device, quick_batches=None):
    from optic_layers import (build_optical_model, compute_alignment_ratio,
                              print_alignment_detail, evaluate_model)

    print(f"\n{'='*60}")
    print(f"  {model_name}  [Optic mode]")
    print(f"{'='*60}")

    print(f"\n  [1/3] Building optical model...")
    model = model_class(num_classes=NUM_CLASSES)
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight not found: {weight_path}"); return None
    state_dict = torch.load(weight_path, map_location='cpu')
    ms = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  Skipped {len(state_dict)-len(filtered)} QAT params")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print_alignment_detail(model, f"{model_name} (Original)")

    print(f"\n  [2/3] Converting to optical...")
    build_optical_model(model, engine, pad_to_8=True)
    print_alignment_detail(model, f"{model_name} (Optical)")

    print(f"\n  [3/3] Evaluating via osimulator...")
    t0 = time.time()
    result = evaluate_model(model, val_loader, device,
                           criterion=nn.CrossEntropyLoss(),
                           max_batches=quick_batches,
                           desc=f"{model_name} optic", print_interval=1)
    t = time.time() - t0
    print(f"  Optical Acc: {result['accuracy']:.2%}  Time: {t:.1f}s")
    return {"name": model_name, "arch": arch_label, "mode": "Optic",
            "params": sum(p.numel() for p in model.parameters()),
            "optic_acc": result["accuracy"], "optic_time": t}


# ============================================================
def print_report(results_qat, results_optic):
    print("\n\n" + "=" * 100)
    print("  OPTIC-SPACENET MIXED PRECISION: Optical Computing Inference Report")
    print("  Strategy: Conv=int4 (光计算) + Linear=fp32 (电计算)")
    print("=" * 100)

    if results_qat:
        print(f"\n  [QAT Mode] Mixed precision pseudo-quantization")
        print(f"  {'Model':<30s} {'Params':>8s} {'FP32 Acc':>9s} "
              f"{'Int4 Acc':>9s} {'Quant Loss':>9s}")
        print("  " + "-" * 72)
        for r in results_qat:
            if r is None: continue
            print(f"  {r['name']:<30s} {r['params']:>8,} "
                  f"{r['fp32_acc']:>8.2%} {r['int4_acc']:>8.2%} "
                  f"{r['quant_loss']:>8.2%}")

    if results_optic:
        print(f"\n  [Optic Mode] osimulator hardware simulation")
        print(f"  {'Model':<30s} {'Params':>8s} {'Optic Acc':>9s} {'Time':>8s}")
        print("  " + "-" * 62)
        for r in results_optic:
            if r is None: continue
            print(f"  {r['name']:<30s} {r['params']:>8,} "
                  f"{r['optic_acc']:>8.2%} {r['optic_time']:>7.1f}s")

    print(f"\n  Reference (training logs):")
    print(f"    Model 1 Mixed: 98.26% int4  (weights/baseline_vgg_mixed_ste.pth)")
    print(f"    Model 2 Mixed: 91.26% int4  (weights/spacenet_v1_mixed_ste.pth)")
    print(f"    Model 3 Mixed: 91.13% int4  (weights/spacenet_v2_mixed_ste.pth)")
    print("=" * 100)


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
    print("  Optic-SpaceNet Mixed: Optical Inference Migration")
    print(f"  Mode: {'Optic (osimulator)' if use_optic else 'QAT Mixed'}")
    print(f"  Batch={batch_size}" +
          (f", quick={quick_batches}" if quick_batches else ""))
    print("=" * 60)

    print("\n--- Loading Data ---")
    train_loader, val_loader = load_data(batch_size=batch_size)

    models = [
        {"class": BaselineVGG_Mixed, "weight": "weights/baseline_vgg_mixed_ste.pth",
         "name": "Model 1 Mixed (VGG+BN)", "arch": "flat+BN"},
        {"class": OpticSpaceNet_Mixed, "weight": "weights/spacenet_v1_mixed_ste.pth",
         "name": "Model 2 Mixed (SpaceNet V1)", "arch": "seq+BN"},
        {"class": OpticSpaceNet_Mixed, "weight": "weights/spacenet_v2_mixed_ste.pth",
         "name": "Model 3 Mixed (KD+SpaceNet)", "arch": "seq+BN"},
    ]

    results_qat, results_optic = [], []

    if use_optic:
        print("\n--- Initializing Optical Engine (osimulator) ---")
        from optic_layers import OpticalEngine
        engine = OpticalEngine(use_real=True)
        for cfg in models:
            r = evaluate_optic_mixed(cfg["class"], cfg["weight"], cfg["name"],
                                    cfg["arch"], engine, val_loader, DEVICE,
                                    quick_batches=quick_batches)
            results_optic.append(r)
            engine.reset_stats()
        engine.print_stats()
    else:
        for cfg in models:
            r = evaluate_qat_mixed(cfg["class"], cfg["weight"], cfg["name"],
                                  cfg["arch"], val_loader, DEVICE,
                                  quick_batches=quick_batches)
            results_qat.append(r)

    print_report(results_qat, results_optic)
    return results_qat, results_optic


if __name__ == "__main__":
    main()
