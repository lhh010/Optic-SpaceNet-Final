"""
================================================================================
 模型一 QAT (Baseline VGG): 标准微型 VGG — 3×3 卷积 → QAT 微调
================================================================================
 目的: 对预训练的 float32 Baseline VGG 进行 QAT 微调，使其适应
       int4 光计算推理，将量化精度损失降至最低。

 背景:
   - Float32 最佳准确率: 97.17% (60 epochs 标准训练)
   - 首次迁移 (PTQ): 直接 int4 量化推理，精度大幅下降 (~10-20%)
   - QAT 微调后预期:   ~95-97% (接近 float32 水平)

 训练策略:
   Phase 1 — 加载预训练 float32 权重
   Phase 2 — Conv+BN 融合 + QAT 层替换
   Phase 3 — QAT 微调 (低学习率, 15 epochs)
   Phase 4 — 保存 QAT-trained 权重, 对比评估

 与原有文件的关系:
   - model1_baseline.py:       标准 float32 训练 (产出 weights/baseline_vgg.pth)
   - model1_baseline_qat.py:   QAT 微调 (产出 weights/baseline_vgg_qat.pth)
   - optic_inference.py:        光计算推理评估 (使用 QAT 权重)

 用法:
   python model1_baseline_qat.py
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import numpy as np

# 导入 QAT 核心模块
from optic_qat import (
    fake_int4_quantize,
    QATConv2d, QATLinear,
    prepare_qat_model,
    enable_qat, disable_qat,
    calibrate_qat_model,
    evaluate_model,
    compare_qat_vs_float,
    compute_alignment_ratio,
    print_alignment_detail,
)

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
QAT_EPOCHS = 15          # QAT 微调轮数 (远少于标准训练的 60 轮)
LEARNING_RATE = 1e-4     # QAT 用低学习率 (标准训练的 1/10)
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
IMG_SIZE = 64
SEED = 42
FP32_WEIGHTS = "weights/baseline_vgg.pth"       # 预训练 float32 权重
QAT_WEIGHTS = "weights/baseline_vgg_qat.pth"    # QAT 微调后输出权重

print(f"设备: {DEVICE}")
print(f"数据目录: {DATA_DIR}")


# ============================================================
#  模型定义: Mini-VGG (全 3×3 卷积) — 与 model1_baseline.py 一致
# ============================================================
class BaselineVGG(nn.Module):
    """
    标准微型 VGG 网络，全部使用 3×3 卷积。
    - 3×3 展平长度为 9，在 8×2 光计算硬件上补零到 16，
      利用率仅 56.25% (单 kernel)，综合利用率约 99.8%。
    - 虽然对齐率看似高 (后续层 288/288, 576/576 等都被 8 整除)，
      但第一层 27→32 浪费了 15.6% 的硬件资源。
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: 3→32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 32→64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 64→128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 分类头
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


# ============================================================
#  数据加载
# ============================================================
def load_data():
    """加载 EuroSAT 数据集，返回 train/val DataLoader"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
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

    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")
    print(f"类别: {train_full.classes}")
    return train_loader, val_loader


# ============================================================
#  QAT 微调训练
# ============================================================
def train_qat_epoch(model, loader, criterion, optimizer):
    """QAT 训练一个 epoch (QAT 层自动伪量化)"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ============================================================
#  主函数
# ============================================================
def main():
    print("=" * 60)
    print("  模型一 QAT (Baseline VGG): int4 QAT 微调")
    print("=" * 60)

    # ---- 加载数据 ----
    train_loader, val_loader = load_data()

    # ---- Phase 1: 加载预训练 float32 模型 ----
    print(f"\n[Phase 1] 加载预训练 float32 权重: {FP32_WEIGHTS}")
    model = BaselineVGG(num_classes=NUM_CLASSES)

    if not os.path.exists(FP32_WEIGHTS):
        print(f"  [错误] 权重文件不存在: {FP32_WEIGHTS}")
        print(f"  请先运行 model1_baseline.py 进行标准训练。")
        return

    state_dict = torch.load(FP32_WEIGHTS, map_location='cpu')
    model.load_state_dict(state_dict)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,}")
    print(f"  权重加载成功!")

    # ---- 评估 float32 baseline ----
    print(f"\n[Phase 1b] 评估 float32 基准准确率 (eval mode)...")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    result_fp32 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Float32 验证准确率: {result_fp32['accuracy']:.2%}")
    print(f"  Float32 验证损失:   {result_fp32['loss']:.4f}")

    # ---- Phase 2: 准备 QAT 模型 ----
    print(f"\n[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...")

    # Step 2a: BN 融合前需要 eval 模式让 BN 有 running stats
    model.eval()
    with torch.no_grad():
        # 跑一个 batch 确保 BN 的 running_mean/var 已更新
        for images, _ in train_loader:
            _ = model(images.to(DEVICE))
            break

    # Step 2b: 转换为 QAT 模型 (自动融合 Conv+BN)
    prepare_qat_model(model, fuse_bn=True, inplace=True)
    qat_counts = {}
    for m in model.modules():
        if isinstance(m, QATConv2d):
            qat_counts['QATConv2d'] = qat_counts.get('QATConv2d', 0) + 1
        elif isinstance(m, QATLinear):
            qat_counts['QATLinear'] = qat_counts.get('QATLinear', 0) + 1
    print(f"  QAT 层: {qat_counts}")

    # ---- 打印硬件对齐率 ----
    alignment = print_alignment_detail(model, "BaselineVGG (QAT)")

    # ---- 校准 ----
    print(f"\n[Phase 2b] 校准 QAT 模型...")
    calibrate_qat_model(model, train_loader, DEVICE, num_batches=3)

    # ---- Phase 3: QAT 微调 ----
    print(f"\n[Phase 3] QAT 微调 ({QAT_EPOCHS} epochs, lr={LEARNING_RATE})")
    print(f"  说明: 使用低学习率 (1/10 of FP32 training)")
    print(f"        伪 int4 量化在每层前向时自动施加")
    print(f"        STE 梯度让模型学会对量化噪声具有鲁棒性")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

    best_acc = 0.0
    best_state = None
    total_train_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>6s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'Time':>7s}")
    print("  " + "-" * 65)

    for epoch in range(1, QAT_EPOCHS + 1):
        t0 = time.time()

        # QAT 训练 (模型在 train() 模式, QAT 层自动伪量化)
        train_loss, train_acc = train_qat_epoch(
            model, train_loader, criterion, optimizer
        )

        # 验证 (eval 模式, QAT 层也保持伪量化以模拟推理精度)
        model.eval()
        result_val = evaluate_model(model, val_loader, DEVICE, criterion)
        val_loss, val_acc = result_val['loss'], result_val['accuracy']
        elapsed = time.time() - t0
        total_train_time += elapsed

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  {epoch:>5d}  | {train_loss:>10.4f} {train_acc:>8.2%} | "
              f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | {elapsed:>6.1f}s")

    # 加载最佳权重
    model.load_state_dict(best_state)

    # ---- Phase 4: 评估与保存 ----
    print(f"\n[Phase 4] QAT 训练完成 — 评估与保存")

    # 4a: QAT vs Float 对比 (诊断 QAT 是否成功)
    comparison = compare_qat_vs_float(model, val_loader, DEVICE, criterion)

    # 4b: 导出并保存 QAT 权重
    print(f"\n  保存 QAT-trained 权重至: {QAT_WEIGHTS}")
    torch.save(model.state_dict(), QAT_WEIGHTS)
    print(f"  文件大小: {os.path.getsize(QAT_WEIGHTS) / 1024:.1f} KB")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  网络结构:          Mini-VGG (全 3×3 卷积)")
    print(f"  参数量:            {param_count:,}")
    print(f"  Float32 基准准确率: {result_fp32['accuracy']:.2%}")
    print(f"  QAT 最佳准确率:     {best_acc:.2%}")
    print(f"  QAT vs Float gap:  {comparison['accuracy_gap']:.2%}")
    print(f"  QAT 训练耗时:      {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  8×2 硬件对齐率:    {alignment:.1%}")
    print(f"\n  预期光计算推理:")
    print(f"    原生 float32:     {result_fp32['accuracy']:.1%}")
    print(f"    PTQ (直接 int4):  ~{result_fp32['accuracy'] - 0.10:.1%} (大幅下降)")
    print(f"    QAT (微调后 int4): {best_acc:.1%} (接近 float32)")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
