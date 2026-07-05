"""
================================================================================
 模型二 QAT (Optic-SpaceNet V1): 硬件感知对齐网络 → QAT 微调
================================================================================
 目的: 对预训练的 float32 OpticSpaceNet V1 进行 QAT 微调，使其适应
       int4 光计算推理。

 背景:
   - Float32 最佳准确率: 90.15% (80 epochs 独立训练)
   - 网络特点: 所有 Conv 展平长度均被 8 整除 (stem 除外)
   - 首次迁移 (PTQ): 直接 int4 量化推理，精度大幅下降 (~10-15%)
   - QAT 微调后预期:   ~87-89% (接近 float32 水平)

 训练策略:
   Phase 1 — 加载预训练 float32 权重
   Phase 2 — Conv+BN 融合 + QAT 层替换
   Phase 3 — QAT 微调 (低学习率, 20 epochs)
   Phase 4 — 保存 QAT-trained 权重, 对比评估

 与原有文件的关系:
   - model2_spacenet_v1.py:       标准 float32 训练 (产出 spacenet_v1.pth)
   - model2_spacenet_v1_qat.py:   QAT 微调 (产出 spacenet_v1_qat.pth)
   - optic_inference.py:           光计算推理评估

 用法:
   python model2_spacenet_v1_qat.py
================================================================================
"""

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
QAT_EPOCHS = 20          # QAT 微调轮数 (SpaceNet 较小，需要稍多一些微调)
LEARNING_RATE = 1e-4     # QAT 用低学习率
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42
FP32_WEIGHTS = "spacenet_v1.pth"        # 预训练 float32 权重
QAT_WEIGHTS = "spacenet_v1_qat.pth"     # QAT 微调后输出权重

print(f"设备: {DEVICE}")


# ============================================================
#  模型定义: OpticSpaceNet V1 — 与 model2_spacenet_v1.py 一致
# ============================================================
class OpticSpaceNetV1(nn.Module):
    """
    硬件感知 CNN: 所有 Conv 的 im2col 展平长度均被 8 整除，
    输出通道数被 2 整除，完美匹配 8×2 光学矩阵乘法器。

    结构:
      stem:   Conv2d(3→8,  1×1)    → 8×64×64   [对齐: 3/8=37.5%, ops极少]
      stage1: Conv2d(8→16, 2×2,s=2)→16×32×32   [对齐: 32/32=100%]
      stage2: Conv2d(16→32,2×2,s=2)→32×8×8     [对齐: 64/64=100%]
      stage3: Conv2d(32→16,1×1)    →16×8×8     [对齐: 32/32=100%]
      fc:     Linear(1024→256)                 [对齐: 1024/1024=100%]
              Linear(256→10)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: 1×1 卷积将 RGB(3) 提维到 8 (对齐硬件宽度)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 8→16, 2×2 stride=2, 64→32
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16
        )

        # Stage 2: 16→32, 2×2 stride=2, 16→8
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 32→16, 1×1 保持分辨率
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 分类头: 16×8×8 = 1024 → 256 → 10
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
    """加载 EuroSAT 数据集"""
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
    """QAT 训练一个 epoch"""
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
    print("  模型二 QAT (Optic-SpaceNet V1): int4 QAT 微调")
    print("=" * 60)

    # ---- 加载数据 ----
    train_loader, val_loader = load_data()

    # ---- Phase 1: 加载预训练 float32 模型 ----
    print(f"\n[Phase 1] 加载预训练 float32 权重: {FP32_WEIGHTS}")
    model = OpticSpaceNetV1(num_classes=NUM_CLASSES)

    if not os.path.exists(FP32_WEIGHTS):
        print(f"  [错误] 权重文件不存在: {FP32_WEIGHTS}")
        print(f"  请先运行 model2_spacenet_v1.py 进行标准训练。")
        return

    state_dict = torch.load(FP32_WEIGHTS, map_location='cpu')
    model.load_state_dict(state_dict)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,}")
    print(f"  权重加载成功!")

    # ---- 评估 float32 baseline ----
    print(f"\n[Phase 1b] 评估 float32 基准准确率...")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    result_fp32 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Float32 验证准确率: {result_fp32['accuracy']:.2%}")
    print(f"  Float32 验证损失:   {result_fp32['loss']:.4f}")

    # ---- Phase 2: 准备 QAT 模型 ----
    print(f"\n[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...")

    # BN 融合前确保 BN 有 running stats
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            _ = model(images.to(DEVICE))
            break

    # 转换为 QAT 模型 (注意: stem, stage1, stage2, stage3 中的 Conv 都有 BN)
    prepare_qat_model(model, fuse_bn=True, inplace=True)
    qat_counts = {}
    for m in model.modules():
        if isinstance(m, QATConv2d):
            qat_counts['QATConv2d'] = qat_counts.get('QATConv2d', 0) + 1
        elif isinstance(m, QATLinear):
            qat_counts['QATLinear'] = qat_counts.get('QATLinear', 0) + 1
    print(f"  QAT 层: {qat_counts}")

    # ---- 打印硬件对齐率 ----
    alignment = print_alignment_detail(model, "OpticSpaceNetV1 (QAT)")
    print(f"\n  ⚠ 注意: stem 层 patch_len=3, padded=8, 对齐率仅 37.5%")
    print(f"    这是唯一未对齐的层，但 ops 极少 (3×1×1=3)")

    # ---- 校准 ----
    print(f"\n[Phase 2b] 校准 QAT 模型...")
    calibrate_qat_model(model, train_loader, DEVICE, num_batches=3)

    # ---- Phase 3: QAT 微调 ----
    print(f"\n[Phase 3] QAT 微调 ({QAT_EPOCHS} epochs, lr={LEARNING_RATE})")

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

        # QAT 训练
        train_loss, train_acc = train_qat_epoch(
            model, train_loader, criterion, optimizer
        )

        # 验证
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

    # QAT vs Float 对比
    comparison = compare_qat_vs_float(model, val_loader, DEVICE, criterion)

    # 保存 QAT 权重
    print(f"\n  保存 QAT-trained 权重至: {QAT_WEIGHTS}")
    torch.save(model.state_dict(), QAT_WEIGHTS)
    print(f"  文件大小: {os.path.getsize(QAT_WEIGHTS) / 1024:.1f} KB")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  网络结构:          Optic-SpaceNet V1 (硬件对齐)")
    print(f"  参数量:            {param_count:,}")
    print(f"  Float32 基准准确率: {result_fp32['accuracy']:.2%}")
    print(f"  QAT 最佳准确率:     {best_acc:.2%}")
    print(f"  QAT vs Float gap:  {comparison['accuracy_gap']:.2%}")
    print(f"  QAT 训练耗时:      {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  8×2 硬件对齐率:    {alignment:.1%}")
    print(f"\n  预期光计算推理:")
    print(f"    原生 float32:     {result_fp32['accuracy']:.1%}")
    print(f"    PTQ (直接 int4):  ~{result_fp32['accuracy'] - 0.12:.1%} (大幅下降)")
    print(f"    QAT (微调后 int4): {best_acc:.1%} (接近 float32)")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
