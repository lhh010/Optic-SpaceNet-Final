"""
================================================================================
 模型一 (Baseline): 标准微型 VGG — 3×3 卷积网络
================================================================================
 目的: 证明传统 3×3 卷积在 8×2 光计算硬件上利用率低 (~37.5%)，
       展平长度=9 无法被 8 整除，大量补零导致推理极慢。
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import time
import numpy as np

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
IMG_SIZE = 64

print(f"设备: {DEVICE}")
print(f"数据目录: {DATA_DIR}")

# ============================================================
#  模型定义: Mini-VGG (全 3×3 卷积)
# ============================================================
class BaselineVGG(nn.Module):
    """
    标准微型 VGG 网络，全部使用 3×3 卷积。
    - 3×3 展平长度为 9，在 8×2 光计算硬件上必须补零到 16，
      利用率仅 9/16 ≈ 56.25%（单 kernel），实际综合利用率约 37.5%。
    - 这是报告中的"炮灰"对照组。
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: 3→32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3×3, 展平=3×9=27→补到32, 利用率84%
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 3×3, 展平=32×9=288, 288%8=0, 利用率100%*
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # 64→32
        )

        # Block 2: 32→64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 3×3, 展平=32×9=288, OK
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 3×3, 展平=64×9=576, OK
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # 32→16
        )

        # Block 3: 64→128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 3×3, 展平=64×9=576, OK
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# 3×3, 展平=128×9=1152, OK
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # 16→8
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
#  硬件对齐率计算
# ============================================================
def compute_alignment_ratio(model):
    """
    计算模型在 8×2 光计算硬件上的对齐率。
    对于每个 Conv2d 层，im2col 展平长度 = C_in × K_h × K_w。
    如果该长度不能被 8 整除，则需要补零到最近的 8 的倍数。
    对齐率 = 原始长度 / 补零后长度。
    """
    total_ops = 0
    aligned_ops = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_c = module.in_channels
            k_h, k_w = module.kernel_size
            patch_len = in_c * k_h * k_w
            padded_len = ((patch_len + 7) // 8) * 8

            # 该层的 MAC 操作数 = out_c * OH * OW * patch_len
            # 用对齐率加权: patch_len / padded_len
            ratio = patch_len / padded_len
            total_ops += 1
            aligned_ops += ratio

    if total_ops == 0:
        return 0.0
    return aligned_ops / total_ops


def print_alignment_detail(model):
    """打印每层卷积的对齐详情"""
    print("\n  层名                        C_in  K  展平长度  补零后  对齐率")
    print("  " + "-" * 65)
    total_patch = 0
    total_padded = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            patch_len = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            padded_len = ((patch_len + 7) // 8) * 8
            ratio = patch_len / padded_len
            total_patch += patch_len
            total_padded += padded_len
            print(f"  {name:<28s} {module.in_channels:>4d}  {module.kernel_size[0]}×{module.kernel_size[1]}"
                  f"  {patch_len:>6d}     {padded_len:>6d}   {ratio:.1%}")
    overall = total_patch / total_padded if total_padded > 0 else 0
    print(f"  综合硬件对齐率: {overall:.1%} (展平总长度 {total_patch} → 补零后 {total_padded})")
    return overall


# ============================================================
#  数据加载
# ============================================================
def load_data():
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

    # 分别创建训练/验证数据集（各自使用独立的 transform）
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_full = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    n = len(train_full)
    val_size = int(n * VAL_SPLIT)
    indices = list(range(n))
    import numpy as np
    rng = np.random.RandomState(42)
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
#  训练与评估
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
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


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ============================================================
#  主函数
# ============================================================
def main():
    print("=" * 60)
    print("  模型一 (Baseline): 标准 Mini-VGG — 3×3 卷积")
    print("=" * 60)

    # 数据
    train_loader, val_loader = load_data()

    # 模型
    model = BaselineVGG(num_classes=NUM_CLASSES).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {param_count:,}")

    # 硬件对齐率
    alignment = print_alignment_detail(model)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练
    best_acc = 0.0
    total_train_time = 0.0
    print(f"\n开始训练 ({EPOCHS} epochs, batch={BATCH_SIZE}, device={DEVICE})...")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0
        total_train_time += elapsed

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                  f"Time: {elapsed:.1f}s")

    # 结果汇总
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  网络结构:        Mini-VGG (全 3×3 卷积)")
    print(f"  参数量:          {param_count:,}")
    print(f"  训练总耗时:      {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  最佳验证准确率:  {best_acc:.2%}")
    print(f"  8×2 硬件对齐率:  {alignment:.1%}")
    print(f"  光模拟推理预估:  慢 (大量补零，利用率低)")

    # 保存模型
    torch.save(model.state_dict(), "weights/baseline_vgg.pth")
    print(f"\n模型已保存至: weights/baseline_vgg.pth")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
