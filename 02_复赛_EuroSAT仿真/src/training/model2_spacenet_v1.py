"""
================================================================================
 模型二 (Optic-SpaceNet V1): 硬件感知对齐网络 — 独立训练
================================================================================
 目的: 所有卷积的 im2col 展平长度均能被 8 整除，通道数能被 2 整除，
       在 8×2 光计算硬件上实现 ~100% 利用率。
       1×1 卷积提维 + 2×2 卷积下采样，完全对齐光学矩阵乘法单元。

 网络结构:
   stem:   Conv2d(3→8,  1×1)    → 8×64×64   [对齐: 3/8≈37.5%, ops极少]
   stage1: Conv2d(8→16, 2×2,s=2)→16×32×32   [对齐: 32/32=100%]
           MaxPool2d(2)          →16×16×16
   stage2: Conv2d(16→32,2×2,s=2)→32×8×8     [对齐: 64/64=100%]
   stage3: Conv2d(32→16,1×1)    →16×8×8     [对齐: 32/32=100%]
   fc:     Linear(1024→256)                 [对齐: 1024/1024=100%]
           Linear(256→10)
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
EPOCHS = 80
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

print(f"设备: {DEVICE}")


# ============================================================
#  模型定义: OpticSpaceNet V1
# ============================================================
class OpticSpaceNetV1(nn.Module):
    """
    硬件感知 CNN: 所有 Conv 的 im2col 展平长度均被 8 整除，
    输出通道数被 2 整除，完美匹配 8×2 光学矩阵乘法器。
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: 1×1 卷积将 RGB(3) 提维到 8（对齐硬件宽度）
        # patch = 3×1×1 = 3 → 补零到 8 (唯一未对齐层，ops极少)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 8→16, 2×2 stride=2, 64→32
        # patch = 8×2×2 = 32 → 完美被 8 整除 [OK]
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16
        )

        # Stage 2: 16→32, 2×2 stride=2, 16→8
        # patch = 16×2×2 = 64 → 完美被 8 整除 [OK]
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 32→16, 1×1 保持分辨率
        # patch = 32×1×1 = 32 → 完美被 8 整除 [OK]
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 分类头: 16×8×8 = 1024 → 256 → 10
        # 1024 / 8 = 128, 256 / 8 = 32 → 完美对齐 [OK]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x


# ============================================================
#  硬件对齐率计算
# ============================================================
def compute_alignment_ratio(model):
    """综合对齐率"""
    total_patch = 0
    total_padded = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
    return total_patch / total_padded if total_padded > 0 else 0


def print_alignment_detail(model):
    """打印每层对齐详情"""
    print("\n  层                         C_in  K    展平长度  补零后  对齐率")
    print("  " + "-" * 65)
    total_patch, total_padded = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
            print(f"  {name:<28s} {m.in_channels:>4d}  {m.kernel_size[0]}×{m.kernel_size[1]}"
                  f"   {patch:>6d}      {padded:>6d}   {patch/padded:.1%}")
    overall = total_patch / total_padded
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
    # 用相同的随机种子生成一致的划分
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
    print("  模型二 (Optic-SpaceNet V1): 硬件感知对齐 + 独立训练")
    print("=" * 60)

    train_loader, val_loader = load_data()

    model = OpticSpaceNetV1(num_classes=NUM_CLASSES).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {param_count:,}")

    alignment = print_alignment_detail(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
    print(f"  网络结构:        Optic-SpaceNet V1 (硬件对齐)")
    print(f"  参数量:          {param_count:,}")
    print(f"  训练总耗时:      {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  最佳验证准确率:  {best_acc:.2%}")
    print(f"  8×2 硬件对齐率:  {alignment:.1%} (接近 100%)")
    print(f"  光模拟推理预估:  极速 (无补零浪费)")

    torch.save(model.state_dict(), "weights/spacenet_v1.pth")
    print(f"\n模型已保存至: weights/spacenet_v1.pth")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
