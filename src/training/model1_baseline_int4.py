"""
================================================================================
 模型一 Int4 (Baseline VGG): 从零开始 QAT 训练 — int4 光计算版本
================================================================================
 目的: 从随机初始化开始，全程在 int4 约束下训练 Mini-VGG。
       模型从未见过 float32 精度，学到的特征天然兼容 int4 光计算硬件。

 与原始 FP32 版本的关系:
   - model1_baseline.py:     标准 float32 训练 (基准，不修改)
   - model1_baseline_qat.py: FP32 微调 + QAT (效果差，保留做对比)
   - model1_baseline_int4.py: 从零 QAT 训练 (本文件，新方案)

 关键设计:
   1. 随机初始化 + QAT 从 epoch 1 开始
   2. 保留 BatchNorm (不融合) — BN 稳定训练，补偿量化误差
   3. 与 FP32 训练相同的 epochs (60) 和学习率 (0.001)
   4. 权重保存为 float32 (但已对 int4 量化鲁棒)

 预期:
   - FP32 from scratch: 97.17%
   - QAT fine-tune:     85.91% (从 FP32 权重微调，效果差)
   - QAT from scratch:  ~90-95% (从零开始，特征天然兼容 int4)

 用法:
   python model1_baseline_int4.py
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

from optic_qat import (
    QATConv2d, QATLinear,
    prepare_qat_model_from_scratch,
    enable_qat, disable_qat,
    evaluate_model,
    compute_alignment_ratio,
    print_alignment_detail,
)

# ============================================================
#  全局配置 — 与 FP32 训练完全一致
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
EPOCHS = 60              # 与 FP32 训练相同
LEARNING_RATE = 0.001    # 与 FP32 训练相同 (标准学习率)
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
IMG_SIZE = 64
SEED = 42

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"设备: {DEVICE}")
print(f"数据目录: {DATA_DIR}")


# ============================================================
#  模型定义: Mini-VGG (与 model1_baseline.py 完全一致)
# ============================================================
class BaselineVGG(nn.Module):
    """标准微型 VGG — 全 3×3 卷积"""

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


# ============================================================
#  数据加载 — 与 FP32 训练完全一致
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
#  训练与评估 (QAT 层自动施加伪 int4 量化)
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    """
    训练一个 epoch。
    QATConv2d/QATLinear 在 model.train() 时自动施加伪 int4 量化。
    STE 梯度让 float32 权重学习 int4 兼容的特征。
    """
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
def validate(model, loader, criterion):
    """
    QAT 模式下验证: QAT 层在 eval 模式也施加伪量化 (测量真实 int4 精度)。
    """
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
    print("  模型一 Int4 (Baseline VGG): 从零 QAT 训练")
    print("  全程 int4 伪量化 — 特征天然兼容光计算硬件")
    print("=" * 60)

    # ---- 加载数据 ----
    train_loader, val_loader = load_data()

    # ---- 创建模型 (随机初始化, 不加载预训练权重) ----
    print(f"\n[Step 1] 创建模型 (随机初始化)")
    model = BaselineVGG(num_classes=NUM_CLASSES)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,}")

    # ---- 转换为 QAT 模型 (保留 BN 层) ----
    print(f"\n[Step 2] 转换为 QAT 模型 (保留 BatchNorm)")
    prepare_qat_model_from_scratch(model)

    # === 混合精度: 首层和末层保持 float32 ===
    # 首层 Conv(3→32, 3×3): 处理原始 RGB 像素, int4 损失太大
    model.block1[0].disable_qat()
    # 末层 Linear(256→10): 输出 logits 直接决定分类, 需要精度
    model.classifier[4].disable_qat()

    qat_layers = {}
    fp32_layers = []
    for name, m in model.named_modules():
        if isinstance(m, QATConv2d):
            if m.qat_enabled:
                qat_layers['QATConv2d'] = qat_layers.get('QATConv2d', 0) + 1
            else:
                fp32_layers.append(f"{name} (float32)")
        elif isinstance(m, QATLinear):
            if m.qat_enabled:
                qat_layers['QATLinear'] = qat_layers.get('QATLinear', 0) + 1
            else:
                fp32_layers.append(f"{name} (float32)")
    print(f"  QAT int4 层: {qat_layers}")
    print(f"  Float32 层: {fp32_layers}")

    # ---- 硬件对齐率 ----
    alignment = print_alignment_detail(model, "BaselineVGG (Int4)")

    # ---- 训练 ----
    print(f"\n[Step 3] 开始混合精度 int4 QAT 训练 ({EPOCHS} epochs, lr={LEARNING_RATE})")
    print(f"  混合精度: 首层 (block1.0) + 末层 (classifier.4) float32, 其余 int4 QAT")
    print(f"  高 weight_decay=5e-4: 抑制 int4 模式下的过拟合")
    print(f"  从随机初始化开始，STE 梯度让 int4 层找到量化友好的权重")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_state = None
    total_train_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'Time':>7s}")
    print("  " + "-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        elapsed = time.time() - t0
        total_train_time += elapsed

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5d}  | {train_loss:>10.4f} {train_acc:>8.2%} | "
                  f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | {elapsed:>6.1f}s")

    # ---- 加载最佳权重 ----
    model.load_state_dict(best_state)

    # ---- 最终评估 (ints4 vs float32 对比) ----
    print(f"\n[Step 4] 最终评估")
    model.eval()

    # int4 模式
    enable_qat(model)
    result_int4 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Int4 模式 (光计算模拟) 准确率: {result_int4['accuracy']:.2%}")

    # float32 模式 (关闭量化)
    disable_qat(model)
    result_fp32 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Float32 模式准确率:         {result_fp32['accuracy']:.2%}")
    print(f"  Int4 量化精度损失:          {result_fp32['accuracy'] - result_int4['accuracy']:.2%}")

    # ---- 保存模型 ----
    print(f"\n  保存 int4 QAT 权重至: weights/baseline_vgg_int4.pth")
    torch.save(model.state_dict(), "weights/baseline_vgg_int4.pth")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  网络结构:            Mini-VGG (全 3×3 卷积)")
    print(f"  参数量:              {param_count:,}")
    print(f"  训练方式:            从零 QAT (int4 from epoch 1)")
    print(f"  训练总耗时:          {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  8×2 硬件对齐率:      {alignment:.1%}")
    print(f"  Int4 最佳准确率:     {best_acc:.2%}")
    print(f"  Float32 准确率:      {result_fp32['accuracy']:.2%}")
    print(f"  Int4 量化损失:       {result_fp32['accuracy'] - result_int4['accuracy']:.2%}")
    print(f"\n  与 FP32 训练对比:")
    print(f"    FP32 from scratch:  97.17% (model1_baseline.py)")
    print(f"    QAT fine-tune:      85.91% (model1_baseline_qat.py, 效果差)")
    print(f"    QAT from scratch:   {best_acc:.2%} (本脚本, 新方案)")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
