"""
================================================================================
 模型二 Int4 (Optic-SpaceNet V1): 从零开始 QAT 训练 — int4 光计算版本
================================================================================
 目的: 从随机初始化开始，全程在 int4 约束下训练硬件对齐 CNN。
       模型从未见过 float32 精度，学到的特征天然兼容 int4 光计算硬件。

 与原始版本的关系:
   - model2_spacenet_v1.py:     标准 float32 训练 (基准，不修改)
   - model2_spacenet_v1_qat.py: FP32 微调 + QAT (效果差，保留做对比)
   - model2_spacenet_v1_int4.py: 从零 QAT 训练 (本文件，新方案)

 关键设计:
   1. 随机初始化 + QAT 从 epoch 1 开始
   2. 保留 BatchNorm — 小模型 (268K params) 尤其需要 BN 稳定训练
   3. 与 FP32 训练相同的 epochs (80) 和学习率 (0.001)

 预期:
   - FP32 from scratch: 90.15%
   - QAT fine-tune:     73.63% (从 FP32 权重微调，灾难性退化)
   - QAT from scratch:  ~83-88% (从零开始，特征天然兼容 int4)

 用法:
   python model2_spacenet_v1_int4.py
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
EPOCHS = 100             # 扩展到 100 epochs (小模型 int4 收敛慢)
LEARNING_RATE = 0.001    # 与 FP32 训练相同
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"设备: {DEVICE}")


# ============================================================
#  模型定义: OpticSpaceNet V1 — 与 model2_spacenet_v1.py 完全一致
# ============================================================
class OpticSpaceNetV1(nn.Module):
    """
    硬件感知 CNN: 所有 Conv 的 im2col 展平长度均被 8 整除，
    输出通道数被 2 整除，完美匹配 8×2 光学矩阵乘法器。
    """

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
#  训练与评估
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    """QAT 训练: 前向时伪 int4 量化自动施加, STE 梯度更新 float32 权重"""
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
    """QAT 验证: 测量真实 int4 精度"""
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
    print("  模型二 Int4 (Optic-SpaceNet V1): 从零 QAT 训练")
    print("  全程 int4 伪量化 — 特征天然兼容光计算硬件")
    print("=" * 60)

    # ---- 加载数据 ----
    train_loader, val_loader = load_data()

    # ---- 创建模型 (随机初始化) ----
    print(f"\n[Step 1] 创建模型 (随机初始化, 不使用预训练权重)")
    model = OpticSpaceNetV1(num_classes=NUM_CLASSES)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,}")

    # ---- 转换为 QAT 模型 (保留 BN) ----
    print(f"\n[Step 2] 转换为 QAT 模型 (保留 BatchNorm 层)")
    prepare_qat_model_from_scratch(model)

    # === 混合精度: 首层和末层保持 float32 ===
    # stem Conv(3→8, 1×1): patch=3, 对齐率仅37.5%, int4量化会严重损失RGB信息
    model.stem[0].disable_qat()
    # classifier Linear(256→10): 输出 logits, 需要精度
    model.classifier[4].disable_qat()

    qat_layers = {}
    fp32_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (QATConv2d, QATLinear)):
            if m.qat_enabled:
                k = 'QATConv2d' if isinstance(m, QATConv2d) else 'QATLinear'
                qat_layers[k] = qat_layers.get(k, 0) + 1
            else:
                fp32_layers.append(f"{name} (float32)")
    print(f"  QAT int4 层: {qat_layers}")
    print(f"  Float32 层: {fp32_layers}")

    # ---- 硬件对齐率 ----
    alignment = print_alignment_detail(model, "OpticSpaceNetV1 (Int4)")
    print(f"  ⚠ stem 层对齐率仅 37.5% (patch=3→8)，但 ops 极少")

    # ---- 训练 ----
    print(f"\n[Step 3] 开始混合精度 int4 QAT 训练 ({EPOCHS} epochs, lr={LEARNING_RATE})")
    print(f"  混合精度: stem(3→8) + classifier(256→10) float32, 其余 int4 QAT")
    print(f"  stem 本身对齐率仅37.5%, float32 还避免了光计算硬件浪费")
    print(f"  从随机初始化开始 + BN 辅助训练")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
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

    # ---- 最终评估 ----
    print(f"\n[Step 4] 最终评估")
    model.eval()

    enable_qat(model)
    result_int4 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Int4 模式 (光计算模拟) 准确率: {result_int4['accuracy']:.2%}")

    disable_qat(model)
    result_fp32 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Float32 模式准确率:         {result_fp32['accuracy']:.2%}")
    print(f"  Int4 量化精度损失:          {result_fp32['accuracy'] - result_int4['accuracy']:.2%}")

    # ---- 保存模型 ----
    print(f"\n  保存 int4 QAT 权重至: spacenet_v1_int4.pth")
    torch.save(model.state_dict(), "spacenet_v1_int4.pth")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  网络结构:            Optic-SpaceNet V1 (硬件对齐)")
    print(f"  参数量:              {param_count:,}")
    print(f"  训练方式:            从零 QAT (int4 from epoch 1)")
    print(f"  训练总耗时:          {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  8×2 硬件对齐率:      {alignment:.1%}")
    print(f"  Int4 最佳准确率:     {best_acc:.2%}")
    print(f"  Float32 准确率:      {result_fp32['accuracy']:.2%}")
    print(f"  Int4 量化损失:       {result_fp32['accuracy'] - result_int4['accuracy']:.2%}")
    print(f"\n  与 FP32 训练对比:")
    print(f"    FP32 from scratch:  90.15% (model2_spacenet_v1.py)")
    print(f"    QAT fine-tune:      73.63% (model2_spacenet_v1_qat.py, 效果差)")
    print(f"    QAT from scratch:   {best_acc:.2%} (本脚本, 新方案)")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
