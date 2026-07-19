"""
================================================================================
  模型四 (MiniVGG-GAP): 标准 CNN 设计 — 与 model2 参数量持平
================================================================================
  设计原则:
    - 3×3 conv (有空间感受野)
    - 每 stage 2 层 Conv (充分非线性)
    - 通道渐进增长: 32→48→72→108
    - GAP 替代巨量 FC (FC 只占 <1% 参数)
    - BN + bias=False

  对比 model2:
    - 参数量: ~290K vs 268K (持平)
    - 计算量: ~??? MACs vs 1.05M MOPs (均可在 CPU 训练)
    - 目的: 验证 2×2 kernel + FC 瓶颈是否严重限制 model2 表达力

  用法:
    python src/training/model4_minivgg_gap.py
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
import time
import numpy as np

from train_phase4_runner import load_eurosat_data

# ============================================================
#  全局配置 (对齐 model2 FP32 基线: model2_spacenet_v1.py)
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"设备: {DEVICE}")
print(f"数据目录: {DATA_DIR}")


# ============================================================
#  模型定义: MiniVGG + GAP
# ============================================================
class MiniVGG(nn.Module):
    """
    结构 (早下采样, 避免 64×64 上做重卷积):
      stem:   Conv 3→32, 3×3, stride=2                   [32×32]
              MaxPool2d(2)                                 [16×16]
      stage1: [Conv 32→48, 3×3 → Conv 48→48, 3×3]        [16×16]
              MaxPool2d(2)                                 [8×8]
      stage2: [Conv 48→72, 3×3 → Conv 72→72, 3×3]        [8×8]
              MaxPool2d(2)                                 [4×4]
      stage3: [Conv 72→96, 3×3 → Conv 96→96, 3×3]        [4×4]
      head:   GAP → Linear(96, 10)
    参数量: ~260K, 计算量: ~17M MACs/img (model2: ~1.4M)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, num_classes),
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


# ============================================================
#  训练
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def main():
    device = DEVICE

    # 数据
    train_loader, val_loader = load_eurosat_data(
        DATA_DIR, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED)

    # 模型
    model = MiniVGG(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {n_params:,}")
    print(f"模型结构:\n{model}\n")

    # 优化器 + 调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_state = None
    history = []

    print(f"{'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>7s} | {'Time':>7s}")
    print("-" * 73)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append((epoch, train_acc, val_acc))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5d}  | {train_loss:>10.4f} {train_acc:>8.2%} | "
                  f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>6.2%} | "
                  f"{elapsed:>6.1f}s")

    # 最终评估
    if best_state is not None:
        model.load_state_dict(best_state)
    _, final_val_acc = evaluate(model, val_loader, criterion, device)

    # 保存
    save_path = "weights/minivgg_gap.pth"
    torch.save(best_state, save_path)

    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"{'='*60}")
    print(f"  模型:        MiniVGG-GAP")
    print(f"  参数量:      {n_params:,}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  最佳 val:    {best_acc:.2%}")
    print(f"  最终 val:    {final_val_acc:.2%}")
    print(f"  权重:        {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
