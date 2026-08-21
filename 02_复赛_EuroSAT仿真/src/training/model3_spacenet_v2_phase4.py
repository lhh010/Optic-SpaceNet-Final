"""
================================================================================
 模型三 Phase 4 (SpaceNet V2 KD): STE + 噪声注入 + 非对称量化 + KD
================================================================================
 基于初赛验证方法 — 在 Model 2 基础上加入知识蒸馏引导。

 用法:
   python model3_spacenet_v2_phase4.py
   python model3_spacenet_v2_phase4.py --mode lsqplus
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os, sys, time
import numpy as np

from optic_qat_v2 import (
    QATConv2d_v2, QATLinear_v2,
    prepare_model_phase4, set_quant_lr,
    evaluate_model_v2,
)

DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
STUDENT_EPOCHS = 100
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42
TEMPERATURE = 4.0
ALPHA = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)
MODE = "lsqplus" if "--mode" in " ".join(sys.argv) and "lsqplus" in sys.argv else "ste"
NOISE = MODE == "ste"

print(f"设备: {DEVICE}, 模式: {MODE}, 噪声注入: {NOISE}")


# ============================================================
class OpticSpaceNetStudent(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, 2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, 2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, 1, bias=False),
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
def build_teacher():
    teacher = models.resnet18(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, NUM_CLASSES)
    if os.path.exists("weights/teacher_resnet18.pth"):
        teacher.load_state_dict(torch.load("weights/teacher_resnet18.pth", map_location='cpu'))
        print("  教师权重加载成功")
    teacher.to(DEVICE).eval()
    return teacher


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
    train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
    val_dataset = torch.utils.data.Subset(val_full, indices[:val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
    return train_loader, val_loader


# ============================================================
def kd_loss(s_logits, t_logits, labels, T, alpha):
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                     F.softmax(t_logits / T, dim=1),
                     reduction='batchmean') * (T * T)
    hard = F.cross_entropy(s_logits, labels)
    return alpha * soft + (1 - alpha) * hard


def train_epoch(model, teacher, loader, optimizer):
    model.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            t_logits = teacher(images)
        s_logits = model(images)
        loss = kd_loss(s_logits, t_logits, labels, TEMPERATURE, ALPHA)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (s_logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ============================================================
def main():
    print("=" * 60)
    print(f"  Model 3 Phase 4: KD + {MODE.upper()} + uint4/int4 非对称量化")
    print("=" * 60)

    train_loader, val_loader = load_data()

    print(f"\n[Step 1] 加载教师 (ResNet-18)")
    teacher = build_teacher()

    print(f"\n[Step 2] 创建学生 (bias=False)")
    model = OpticSpaceNetStudent(num_classes=NUM_CLASSES)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n[Step 3] 转换为 Phase 4 QAT")
    prepare_model_phase4(model, mode=MODE, noise=NOISE)

    qc = sum(1 for m in model.modules() if isinstance(m, QATConv2d_v2))
    ql = sum(1 for m in model.modules() if isinstance(m, QATLinear_v2))
    print(f"  QATConv2d_v2: {qc}, QATLinear_v2: {ql}")

    print(f"\n[Step 4] KD + QAT 联合训练 ({STUDENT_EPOCHS} epochs)")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if MODE == "lsqplus":
        param_groups = set_quant_lr(model, base_lr=LEARNING_RATE)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS)

    best_acc, best_state = 0.0, None
    total_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>5s} | {'KD Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'Time':>7s}")
    print("  " + "-" * 65)

    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, teacher, train_loader, optimizer)
        result = evaluate_model_v2(model, val_loader, DEVICE, criterion)
        val_loss, val_acc = result['loss'], result['accuracy']
        elapsed = time.time() - t0
        total_time += elapsed
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5d}  | {train_loss:>10.4f} {train_acc:>8.2%} | "
                  f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | {elapsed:>6.1f}s")

    model.load_state_dict(best_state)
    fname = f"weights/spacenet_v2_phase4_{MODE}.pth"
    torch.save(model.state_dict(), fname)
    print(f"\n  模型已保存: {fname}")

    print(f"\n{'='*60}")
    print(f"  结果: Int4 = {best_acc:.2%}")
    print(f"  FP32 KD 基准: 91.44%, Phase 2: 83.26%")
    print(f"  模式: {MODE}, 耗时: {total_time/60:.1f}min")
    print(f"{'='*60}")

    return best_acc


if __name__ == "__main__":
    main()
