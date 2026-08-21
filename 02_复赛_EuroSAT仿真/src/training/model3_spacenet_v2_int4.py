"""
================================================================================
 模型三 Int4 (Optic-SpaceNet V2): 从零开始 KD+QAT 联合训练 — int4 光计算版本
================================================================================
 目的: 从随机初始化开始，用知识蒸馏 (KD) + QAT 联合训练硬件对齐 CNN。
       教师模型 (ResNet-18) 提供软标签引导，学生从零开始学习 int4 兼容特征。

 与原始版本的关系:
   - model3_spacenet_v2.py:     标准 KD 训练 (基准，不修改)
   - model3_spacenet_v2_qat.py: FP32 微调 + QAT (效果差，保留做对比)
   - model3_spacenet_v2_int4.py: 从零 KD+QAT 训练 (本文件，新方案)

 训练策略:
   - 教师: ResNet-18 (已预训练, 固定不更新)
   - 学生: OpticSpaceNet (随机初始化, 268K params)
   - 损失: α·KL(teacher/T, student/T)·T^2 + (1-α)·CE(student, label)
   - QAT: 从 epoch 1 开始施加伪 int4 量化
   - BN: 保留不融合，稳定训练

 关键设计:
   1. 学生从随机初始化开始 (不加载 FP32 权重)
   2. KD + QAT 同时从 epoch 1 生效
   3. 与 FP32 KD 训练相同的 epochs (100)
   4. 教师提供 soft label 引导, 学生学 int4 兼容特征

 预期:
   - FP32 KD:           91.44%
   - QAT fine-tune:     73.22% (从 KD 权重微调，灾难性退化)
   - QAT from scratch:  ~85-90% (KD 引导 + int4 天然兼容)

 用法:
   python model3_spacenet_v2_int4.py
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
import os
import time
import copy
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
#  全局配置 — 与 FP32 KD 训练完全一致
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
STUDENT_EPOCHS = 120     # 扩展到 120 epochs (KD+QAT 收敛慢)
LEARNING_RATE = 0.001    # 与 FP32 KD 训练相同
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

# 蒸馏超参数
TEMPERATURE = 4.0
ALPHA = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"设备: {DEVICE}")


# ============================================================
#  学生模型: OpticSpaceNet — 与 model3_spacenet_v2.py 完全一致
# ============================================================
class OpticSpaceNetStudent(nn.Module):
    """硬件感知 CNN, 268K 参数"""

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
#  教师模型: ResNet-18
# ============================================================
def build_teacher(num_classes=10):
    """构建并加载预训练教师模型"""
    teacher = models.resnet18(weights=None)
    in_features = teacher.fc.in_features
    teacher.fc = nn.Linear(in_features, num_classes)

    if os.path.exists("weights/teacher_resnet18.pth"):
        state = torch.load("weights/teacher_resnet18.pth", map_location='cpu')
        teacher.load_state_dict(state)
        print(f"  教师权重加载自: weights/teacher_resnet18.pth")
    else:
        print(f"  [警告] 教师权重文件不存在，教师将使用随机权重!")
        print(f"  请先运行 model3_spacenet_v2.py 生成教师权重。")

    teacher.to(DEVICE)
    teacher.eval()
    return teacher


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
#  蒸馏损失 (Hinton et al., 2015)
# ============================================================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    KD Loss = α·KL(σ(teacher/T) || σ(student/T))·T^2 + (1-α)·CE(student, labels)
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


# ============================================================
#  KD + QAT 联合训练
# ============================================================
def train_epoch(model, teacher, loader, optimizer):
    """
    一个 epoch 的 KD+QAT 联合训练。
    教师输出软标签，学生通过 QAT 层进行伪 int4 量化，
    两股力量同时塑造学生: 教师引导 + int4 约束。
    """
    model.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 教师前向 (无梯度, 固定不变)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # 学生前向 (QAT 层自动施加伪 int4 量化)
        student_logits = model(images)

        # KD 损失
        loss = distillation_loss(student_logits, teacher_logits, labels,
                                 TEMPERATURE, ALPHA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    """验证: QAT 层在 eval 模式也施加伪量化"""
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
    print("  模型三 Int4 (Optic-SpaceNet V2): 从零 KD+QAT 联合训练")
    print("  教师引导 + int4 约束 — 学生学 int4 兼容特征")
    print("=" * 60)

    # ---- 加载数据 ----
    train_loader, val_loader = load_data()

    # ---- 加载教师 ----
    print(f"\n[Step 1] 加载教师模型 (ResNet-18)")
    teacher = build_teacher(num_classes=NUM_CLASSES)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  教师参数量: {teacher_params:,}")

    # ---- 创建学生 (随机初始化) ----
    print(f"\n[Step 2] 创建学生模型 (随机初始化, 不使用预训练权重)")
    model = OpticSpaceNetStudent(num_classes=NUM_CLASSES)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  学生参数量: {param_count:,}")

    # ---- 转换为 QAT 模型 (保留 BN) ----
    print(f"\n[Step 3] 转换为 QAT 模型 (保留 BatchNorm 层)")
    prepare_qat_model_from_scratch(model)

    # === 混合精度: 首层和末层保持 float32 ===
    model.stem[0].disable_qat()
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
    alignment = print_alignment_detail(model, "OpticSpaceNetStudent (Int4)")

    # ---- KD + QAT 联合训练 ----
    print(f"\n[Step 4] 开始混合精度 KD+QAT 联合训练")
    print(f"  学生 epochs: {STUDENT_EPOCHS}, lr={LEARNING_RATE}")
    print(f"  蒸馏温度 T={TEMPERATURE}, α={ALPHA}")
    print(f"  混合精度: stem(3→8) + classifier(256→10) float32, 其余 int4 QAT")
    print(f"  教师引导方向 + int4 约束 + float32 首末层保护敏感特征")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS)

    best_acc = 0.0
    best_state = None
    total_train_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>5s} | {'KD Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'Time':>7s}")
    print("  " + "-" * 65)

    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, teacher, train_loader, optimizer
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
    print(f"\n[Step 5] 最终评估")
    model.eval()

    enable_qat(model)
    result_int4 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Int4 模式 (光计算模拟) 准确率: {result_int4['accuracy']:.2%}")

    disable_qat(model)
    result_fp32 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Float32 模式准确率:         {result_fp32['accuracy']:.2%}")
    print(f"  Int4 量化精度损失:          {result_fp32['accuracy'] - result_int4['accuracy']:.2%}")

    # ---- 保存模型 ----
    print(f"\n  保存 int4 QAT 权重至: weights/spacenet_v2_int4.pth")
    torch.save(model.state_dict(), "weights/spacenet_v2_int4.pth")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  教师模型:            ResNet-18 ({teacher_params:,} params)")
    print(f"  学生模型:            OpticSpaceNet ({param_count:,} params)")
    print(f"  训练方式:            从零 KD+QAT 联合 (int4 from epoch 1)")
    print(f"  训练总耗时:          {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  8×2 硬件对齐率:      {alignment:.1%}")
    print(f"  Int4 最佳准确率:     {best_acc:.2%}")
    print(f"  Float32 准确率:      {result_fp32['accuracy']:.2%}")
    print(f"  Int4 量化损失:       {result_fp32['accuracy'] - result_int4['accuracy']:.2%}")
    print(f"\n  与 FP32 KD 训练对比:")
    print(f"    FP32 KD from scratch: 91.44% (model3_spacenet_v2.py)")
    print(f"    QAT fine-tune:        73.22% (model3_spacenet_v2_qat.py, 效果差)")
    print(f"    QAT from scratch:     {best_acc:.2%} (本脚本, 新方案)")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
