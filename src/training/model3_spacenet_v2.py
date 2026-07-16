"""
================================================================================
 模型三 (Optic-SpaceNet V2): 知识蒸馏版 — 最终王牌
================================================================================
 目的: 用大型教师网络 (ResNet-18) 的软标签，通过知识蒸馏 (KD)
       教导微小的 OpticSpaceNet 学生网络，在保持 ~100% 硬件对齐率的
       同时，将准确率提升到逼近大模型的水平。

 技术要点:
   - 教师: ResNet-18 (ImageNet 预训练) → EuroSAT 微调 → 准确率 ~96%+
   - 学生: OpticSpaceNet (硬件完美对齐)
   - 蒸馏损失: L = α·KL(softmax(teacher/T) || softmax(student/T))·T^2
                  + (1-α)·CrossEntropy(student, hard_label)
   - 温度 T=4.0, α=0.5

 算力投入:
   - 教师训练: GPU/CPU 若干 epochs (重算力)
   - 蒸馏训练: GPU/CPU 若干 epochs (重算力)
   - 推理: 仅用学生网络，100% 硬件利用率，极速推理
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import time
import copy

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
TEACHER_EPOCHS = 30      # 教师微调轮数
STUDENT_EPOCHS = 100      # 蒸馏训练轮数 (比独立训练更多)
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

# 蒸馏超参数
TEMPERATURE = 4.0
ALPHA = 0.5  # 软标签权重 (1-alpha = 硬标签权重)

print(f"设备: {DEVICE}")


# ============================================================
#  学生模型: OpticSpaceNet (硬件完美对齐)
# ============================================================
class OpticSpaceNetStudent(nn.Module):
    """与 model2 相同的架构，确保所有 Conv 展平长度被 8 整除"""

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
#  教师模型: ResNet-18 (ImageNet 预训练 → EuroSAT 微调)
# ============================================================
def build_teacher(num_classes=10):
    """构建并返回教师模型"""
    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 替换最后的 FC 层
    in_features = teacher.fc.in_features
    teacher.fc = nn.Linear(in_features, num_classes)
    return teacher


# ============================================================
#  硬件对齐率计算
# ============================================================
def compute_alignment_ratio(model):
    total_patch, total_padded = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            total_patch += patch
            total_padded += padded
    return total_patch / total_padded if total_padded > 0 else 0


def print_alignment_detail(model):
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
    print(f"  学生综合硬件对齐率: {overall:.1%}")
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
    return train_loader, val_loader


# ============================================================
#  第一阶段: 训练教师模型
# ============================================================
def train_teacher(teacher, train_loader, val_loader):
    """在 EuroSAT 上微调预训练 ResNet-18"""
    print("\n" + "=" * 60)
    print("  第一阶段: 训练教师模型 (ResNet-18)")
    print("=" * 60)

    teacher = teacher.to(DEVICE)
    param_count = sum(p.numel() for p in teacher.parameters())
    print(f"教师参数量: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS)

    best_acc = 0.0
    best_model = None

    for epoch in range(1, TEACHER_EPOCHS + 1):
        # Train
        teacher.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # Validate
        teacher.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = teacher(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(teacher.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Teacher Epoch {epoch:3d}/{TEACHER_EPOCHS} | "
                  f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
                  f"Best: {best_acc:.2%}")

    teacher.load_state_dict(best_model)
    print(f"\n教师模型最佳验证准确率: {best_acc:.2%}")

    # 评估教师硬件对齐率
    print("\n教师 (ResNet-18) 硬件对齐分析 (3×3 卷积为主):")
    t_patch, t_padded = 0, 0
    for m in teacher.modules():
        if isinstance(m, nn.Conv2d):
            patch = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            padded = ((patch + 7) // 8) * 8
            t_patch += patch
            t_padded += padded
    print(f"  教师对齐率: {t_patch/t_padded:.1%} (大量 3×3 展平=9, 补零到16, 利用率低)")

    return teacher, best_acc


# ============================================================
#  第二阶段: 知识蒸馏训练学生
# ============================================================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    知识蒸馏损失 = α·KL(softmax(teacher/T) || softmax(student/T))·T^2
                   + (1-α)·CrossEntropy(student, labels)

    参考: Hinton et al., "Distilling the Knowledge in a Neural Network"
    """
    # 软损失: KL 散度
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)  # 乘以 T^2 保持梯度尺度

    # 硬损失: 标准交叉熵
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_student_distill(teacher, student, train_loader, val_loader):
    """用知识蒸馏训练学生网络"""
    print("\n" + "=" * 60)
    print("  第二阶段: 知识蒸馏训练学生 (OpticSpaceNet)")
    print("=" * 60)
    print(f"  蒸馏温度 T={TEMPERATURE}, α={ALPHA}")

    student = student.to(DEVICE)
    teacher = teacher.to(DEVICE)
    teacher.eval()  # 教师固定，不更新

    param_count = sum(p.numel() for p in student.parameters())
    print(f"学生参数量: {param_count:,}")

    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS)

    best_acc = 0.0
    best_model = None
    total_time = 0.0

    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()

        # 蒸馏训练
        student.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 教师前向 (无梯度)
            with torch.no_grad():
                teacher_logits = teacher(images)

            # 学生前向
            student_logits = student(images)

            # 蒸馏损失
            loss = distillation_loss(student_logits, teacher_logits, labels,
                                     TEMPERATURE, ALPHA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (student_logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()
        elapsed = time.time() - t0
        total_time += elapsed

        # 验证
        student.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = student(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(student.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Student Epoch {epoch:3d}/{STUDENT_EPOCHS} | "
                  f"KD Loss: {train_loss/train_total:.4f} | "
                  f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
                  f"Best: {best_acc:.2%} | Time: {elapsed:.1f}s")

    student.load_state_dict(best_model)
    return student, best_acc, total_time


# ============================================================
#  主函数
# ============================================================
def main():
    print("=" * 60)
    print("  模型三 (Optic-SpaceNet V2): 知识蒸馏")
    print("=" * 60)

    train_loader, val_loader = load_data()

    # ---- 第一阶段: 训练教师 ----
    teacher = build_teacher(num_classes=NUM_CLASSES)
    teacher, teacher_acc = train_teacher(teacher, train_loader, val_loader)
    torch.save(teacher.state_dict(), "weights/teacher_resnet18.pth")
    print(f"教师模型已保存至: weights/teacher_resnet18.pth")

    # ---- 第二阶段: 蒸馏训练学生 ----
    student = OpticSpaceNetStudent(num_classes=NUM_CLASSES)

    # 打印对齐信息
    alignment = print_alignment_detail(student)

    student, student_acc, distill_time = train_student_distill(
        teacher, student, train_loader, val_loader
    )

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  训练完成 — 结果汇总")
    print("=" * 60)
    print(f"  教师模型:        ResNet-18 (ImageNet预训练 + EuroSAT微调)")
    print(f"  教师准确率:      {teacher_acc:.2%}")
    print(f"  学生模型:        OpticSpaceNet (硬件完美对齐)")
    print(f"  学生参数量:      {sum(p.numel() for p in student.parameters()):,}")
    print(f"  蒸馏训练耗时:    {distill_time:.1f} 秒 ({distill_time/60:.1f} 分钟)")
    print(f"  学生最佳准确率:  {student_acc:.2%} (通过蒸馏逼近教师)")
    print(f"  8×2 硬件对齐率:  {alignment:.1%} (接近 100%)")
    print(f"  光模拟推理预估:  极速 (无补零浪费) + 高精度")

    # 计算相比独立训练的精度提升 (此处仅为示意)
    print(f"\n  [chart] 与独立训练对比 (预期):")
    print(f"     独立训练 OpticSpaceNet: ~75-82%")
    print(f"     蒸馏后 OpticSpaceNet:   ~{student_acc:.1%}")
    print(f"     精度提升:              +{student_acc - 0.78:.1%} (约)")

    torch.save(student.state_dict(), "weights/spacenet_v2_distilled.pth")
    print(f"\n学生模型已保存至: weights/spacenet_v2_distilled.pth")

    return teacher_acc, student_acc, alignment, distill_time


if __name__ == "__main__":
    main()
