"""
================================================================================
 模型三 QAT (Optic-SpaceNet V2): 知识蒸馏版 → QAT 微调
================================================================================
 目的: 对预训练的 float32 OpticSpaceNet V2 (知识蒸馏版) 进行 QAT 微调，
       使其适应 int4 光计算推理，同时保持蒸馏带来的精度优势。

 背景:
   - 教师模型: ResNet-18 (ImageNet 预训练 + EuroSAT 微调) → 97.83%
   - 学生模型: OpticSpaceNet (268K 参数, 硬件完美对齐)
   - Float32 KD 最佳准确率: 91.44% (100 epochs 蒸馏训练)
   - 首次迁移 (PTQ): 直接 int4 量化推理，精度大幅下降 (~10-15%)
   - QAT 微调后预期:   ~89-91% (最大程度保持蒸馏精度)

 训练策略:
   本脚本提供两种 QAT 模式:

   Mode A — 标准 QAT 微调 (默认):
     Phase 1 — 加载预训练 KD float32 权重
     Phase 2 — Conv+BN 融合 + QAT 层替换
     Phase 3 — QAT 微调 (低学习率, 20 epochs, CrossEntropy loss)
     Phase 4 — 保存 QAT-trained 权重

   Mode B — KD + QAT 联合微调 (可选, 需要教师模型):
     Phase 1 — 加载预训练 KD float32 权重 + 教师模型
     Phase 2 — Conv+BN 融合 + QAT 层替换
     Phase 3 — QAT 微调 (使用蒸馏损失: α·KL + (1-α)·CE)
     Phase 4 — 保存 QAT-trained 权重
     [推荐] KD + QAT 联合微调可以更好地保持蒸馏精度

 与原有文件的关系:
   - model3_spacenet_v2.py:       标准 KD 训练 (产出 spacenet_v2_distilled.pth)
   - model3_spacenet_v2_qat.py:   QAT 微调 (产出 spacenet_v2_qat.pth)
   - optic_inference.py:          光计算推理评估

 用法:
   # Mode A: 标准 QAT 微调
   python model3_spacenet_v2_qat.py

   # Mode B: KD + QAT 联合微调
   python model3_spacenet_v2_qat.py --use_kd
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import copy
import sys
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
QAT_EPOCHS = 20          # QAT 微调轮数
LEARNING_RATE = 1e-4     # QAT 低学习率
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

# 文件路径
FP32_WEIGHTS = "spacenet_v2_distilled.pth"   # 预训练 KD float32 权重
QAT_WEIGHTS = "spacenet_v2_qat.pth"          # QAT 微调后输出权重
TEACHER_WEIGHTS = "teacher_resnet18.pth"     # 教师权重 (Mode B 需要)

# Mode B 蒸馏超参数
TEMPERATURE = 4.0
ALPHA = 0.5  # 软标签权重

print(f"设备: {DEVICE}")


# ============================================================
#  学生模型: OpticSpaceNet — 与 model3 一致
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
    """构建教师模型 (与 model3 一致)"""
    teacher = models.resnet18(weights=None)  # 不下载预训练, 从本地加载
    in_features = teacher.fc.in_features
    teacher.fc = nn.Linear(in_features, num_classes)
    return teacher


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
#  蒸馏损失 (Hinton et al., 2015)
# ============================================================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    知识蒸馏损失 = α·KL(softmax(teacher/T) || softmax(student/T))·T²
                   + (1-α)·CrossEntropy(student, labels)
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


# ============================================================
#  QAT 训练 (标准 CrossEntropy)
# ============================================================
def train_qat_epoch(model, loader, criterion, optimizer):
    """标准 QAT 训练一个 epoch"""
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
#  QAT + KD 联合训练
# ============================================================
def train_qat_kd_epoch(model, teacher, loader, optimizer):
    """KD+QAT 联合训练一个 epoch"""
    model.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 教师前向 (无梯度)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # 学生前向 (QAT 伪量化自动施加)
        student_logits = model(images)

        # 蒸馏损失
        loss = distillation_loss(student_logits, teacher_logits, labels,
                                 TEMPERATURE, ALPHA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ============================================================
#  主函数
# ============================================================
def main():
    use_kd = "--use_kd" in sys.argv

    print("=" * 60)
    mode_name = "Mode B: KD + QAT 联合微调" if use_kd else "Mode A: 标准 QAT 微调"
    print(f"  模型三 QAT (Optic-SpaceNet V2): {mode_name}")
    print("=" * 60)

    # ---- 加载数据 ----
    train_loader, val_loader = load_data()

    # ---- Phase 1: 加载预训练模型 ----
    print(f"\n[Phase 1] 加载预训练权重")

    # 学生模型
    print(f"  学生权重: {FP32_WEIGHTS}")
    model = OpticSpaceNetStudent(num_classes=NUM_CLASSES)

    if not os.path.exists(FP32_WEIGHTS):
        print(f"  [错误] 学生权重文件不存在: {FP32_WEIGHTS}")
        print(f"  请先运行 model3_spacenet_v2.py 进行蒸馏训练。")
        return

    state_dict = torch.load(FP32_WEIGHTS, map_location='cpu')
    model.load_state_dict(state_dict)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  学生参数量: {param_count:,}")
    print(f"  学生权重加载成功!")

    # 教师模型 (Mode B 需要)
    teacher = None
    if use_kd:
        print(f"  教师权重: {TEACHER_WEIGHTS}")
        if not os.path.exists(TEACHER_WEIGHTS):
            print(f"  [警告] 教师权重文件不存在: {TEACHER_WEIGHTS}")
            print(f"  将回退到 Mode A (标准 QAT)")
            use_kd = False
        else:
            teacher = build_teacher(num_classes=NUM_CLASSES)
            teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location='cpu'))
            teacher.to(DEVICE)
            teacher.eval()
            print(f"  教师参数量: {sum(p.numel() for p in teacher.parameters()):,}")
            print(f"  教师权重加载成功!")

    # ---- 评估 float32 baseline ----
    print(f"\n[Phase 1b] 评估 float32 基准准确率...")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    result_fp32 = evaluate_model(model, val_loader, DEVICE, criterion)
    print(f"  Float32 验证准确率: {result_fp32['accuracy']:.2%}")
    print(f"  Float32 验证损失:   {result_fp32['loss']:.4f}")

    # ---- Phase 2: 准备 QAT 模型 ----
    print(f"\n[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...")

    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            _ = model(images.to(DEVICE))
            break

    prepare_qat_model(model, fuse_bn=True, inplace=True)
    qat_counts = {}
    for m in model.modules():
        if isinstance(m, QATConv2d):
            qat_counts['QATConv2d'] = qat_counts.get('QATConv2d', 0) + 1
        elif isinstance(m, QATLinear):
            qat_counts['QATLinear'] = qat_counts.get('QATLinear', 0) + 1
    print(f"  QAT 层: {qat_counts}")

    # ---- 硬件对齐率 ----
    alignment = print_alignment_detail(model, "OpticSpaceNetStudent (QAT)")

    # ---- 校准 ----
    print(f"\n[Phase 2b] 校准 QAT 模型...")
    calibrate_qat_model(model, train_loader, DEVICE, num_batches=3)

    # ---- Phase 3: QAT 微调 ----
    print(f"\n[Phase 3] QAT 微调 ({QAT_EPOCHS} epochs, lr={LEARNING_RATE})")
    if use_kd:
        print(f"  模式: KD + QAT (T={TEMPERATURE}, α={ALPHA})")
        print(f"  教师模型固定, 学生通过蒸馏损失 + 伪量化进行微调")
    else:
        print(f"  模式: 标准 QAT (CrossEntropy loss)")

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

        if use_kd and teacher is not None:
            # KD + QAT 联合训练
            train_loss, train_acc = train_qat_kd_epoch(
                model, teacher, train_loader, optimizer
            )
        else:
            # 标准 QAT 训练
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
    print(f"  学生模型:          OpticSpaceNet (硬件完美对齐)")
    print(f"  学生参数量:        {param_count:,}")
    if use_kd and teacher is not None:
        print(f"  教师模型:          ResNet-18")
    print(f"  Float32 基准准确率: {result_fp32['accuracy']:.2%}")
    print(f"  QAT 最佳准确率:     {best_acc:.2%}")
    print(f"  QAT vs Float gap:  {comparison['accuracy_gap']:.2%}")
    print(f"  QAT 训练耗时:      {total_train_time:.1f} 秒 ({total_train_time/60:.1f} 分钟)")
    print(f"  8×2 硬件对齐率:    {alignment:.1%}")
    print(f"  QAT 模式:          {'KD + QAT' if use_kd else '标准 QAT'}")
    print(f"\n  预期光计算推理:")
    print(f"    原生 float32:     {result_fp32['accuracy']:.1%}")
    print(f"    PTQ (直接 int4):  ~{result_fp32['accuracy'] - 0.12:.1f} (大幅下降)")
    print(f"    QAT (微调后 int4): {best_acc:.1f} (接近 float32)")
    print(f"\n  💡 提示: 如需更好的精度，使用 KD+QAT 联合微调:")
    print(f"     python model3_spacenet_v2_qat.py --use_kd")

    return best_acc, alignment, total_train_time


if __name__ == "__main__":
    main()
