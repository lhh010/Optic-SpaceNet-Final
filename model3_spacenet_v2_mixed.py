"""
================================================================================
 模型三 Mixed (Optic-SpaceNet V2 KD): KD + Conv=int4 (光计算) + Linear=fp32 (电计算)
================================================================================
 量化策略:
   - 所有 Conv2d: int4 QAT + 知识蒸馏引导, bias=False
   - 所有 Linear:  float32 常规训练, bias=True
   - 教师 (ResNet-18): fp32 固定, 提供软标签

 蒸馏配置:
   - T=4.0, α=0.7 (高 KD 权重 — 教师强引导 int4 特征学习)
   - Loss = 0.7*KD + 0.3*CE

 与全部 int4 KD 方案 (model3_spacenet_v2_phase4.py) 对比:
   - 全部 int4: KD + 全部层 int4 QAT
   - 混合方案:   KD + Conv int4 + Linear fp32
   - Linear 保留 fp32 使分类头保持高精度

 用法:
   python model3_spacenet_v2_mixed.py                  # STE + KD (推荐)
   python model3_spacenet_v2_mixed.py --mode lsqplus   # LSQ+ + KD
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import numpy as np

from optic_qat_v3 import (
    prepare_model_v3, enable_qat, disable_qat,
    evaluate_model_v3, print_alignment_detail,
    set_quant_lr,
)
from train_mixed_runner import WarmupCosineScheduler, load_eurosat_data


# ============================================================
#  教师模型
# ============================================================

def create_teacher_model(num_classes=10):
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
#  学生模型: OpticSpaceNet — Conv(int4) + Linear(fp32)
# ============================================================

class OpticSpaceNetStudent(nn.Module):
    """硬件对齐 CNN: Conv → 光计算 int4, Linear → 电计算 fp32"""

    def __init__(self, num_classes=10):
        super().__init__()

        # === 卷积部分 (光计算 int4, bias=False) ===
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

        # === 全连接部分 (电计算 fp32, bias=True) ===
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x


# ============================================================
#  KD 训练器 (Conv=int4 + Linear=fp32)
# ============================================================

class KDMixedTrainer:
    """KD + 混合精度训练: 教师(fp32) 引导 学生(Conv=int4, Linear=fp32)"""

    def __init__(self, teacher, student, train_loader, val_loader, config):
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get('device', torch.device('cpu'))

    def run(self):
        cfg = self.config
        student = self.student
        teacher = self.teacher

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.to(self.device)

        # === 转换学生 ===
        print(f"\n[Step 1] 转换学生: Conv=int4 QAT, Linear=fp32")
        prepare_model_v3(
            student,
            mode=cfg['mode'],
            weight_bits=cfg.get('weight_bits', 4),
            act_bits=cfg.get('act_bits', 8),
            noise=cfg.get('noise', True),
            noise_std_ratio=cfg.get('noise_std_ratio', 0.02),
            quantize_linear=False,
            preserve_bn=True,
        )

        qc = sum(1 for m in student.modules()
                 if type(m).__name__ == 'QATConv2d_v3' and m.qat_enabled)
        lin = sum(1 for m in student.modules() if isinstance(m, nn.Linear))
        print(f"  int4 QAT Conv: {qc}, fp32 Linear: {lin}")

        alignment = print_alignment_detail(student, "Student (Conv=int4, Linear=fp32)")

        # === 训练 ===
        epochs = cfg.get('epochs', 120)
        base_lr = cfg.get('learning_rate', 0.001)
        warmup = cfg.get('warmup_epochs', 5)
        T = cfg.get('temperature', 4.0)
        alpha = cfg.get('alpha', 0.7)

        print(f"\n[Step 2] KD+混合精度训练 ({epochs} epochs)")
        print(f"  蒸馏: T={T}, α={alpha}")
        print(f"  教师: ResNet-18 (fp32), 学生: Conv=int4 + Linear=fp32")

        student.to(self.device)

        if cfg['mode'] == 'lsqplus':
            param_groups = set_quant_lr(student, base_lr=base_lr)
            optimizer = optim.AdamW(param_groups, weight_decay=cfg.get('weight_decay', 1e-4))
        else:
            optimizer = optim.AdamW(student.parameters(), lr=base_lr,
                                    weight_decay=cfg.get('weight_decay', 5e-4))

        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup,
            total_epochs=epochs, base_lr=base_lr
        )

        best_acc, best_state = 0.0, None
        total_time = 0.0

        print("-" * 70)
        print(f"  {'Epoch':>5s} | {'KD Loss':>10s} {'Train Acc':>9s} | "
              f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'Time':>7s}")
        print("  " + "-" * 65)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            kd_loss, train_acc = self._train_epoch(student, teacher, T, alpha, optimizer)
            val_result = evaluate_model_v3(student, self.val_loader, self.device,
                                           nn.CrossEntropyLoss())
            val_loss, val_acc = val_result['loss'], val_result['accuracy']
            elapsed = time.time() - t0
            total_time += elapsed
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>5d}  | {kd_loss:>10.4f} {train_acc:>8.2%} | "
                      f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | {elapsed:>6.1f}s")

        student.load_state_dict(best_state)

        # === 评估 ===
        print(f"\n[Step 3] 最终评估")
        enable_qat(student)
        result_qat = evaluate_model_v3(student, self.val_loader, self.device,
                                       nn.CrossEntropyLoss())
        print(f"  Int4 模式 (Conv=int4 光计算) 准确率: {result_qat['accuracy']:.2%}")

        disable_qat(student)
        result_fp32 = evaluate_model_v3(student, self.val_loader, self.device,
                                        nn.CrossEntropyLoss())
        print(f"  Float32 模式准确率:                {result_fp32['accuracy']:.2%}")

        fname = cfg.get('save_path', 'spacenet_v2_mixed_ste.pth')
        torch.save(student.state_dict(), fname)
        print(f"\n  模型已保存: {fname}")

        print(f"\n{'='*60}")
        print(f"  训练完成 — 结果汇总")
        print(f"{'='*60}")
        print(f"  策略:              Conv=int4 (光计算) + Linear=fp32 (电计算)")
        print(f"  教师:              ResNet-18 (fp32)")
        print(f"  学生参数量:        {sum(p.numel() for p in student.parameters()):,}")
        print(f"  蒸馏:              T={T}, α={alpha}")
        print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  硬件对齐率:        {alignment:.1%}")
        print(f"  Int4 最佳准确率:   {best_acc:.2%}")
        print(f"  Float32 准确率:    {result_fp32['accuracy']:.2%}")
        print(f"  FP32 KD 基准:      {cfg.get('fp32_baseline', 'N/A')}")
        print(f"{'='*60}")

        return best_acc

    def _train_epoch(self, student, teacher, T, alpha, optimizer):
        student.train()
        total_kd_loss, correct, total = 0.0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            soft_s = F.log_softmax(student_logits / T, dim=1)
            soft_t = F.softmax(teacher_logits / T, dim=1)
            kd = F.kl_div(soft_s, soft_t, reduction='batchmean') * (T * T)
            ce = F.cross_entropy(student_logits, labels)
            loss = alpha * kd + (1 - alpha) * ce
            loss.backward()
            optimizer.step()
            total_kd_loss += kd.item() * images.size(0)
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        return total_kd_loss / total, correct / total


# ============================================================
#  主函数
# ============================================================

def main():
    mode = "ste"
    act_bits = 8
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
    if "--act-bits" in sys.argv:
        idx = sys.argv.index("--act-bits")
        if idx + 1 < len(sys.argv):
            act_bits = int(sys.argv[idx + 1])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"设备: {DEVICE}, 模式: {mode}, act_bits: {act_bits}")

    print(f"\n{'='*60}")
    print(f"  Model 3 Mixed: KD + Conv=int4 (光计算) + Linear=fp32 (电计算)")
    print(f"{'='*60}")
    train_loader, val_loader = load_eurosat_data(batch_size=64)

    # 教师
    print(f"\n[Step 0] 加载教师 (ResNet-18)")
    teacher = create_teacher_model(num_classes=10)
    try:
        teacher.load_state_dict(torch.load("teacher_resnet18.pth", map_location='cpu'))
        print(f"  教师权重加载成功")
    except FileNotFoundError:
        print(f"  [!] 教师权重未找到, 请先运行 model3_spacenet_v2.py")
        return None
    teacher.to(DEVICE)

    # 学生
    student = OpticSpaceNetStudent(num_classes=10)
    print(f"  学生参数量: {sum(p.numel() for p in student.parameters()):,}")
    print(f"  4×Conv → int4 QAT, 2×Linear → fp32")

    config = {
        'mode': mode,
        'weight_bits': 4,
        'act_bits': act_bits,
        'noise': (mode == 'ste'),
        'noise_std_ratio': 0.02,
        'epochs': 120,
        'learning_rate': 0.001,
        'warmup_epochs': 5,
        'weight_decay': 5e-4 if mode == 'ste' else 1e-4,
        'temperature': 4.0,
        'alpha': 0.7,
        'model_name': f'OpticSpaceNet KD Mixed (Conv=int4, Linear=fp32, {mode})',
        'save_path': f'spacenet_v2_mixed_{mode}.pth',
        'fp32_baseline': '91.44% (全 fp32 KD)',
        'device': DEVICE,
        'num_classes': 10,
    }

    trainer = KDMixedTrainer(teacher, student, train_loader, val_loader, config)
    best_acc = trainer.run()
    return best_acc


if __name__ == "__main__":
    main()
