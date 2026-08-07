"""
================================================================================
 模型三 Phase4 v3 (SpaceNet V2 KD): int8 权重 + Gazelle 硬件噪声 + KD
================================================================================
 改进 (vs v2 int4):
   - int8 权重 (匹配 osimulator 原生 8-bit) → 量化噪声降低 16 倍
   - Gazelle 硬件匹配噪声 (DAC ENOB=7.5 + TIA 噪声)
   - 首层 stem Conv 保留 FP32 (对齐率仅 37.5%, 匹配 osimulator 推理路径)
   - KD 蒸馏 (ResNet-18 Teacher, 97.83%), T=4.0, α=0.7

 与 osimulator 推理路径天然对齐:
   - 训练: first_conv_fp32=True  →  推理: keep_first_conv_electronic=True
   - 训练: int8 权重             →  推理: weight_bit=8 (osimulator 原生)
   - 预期光学精度: 92-93% (vs v2 int4 的 ~84.5%)

 用法:
   python model3_spacenet_v2_phase4_v3.py                  # int8+KD (推荐)
   python model3_spacenet_v2_phase4_v3.py --wbits 4        # int4+KD 对比

 产出权重: weights/spacenet_v2_phase4_v3_int8.pth (或 _int4.pth)

  TODO (v4.1 修复后, 见 docs/TODO.md §v4.1 重跑清单):
    - [ ] 重训: optic_qat_v4 激活量化改为 uint8+zp 且训练时注入激活噪声,
          旧权重是旧语义训练, 需重训后复测 osim gap (预期从 ~1.6pt 收窄)
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import numpy as np

from optic_qat_v4 import (
    prepare_model_v4, enable_qat, disable_qat,
    evaluate_model_v4, print_alignment_detail, GazelleNoiseInjector,
)
from train_phase4_runner import WarmupCosineScheduler, load_eurosat_data


# ============================================================
#  教师模型
# ============================================================

def create_teacher(num_classes=10):
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
#  学生模型: SpaceNet — bias=False, 匹配光计算硬件
# ============================================================

class OpticSpaceNetStudent(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.classifier(x)
        return x


# ============================================================
#  KD + Phase4 v3 训练器 (int8 + Gazelle 噪声)
# ============================================================

class KDPhase4V3Trainer:
    """KD + Phase4 v3: 教师(fp32) → 学生(int8 QAT + Gazelle 噪声, stem FP32)"""

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

        wbits = cfg.get('weight_bits', 8)

        # === Step 1: 转换学生 ===
        # 关键配置: first_conv_fp32=True → stem 保持 FP32 (匹配 osimulator)
        #           其余 Conv+Linear → int8 QAT + Gazelle 噪声
        print(f"\n[Step 1] 转换学生: stem FP32, 其余 Conv+Linear→int{wbits} QAT")
        print(f"  噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4)")
        prepare_model_v4(
            student,
            weight_bits=wbits,
            act_bits=cfg.get('act_bits', 8),
            noise=cfg.get('noise', True),
            first_conv_fp32=True,       # ★ stem FP32 (匹配 osimulator)
            quantize_linear=True,        # ★ Linear 也量化
            preserve_bn=True,
        )

        qc = sum(1 for m in student.modules()
                 if hasattr(m, 'qat_enabled') and m.qat_enabled
                 and 'Conv' in type(m).__name__)
        ql = sum(1 for m in student.modules()
                 if hasattr(m, 'qat_enabled') and m.qat_enabled
                 and 'Linear' in type(m).__name__)
        bn = sum(1 for m in student.modules() if isinstance(m, nn.BatchNorm2d))
        print(f"  int{wbits} QAT Conv: {qc}, int{wbits} QAT Linear: {ql}, "
              f"BN: {bn} (float32), stem: FP32")

        alignment = print_alignment_detail(student,
                                           f"Student (stem FP32 + int{wbits})")

        # === Step 2: KD 训练 ===
        epochs = cfg.get('epochs', 100)
        base_lr = cfg.get('learning_rate', 0.001)
        warmup = cfg.get('warmup_epochs', 5)
        T = cfg.get('temperature', 4.0)
        alpha = cfg.get('alpha', 0.7)
        weight_decay = cfg.get('weight_decay', 5e-4)
        label_smoothing = cfg.get('label_smoothing', 0.05)

        print(f"\n[Step 2] KD+Phase4 v3 训练 ({epochs} epochs, T={T}, α={alpha})")
        print(f"  教师: ResNet-18 (fp32, 97.83%)")
        print(f"  学生: stem FP32 + Conv/Linear int{wbits} + Gazelle 噪声")
        print(f"  int{wbits} 权重 {'(硬件原生精度, 匹配 osimulator)' if wbits == 8 else '(保守)'}")

        student.to(self.device)

        criterion_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.AdamW(student.parameters(), lr=base_lr,
                                weight_decay=weight_decay)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup,
            total_epochs=epochs, base_lr=base_lr
        )

        best_acc, best_state = 0.0, None
        total_time = 0.0

        # --- checkpoint 续训 (断点续跑支持, 每 5 epoch 保存) ---
        fname = cfg.get('save_path', f'weights/spacenet_v2_phase4_v3_int{wbits}.pth')
        ckpt_path = fname + ".ckpt"
        start_epoch, ckpt_interval = 1, 5
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            student.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.current_epoch = ck["scheduler_epoch"]
            best_acc, best_state = ck["best_acc"], ck["best_state"]
            start_epoch = ck["epoch"] + 1
            print(f"\n[resume] 从 checkpoint 恢复: epoch {ck['epoch']}/{epochs} "
                  f"(best {best_acc:.2%}) → 继续 epoch {start_epoch}")

        print("-" * 75)
        print(f"  {'Epoch':>5s} | {'KD Loss':>10s} {'Train Acc':>9s} | "
              f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | "
              f"{'LR':>8s} | {'Time':>7s}")
        print("  " + "-" * 73)

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()

            # 训练
            kd_loss, train_acc = self._train_epoch(
                student, teacher, T, alpha, optimizer, criterion_ce)

            # 验证
            val_result = evaluate_model_v4(student, self.val_loader,
                                           self.device, criterion_ce)
            val_loss, val_acc = val_result['loss'], val_result['accuracy']
            elapsed = time.time() - t0
            total_time += elapsed
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone()
                              for k, v in student.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>5d}  | {kd_loss:>10.4f} {train_acc:>8.2%} | "
                      f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | "
                      f"{optimizer.param_groups[0]['lr']:>7.5f} | {elapsed:>6.1f}s")

            # checkpoint 续训
            if epoch % ckpt_interval == 0:
                torch.save({"epoch": epoch, "model": student.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler_epoch": scheduler.current_epoch,
                            "best_acc": best_acc, "best_state": best_state},
                           ckpt_path)
                print(f"  [ckpt] epoch {epoch} → {ckpt_path}")

        # 恢复最佳权重
        student.load_state_dict(best_state)

        # === Step 3: 最终评估 ===
        print(f"\n[Step 3] 最终评估")
        enable_qat(student)
        result_qat = evaluate_model_v4(student, self.val_loader, self.device,
                                       criterion_ce)
        print(f"  Int{wbits} 模式 (光计算模拟) 准确率: {result_qat['accuracy']:.2%}")

        disable_qat(student)
        result_fp32 = evaluate_model_v4(student, self.val_loader, self.device,
                                        criterion_ce)
        print(f"  Float32 模式准确率:              {result_fp32['accuracy']:.2%}")
        print(f"  Int{wbits} 量化损失:             "
              f"{result_fp32['accuracy'] - result_qat['accuracy']:.2%}")

        fname = cfg.get('save_path', f'weights/spacenet_v2_phase4_v3_int{wbits}.pth')
        torch.save(student.state_dict(), fname)
        print(f"\n  模型已保存: {fname}")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"  [ckpt] 已清理 (训练完成, 删除 {ckpt_path})")

        print(f"\n{'='*60}")
        print(f"  训练完成 — 结果汇总")
        print(f"{'='*60}")
        print(f"  学生模型:          OpticSpaceNet (bias=False)")
        print(f"  教师模型:          ResNet-18 (fp32, 97.83%)")
        print(f"  参数量:            {sum(p.numel() for p in student.parameters()):,}")
        print(f"  权重量化:          int{wbits} "
              f"{'(匹配 osimulator 原生 8-bit)' if wbits == 8 else '(保守)'}")
        print(f"  噪声模型:          Gazelle (DAC 7.5 + TIA)")
        print(f"  首层:              FP32 (对齐率 37.5%, 匹配 osimulator)")
        print(f"  蒸馏:              T={T}, α={alpha}")
        print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  硬件对齐率:        {alignment:.1%}")
        print(f"  Int{wbits} 最佳准确率: {best_acc:.2%}")
        print(f"  Float32 准确率:    {result_fp32['accuracy']:.2%}")
        print(f"  FP32 KD 基准:      {cfg.get('fp32_baseline', '91.44%')}")
        print(f"  osimulator 预期:   ~{best_acc:.1%}% "
              f"(训练推理配置对齐, 应接近训练精度)")
        print(f"{'='*60}")

        return best_acc

    def _train_epoch(self, student, teacher, T, alpha, optimizer, criterion_ce):
        student.train()
        total_kd_loss, correct, total = 0.0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            s_logits = student(images)
            with torch.no_grad():
                t_logits = teacher(images)

            # KD loss
            soft_s = F.log_softmax(s_logits / T, dim=1)
            soft_t = F.softmax(t_logits / T, dim=1)
            kd = F.kl_div(soft_s, soft_t, reduction='batchmean') * (T * T)

            # CE loss (with optional label smoothing)
            ce = criterion_ce(s_logits, labels)

            loss = alpha * kd + (1 - alpha) * ce
            loss.backward()
            optimizer.step()

            total_kd_loss += kd.item() * images.size(0)
            correct += (s_logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        return total_kd_loss / total, correct / total


# ============================================================
def main():
    # 解析参数
    wbits = 8  # 默认 int8 (匹配 osimulator 原生精度)
    if "--wbits" in sys.argv:
        idx = sys.argv.index("--wbits")
        if idx + 1 < len(sys.argv):
            wbits = int(sys.argv[idx + 1])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"设备: {DEVICE}")
    print(f"\n{'='*60}")
    print(f"  Model 3 Phase4 v3: KD + int{wbits} + Gazelle 噪声")
    print(f"  stem FP32 (匹配 osimulator) + Conv/Linear int{wbits} QAT")
    print(f"{'='*60}")

    train_loader, val_loader = load_eurosat_data(batch_size=64)

    # 教师
    print(f"\n[Step 0] 加载教师 (ResNet-18)")
    teacher = create_teacher(num_classes=10)
    try:
        teacher.load_state_dict(torch.load("weights/teacher_resnet18.pth", map_location='cpu'))
        print(f"  教师权重加载成功 (97.83%)")
    except FileNotFoundError:
        print(f"  [ERROR] 教师权重未找到: weights/teacher_resnet18.pth")
        print(f"  请先运行 model3_spacenet_v2.py 生成教师权重")
        return None
    teacher.to(DEVICE)

    # 学生
    student = OpticSpaceNetStudent(num_classes=10)
    print(f"  学生参数量: {sum(p.numel() for p in student.parameters()):,}")
    print(f"  架构: 4×Conv + 2×Linear, bias=False, BN 保留")

    config = {
        'weight_bits': wbits,
        'act_bits': 8,
        'noise': True,
        'epochs': 100 if wbits == 8 else 120,  # int4 需要更多 epoch
        'learning_rate': 0.001,
        'warmup_epochs': 5,
        'weight_decay': 5e-4 if wbits == 8 else 1e-4,
        'label_smoothing': 0.05 if wbits == 8 else 0.0,
        'temperature': 4.0,
        'alpha': 0.7,
        'model_name': f'OpticSpaceNet Phase4 v3 (KD + int{wbits} + Gazelle)',
        'save_path': f'weights/spacenet_v2_phase4_v3_int{wbits}.pth',
        'fp32_baseline': '91.44% (全 fp32 KD)',
        'device': DEVICE,
        'num_classes': 10,
    }

    trainer = KDPhase4V3Trainer(teacher, student, train_loader, val_loader, config)
    best_acc = trainer.run()
    return best_acc


if __name__ == "__main__":
    main()
