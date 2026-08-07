"""
================================================================================
 模型二 Phase4 v3 (SpaceNet V1): int8 权重 + Gazelle 硬件噪声
================================================================================
 改进 (vs v2):
   - int8 权重 (匹配硬件原生 8-bit) → 量化噪声降低 16 倍
   - Gazelle 硬件匹配噪声 (DAC ENOB=7.5 + TIA 噪声)
   - 首层 stem Conv 保留 FP32 (对齐率仅 37.5%)

 用法:
   python model2_spacenet_v1_phase4_v3.py                # int8 (推荐)
   python model2_spacenet_v1_phase4_v3.py --wbits 4      # int4 对比

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
import sys
import time
import numpy as np

from optic_qat_v4 import (
    prepare_model_v4, enable_qat, disable_qat,
    evaluate_model_v4, print_alignment_detail, GazelleNoiseInjector,
)
from train_phase4_runner import WarmupCosineScheduler, load_eurosat_data


# ============================================================
#  模型 (bias=False, 匹配光计算硬件)
# ============================================================

class OpticSpaceNetV1(nn.Module):
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
def main():
    # 解析参数
    wbits = 8  # 默认 int8 (匹配硬件原生精度)
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
    print(f"  Model 2 Phase4 v3: int{wbits} 权重 + Gazelle 硬件噪声")
    print(f"  首层 stem FP32 (对齐率 37.5%), 其余 Conv+Linear int{wbits}")
    print(f"{'='*60}")

    epochs = 120 if wbits == 4 else 100  # int4 需要更多 epoch
    train_loader, val_loader = load_eurosat_data(batch_size=64)

    model = OpticSpaceNetV1(num_classes=10)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Step 1: 转换模型
    print(f"\n[Step 1] 转换为 QAT v4 (int{wbits} 权重, Gazelle 噪声)")
    prepare_model_v4(model, weight_bits=wbits, act_bits=8,
                    noise=True, first_conv_fp32=True,
                    quantize_linear=True, preserve_bn=True)
    alignment = print_alignment_detail(model, "SpaceNet V1 (v4)")

    # Step 2: 训练
    base_lr = 0.001
    warmup = 5
    weight_decay = 5e-4 if wbits == 8 else 1e-4
    label_smoothing = 0.05 if wbits == 8 else 0.0  # int8 可以加点 label smoothing

    print(f"\n[Step 2] 训练 ({epochs} epochs, lr={base_lr}, wd={weight_decay})")
    print(f"  int{wbits} 权重 {'(硬件原生精度)' if wbits == 8 else '(保守, 硬件有余量)'}")
    print(f"  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup,
                                      total_epochs=epochs, base_lr=base_lr)

    # --- checkpoint 续训 (断点续跑支持, 每 5 epoch 保存) ---
    fname = f"weights/spacenet_v1_phase4_v3_int{wbits}.pth"
    ckpt_path = fname + ".ckpt"
    start_epoch, ckpt_interval = 1, 5
    best_acc, best_state = 0.0, None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.current_epoch = ck["scheduler_epoch"]
        best_acc, best_state = ck["best_acc"], ck["best_state"]
        start_epoch = ck["epoch"] + 1
        print(f"\n[resume] 从 checkpoint 恢复: epoch {ck['epoch']}/{epochs} "
              f"(best {best_acc:.2%}) → 继续 epoch {start_epoch}")

    total_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'LR':>8s} | {'Time':>7s}")
    print("  " + "-" * 68)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (model(images).argmax(1) == labels).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        val_result = evaluate_model_v4(model, val_loader, DEVICE, criterion)
        val_loss, val_acc = val_result['loss'], val_result['accuracy']
        elapsed = time.time() - t0
        total_time += elapsed
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5d}  | {train_loss:>10.4f} {train_acc:>8.2%} | "
                  f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | "
                  f"{optimizer.param_groups[0]['lr']:>7.5f} | {elapsed:>6.1f}s")

        # checkpoint 续训
        if epoch % ckpt_interval == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler_epoch": scheduler.current_epoch,
                        "best_acc": best_acc, "best_state": best_state},
                       ckpt_path)
            print(f"  [ckpt] epoch {epoch} → {ckpt_path}")

    # 恢复最佳权重
    model.load_state_dict(best_state)

    # Step 3: 最终评估
    print(f"\n[Step 3] 最终评估")
    enable_qat(model)
    result_qat = evaluate_model_v4(model, val_loader, DEVICE, criterion)
    print(f"  Int{wbits} 模式 (光计算模拟) 准确率: {result_qat['accuracy']:.2%}")

    disable_qat(model)
    result_fp32 = evaluate_model_v4(model, val_loader, DEVICE, criterion)
    print(f"  Float32 模式准确率:              {result_fp32['accuracy']:.2%}")
    print(f"  Int{wbits} 量化损失:             "
          f"{result_fp32['accuracy'] - result_qat['accuracy']:.2%}")

    fname = f"weights/spacenet_v1_phase4_v3_int{wbits}.pth"
    torch.save(model.state_dict(), fname)
    print(f"\n  模型已保存: {fname}")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  [ckpt] 已清理 (训练完成, 删除 {ckpt_path})")

    print(f"\n{'='*60}")
    print(f"  训练完成 — 结果汇总")
    print(f"{'='*60}")
    print(f"  模型:              SpaceNet V1 (bias=False)")
    print(f"  参数量:            {sum(p.numel() for p in model.parameters()):,}")
    print(f"  权重量化:          int{wbits} {'(硬件原生 8-bit)' if wbits == 8 else '(保守)'}")
    print(f"  噪声模型:          Gazelle (DAC 7.5 + TIA)")
    print(f"  首层:              FP32 (对齐率 37.5%)")
    print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  硬件对齐率:        {alignment:.1%}")
    print(f"  Int{wbits} 最佳准确率: {best_acc:.2%}")
    print(f"  Float32 准确率:    {result_fp32['accuracy']:.2%}")
    print(f"  旧版 int4 参考:    74.35% (Phase4, Conv QAT 全关)")
    print(f"  FP32 基准:         90.15%")
    print(f"{'='*60}")

    return best_acc


if __name__ == "__main__":
    main()
