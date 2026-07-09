"""
================================================================================
 模型二 LSQ+ int8 (SpaceNet V1): 可学习 scale/zero_point + Gazelle 硬件噪声
================================================================================
 修复版 LSQ+ (vs optic_qat_v2):
   - int8 激活 (256 级) 替代 uint4 (16 级)
   - in_scale/in_zp 真正参与前向并接收 LSQ 梯度
   - 无内部 ReLU, 无内部输出重量化
   - BN 保留, 首层 FP32
   - scale 初始化从统计计算, 非 1.0

 训练策略:
   - Warmup 阶段 (前 10 epoch): STE fallback (稳定激活分布)
   - LSQ 阶段 (后续): 切换到 LSQ+ 可学习 scale/zp
   - LSQ 参数独立学习率 (0.1x base_lr)

 用法:
   python model2_spacenet_v1_lsq.py
================================================================================
"""

import torch
import torch.nn as nn
import sys
import time
import numpy as np

from optic_qat_lsq import (
    prepare_model_lsq, enable_qat, disable_qat,
    evaluate_model_lsq, set_lsq_lr,
)
from train_phase4_runner import WarmupCosineScheduler, load_eurosat_data


# ============================================================
#  模型
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"设备: {DEVICE}")
    print(f"\n{'='*60}")
    print(f"  Model 2 LSQ+ int8: 可学习 scale/zero_point + Gazelle 噪声")
    print(f"  修复: in_scale 真正参与前向, int8 激活, BN 保留")
    print(f"{'='*60}")

    train_loader, val_loader = load_eurosat_data(batch_size=64)

    model = OpticSpaceNetV1(num_classes=10)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === Step 1: 转换 ===
    print(f"\n[Step 1] 转换为 LSQ+ int8")
    prepare_model_lsq(model, weight_bits=8, act_bits=8,
                      first_conv_fp32=True, quantize_linear=True)

    # === Step 2: 训练 ===
    epochs = 100
    lsq_warmup = 10  # LSQ 预热: 前 10 epoch 用 STE, 之后切 LSQ+
    base_lr = 0.001
    weight_decay = 5e-4

    print(f"\n[Step 2] 训练 ({epochs} epochs, LSQ warmup={lsq_warmup})")
    print(f"  前 {lsq_warmup} epochs: STE fallback (稳定激活分布)")
    print(f"  后 {epochs - lsq_warmup} epochs: LSQ+ (可学习 scale/zp, lr=0.1x)")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 初始: STE 模式 (所有 LSQ 层切换为 STE)
    for m in model.modules():
        if hasattr(m, '_use_lsq'):
            m._use_lsq = False

    param_groups = set_lsq_lr(model, base_lr=base_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5,
                                      total_epochs=epochs, base_lr=base_lr)

    best_acc, best_state = 0.0, None
    total_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'LR':>8s} | {'Time':>7s}")
    print("  " + "-" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 切换到 LSQ+ 模式
        if epoch == lsq_warmup + 1:
            print(f"  >>> Epoch {epoch}: 切换到 LSQ+ (可学习 scale/zp) <<<")
            for m in model.modules():
                if hasattr(m, '_use_lsq') and m.qat_enabled:
                    m._use_lsq = True
            # 重建优化器 (LSQ 参数需要独立 lr)
            param_groups = set_lsq_lr(model, base_lr=base_lr)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        # 验证
        val_result = evaluate_model_lsq(model, val_loader, DEVICE, criterion)
        val_loss, val_acc = val_result['loss'], val_result['accuracy']

        elapsed = time.time() - t0
        total_time += elapsed
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            mode = "LSQ+" if epoch > lsq_warmup else "STE"
            print(f"  {epoch:>5d}  | {train_loss/train_total:>10.4f} "
                  f"{train_correct/train_total:>8.2%} | "
                  f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | "
                  f"{optimizer.param_groups[0]['lr']:>7.5f} | {elapsed:>6.1f}s  [{mode}]")

    model.load_state_dict(best_state)

    # === Step 3: 评估 ===
    print(f"\n[Step 3] 最终评估")

    # LSQ+ 模式
    for m in model.modules():
        if hasattr(m, '_use_lsq') and m.qat_enabled:
            m._use_lsq = True
    enable_qat(model)
    result_lsq = evaluate_model_lsq(model, val_loader, DEVICE, criterion)
    print(f"  LSQ+ int8 模式准确率: {result_lsq['accuracy']:.2%}")

    disable_qat(model)
    result_fp32 = evaluate_model_lsq(model, val_loader, DEVICE, criterion)
    print(f"  Float32 模式准确率:   {result_fp32['accuracy']:.2%}")
    print(f"  LSQ+ 量化损失:        {result_fp32['accuracy'] - result_lsq['accuracy']:.2%}")

    fname = "spacenet_v1_lsq_int8.pth"
    torch.save(model.state_dict(), fname)
    print(f"\n  模型已保存: {fname}")

    # 统计 LSQ 参数
    lsq_count = sum(1 for m in model.modules()
                    if hasattr(m, 'lsq_params'))
    print(f"\n{'='*60}")
    print(f"  训练完成 — 结果汇总")
    print(f"{'='*60}")
    print(f"  模型:              SpaceNet V1 (LSQ+ int8, bias=False)")
    print(f"  参数量:            {sum(p.numel() for p in model.parameters()):,}")
    print(f"  LSQ+ 可学习层:     {lsq_count}")
    print(f"  训练策略:          STE warmup {lsq_warmup}ep + LSQ+ {epochs-lsq_warmup}ep")
    print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  LSQ+ int8 最佳:    {best_acc:.2%}")
    print(f"  Float32:           {result_fp32['accuracy']:.2%}")
    print(f"  STE int8 参考:     93.11% (Phase4 v3)")
    print(f"  FP32 基准:         90.15%")
    print(f"{'='*60}")

    return best_acc


if __name__ == "__main__":
    main()
