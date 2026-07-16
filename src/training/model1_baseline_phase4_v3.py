"""
================================================================================
 模型一 Phase4 v3 (Baseline VGG): int8 权重 + Gazelle 硬件噪声
================================================================================
 改进 (vs Phase4 int4):
   - int8 权重 (匹配硬件原生 8-bit) → 量化噪声降低 16 倍
   - Gazelle 硬件匹配噪声 (DAC ENOB=7.5 + TIA 噪声)
   - 首层 conv1_1 保留 FP32 (处理原始 RGB, 量化最难)
   - flat + BN + bias=False 架构 (匹配光计算硬件, 稳定 QAT)

 这是 Model 2/3 v3 int8 配方向 Model 1 的移植。
 训练<->推理参数对齐 checklist (§16.8):
   - conv1_1: 训练 first_conv_fp32=True <-> 推理 keep_first_conv_electronic=True
   - 其余层: 训练 int8 QAT <-> 推理 OpticConv2d/OpticLinear (8a8w)
   - 噪声: 训练 GazelleNoise <-> 推理 osimulator 物理噪声

 变体 (光计算占比 vs 速度/精度 消融):
   --variant A (默认): 仅 conv1_1 电计算 → 光计算占比 97.7%
   --variant B:        conv1_1 + conv3_2 电计算 → 光计算占比 73.7%, osimulator 快 ~24%

 用法:
   python model1_baseline_phase4_v3.py                # 变体 A (int8, 推荐)
   python model1_baseline_phase4_v3.py --variant B    # 变体 B (conv3_2 也 FP32)
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
    evaluate_model_v4, print_alignment_detail,
)
from train_phase4_runner import WarmupCosineScheduler, load_eurosat_data


# ============================================================
#  模型 (flat + BN + bias=False, 匹配光计算硬件)
#  与 optic_inference_int8_model1.py 的 BaselineVGG 完全一致
# ============================================================

class BaselineVGG(nn.Module):
    """
    Model 1 Phase4 v3: flat VGG, 全 bias=False, BN 保留 (float32 运行稳定 QAT)。

    架构 (6 Conv 3×3 + 2 Linear), 输入 (N,3,64,64):
      block1: conv1_1(3→32) → conv1_2(32→32) → pool    [conv1_1 = FP32 首层]
      block2: conv2_1(32→64) → conv2_2(64→64) → pool
      block3: conv3_1(64→128) → conv3_2(128→128) → pool
      classifier: fc1(8192→256) → fc2(256→10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 3→32, 64×64→32×32 (经 pool)
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 32→64, 32×32→16×16 (经 pool)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 64→128, 16×16→8×8 (经 pool)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Classifier: 128×8×8=8192 → 256 → 10
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================
#  变体 B 后处理: 把 conv3_2 永久保持 FP32 (电计算)
#  不改 optic_qat_v4.py, 仅在脚本内设置 _keep_fp32 标志
# ============================================================

def keep_conv_fp32(model, conv_name):
    """
    将指定 QATConv2d_v4 标记为永久 FP32 (电计算)。
    - _keep_fp32=True: enable_qat() 不会重新打开它
    - _qat_enabled=False: 当前关闭 QAT, 走原生 FP32 conv

    这样训练时该层全程 FP32, 与推理时该层保留电计算 (Conv2d) 严格对齐。
    """
    from optic_qat_v4 import QATConv2d_v4
    layer = getattr(model, conv_name)
    assert isinstance(layer, QATConv2d_v4), (
        f"{conv_name} 应为 QATConv2d_v4, 实际 {type(layer).__name__}")
    layer._keep_fp32 = True
    layer._qat_enabled = False
    print(f"  [Variant B] {conv_name} 保持 FP32 (电计算): "
          f"{layer.in_channels}→{layer.out_channels}")


# ============================================================
def main():
    # ---- 解析参数 ----
    variant = "A"
    if "--variant" in sys.argv:
        idx = sys.argv.index("--variant")
        if idx + 1 < len(sys.argv):
            variant = sys.argv[idx + 1].upper().strip()
    assert variant in ("A", "B"), f"未知变体: {variant} (应为 A 或 B)"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"设备: {DEVICE}")
    print(f"\n{'='*60}")
    print(f"  Model 1 Phase4 v3: Baseline VGG int8 + Gazelle 硬件噪声")
    print(f"  变体 {variant}: 首层 conv1_1 FP32"
          + (" + conv3_2 FP32" if variant == "B" else "")
          + f", 其余 Conv+Linear int8")
    print(f"{'='*60}")

    epochs = 100
    train_loader, val_loader = load_eurosat_data(batch_size=64)

    model = BaselineVGG(num_classes=10)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Step 1: 转换模型 (conv1_1 自动 FP32)
    print(f"\n[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声, 首层 conv1_1 FP32)")
    prepare_model_v4(model, weight_bits=8, act_bits=8,
                    noise=True, first_conv_fp32=True,
                    quantize_linear=True, preserve_bn=True)

    # 变体 B: 额外把 conv3_2 保持 FP32
    if variant == "B":
        print(f"\n[Step 1b] 变体 B: conv3_2 也保持 FP32 (电计算)")
        keep_conv_fp32(model, "conv3_2")

    alignment = print_alignment_detail(model, f"Baseline VGG (v4, 变体 {variant})")

    # Step 2: 训练
    base_lr = 0.001
    warmup = 5
    weight_decay = 5e-4
    label_smoothing = 0.05

    print(f"\n[Step 2] 训练 ({epochs} epochs, lr={base_lr}, wd={weight_decay}, "
          f"label_smoothing={label_smoothing})")
    print(f"  int8 权重 (硬件原生精度) | GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4)")
    print(f"  光计算层 (int8 QAT): conv1_2/conv2_1/conv2_2/conv3_1"
          + ("/conv3_2" if variant == "A" else "")
          + " + fc1/fc2")
    print(f"  电计算层 (FP32): conv1_1" + (" + conv3_2" if variant == "B" else ""))

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup,
                                      total_epochs=epochs, base_lr=base_lr)

    best_acc, best_state = 0.0, None
    total_time = 0.0

    print("-" * 70)
    print(f"  {'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'LR':>8s} | {'Time':>7s}")
    print("  " + "-" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

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

        train_loss /= train_total
        train_acc = train_correct / train_total

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

    # 恢复最佳权重
    model.load_state_dict(best_state)

    # Step 3: 最终评估
    print(f"\n[Step 3] 最终评估 (val set)")
    enable_qat(model)
    # 变体 B: enable_qat 后再次确保 conv3_2 关闭
    if variant == "B":
        keep_conv_fp32(model, "conv3_2")
    result_qat = evaluate_model_v4(model, val_loader, DEVICE, criterion)
    print(f"  Int8 模式 (光计算模拟) 准确率: {result_qat['accuracy']:.2%}")

    disable_qat(model)
    result_fp32 = evaluate_model_v4(model, val_loader, DEVICE, criterion)
    print(f"  Float32 模式准确率:          {result_fp32['accuracy']:.2%}")
    print(f"  Int8 量化损失:               "
          f"{result_fp32['accuracy'] - result_qat['accuracy']:.2%}")

    fname = ("weights/baseline_vgg_phase4_v3_int8.pth" if variant == "A"
             else "weights/baseline_vgg_phase4_v3_int8_vB.pth")
    torch.save(model.state_dict(), fname)
    print(f"\n  模型已保存: {fname}")

    print(f"\n{'='*60}")
    print(f"  训练完成 — 结果汇总 (变体 {variant})")
    print(f"{'='*60}")
    print(f"  模型:              Baseline VGG (flat+BN, bias=False)")
    print(f"  参数量:            {sum(p.numel() for p in model.parameters()):,}")
    print(f"  权重量化:          int8 (硬件原生 8-bit)")
    print(f"  噪声模型:          Gazelle (DAC 7.5 + TIA)")
    print(f"  电计算层 (FP32):   conv1_1" + (" + conv3_2" if variant == "B" else ""))
    print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  硬件对齐率:        {alignment:.1%}")
    print(f"  Int8 最佳准确率:   {best_acc:.2%}")
    print(f"  Float32 准确率:    {result_fp32['accuracy']:.2%}")
    print(f"  量化损失:          {result_fp32['accuracy']-result_qat['accuracy']:+.2%}")
    print(f"  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%")
    print(f"  推理脚本:          python optic_inference_int8_model1.py --variant {variant}")
    print(f"{'='*60}")

    return best_acc


if __name__ == "__main__":
    main()
