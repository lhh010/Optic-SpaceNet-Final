"""
================================================================================
  Model 4 (MiniVGG-GAP) Phase4 v3: int8 QAT + Gazelle 硬件噪声 (本地 CPU 训练)
================================================================================
  参照 Model 1-3 的 phase4_v3 配方 (model2_spacenet_v1_phase4_v3.py):
    - int8 权重 (匹配 osimulator 原生 8-bit)
    - Gazelle 硬件匹配噪声 (DAC ENOB=7.5 + TIA, 训练时注入)
    - 首层 stem Conv 保留 FP32 (对齐率低, 匹配 osimulator 推理路径 keep_first_conv_electronic)
    - 其余 Conv + Linear → int8 QAT (quantize_linear=True, preserve_bn=True)
    - CE + label_smoothing=0.05, AdamW + WarmupCosine

  本机无 GPU/torchvision, 数据加载用 PIL 复刻 load_eurosat_data
  (ImageFolder 排序 + eurosat_split 单一数据源划分 + 与 torchvision 相同的增广语义)。

  策略 (本地 CPU 时间受限):
    - 从既有 FP32 基线 weights/minivgg_gap.pth (val 96.65%) 微调 (QAT fine-tune),
      而非从零 QAT (从零需 ~100 epochs, CPU 不可行)
    - EPOCHS 可用环境变量 MODEL4_EPOCHS 覆盖 (默认 15, ~40-60 min @16 核)

  ★ stem 对齐策略 (Model 4 特有决策, 不能盲抄 Model 2/3):
    - Model 2/3 的 stem 是 1×1 (patch=3→8, 对齐率 37.5%) → 保留 FP32 电计算是明确最优。
    - Model 4 的 stem 是 3×3 (patch=27→32, 对齐率 84.4%) → 情况不同:
        * MODEL4_FIRST_CONV_FP32=1 (默认): stem 走 FP32 电计算, 与 M2/M3 部署
          原则一致 (训练/推理完全对齐, 已验证路径), 代价是放弃 stem 的 84.4% 光利用率。
        * MODEL4_FIRST_CONV_FP32=0: stem 也走 int8 QAT (光计算), 可提升光计算占比,
          但需重新验证 osimulator 上 stem 量化行为 (未验证路径, 慎用)。
    - 无论哪种选择, 训练配置必须与 osimulator 推理路径 (build_optical_model 的
      keep_first_conv_electronic 参数) 保持一致, 否则重蹈 §16 的 BN 偏移教训。

  用法:
    python src/training/model4_minivgg_gap_phase4_v3.py
    MODEL4_EPOCHS=10 python src/training/model4_minivgg_gap_phase4_v3.py
    MODEL4_FIRST_CONV_FP32=0 python src/training/model4_minivgg_gap_phase4_v3.py

  产出权重: weights/minivgg_gap_phase4_v3_int8.pth (QAT 层参数字典, 键与原生 MiniVGG 兼容)

  TODO (v4.1 修复后, 见 docs/TODO.md §v4.1 重跑清单):
    - [ ] 用修复后的 optic_qat_v4 (激活噪声 + uint8+zp 激活量化) 完整重训,
          旧权重 (2026-08-07, 5 epochs) 是旧语义训练
    - [ ] 容器内 osimulator 闭环验证: optic_inference_model4.py --mode qat/optic
          (此前 QAT 权重从未在真机评估过)
    - [ ] (可选) MODEL4_FIRST_CONV_FP32=0 消融: 3×3 stem 对齐率 84.4% 显著优于
          M2/M3 的 1×1 stem (37.5%), 是否值得走光计算需实测对比
================================================================================
"""

import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from optic_qat_v4 import (  # noqa: E402
    prepare_model_v4, enable_qat, disable_qat,
    evaluate_model_v4, print_alignment_detail,
)
from train_phase4_runner import WarmupCosineScheduler  # noqa: E402
from eurosat_split import split_indices  # noqa: E402

# ============================================================
#  全局配置 (对齐 model2 phase4_v3)
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
EPOCHS = int(os.environ.get("MODEL4_EPOCHS", "15"))
LEARNING_RATE = 0.001
WARMUP = 5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
VAL_SPLIT = 0.2
SEED = 42
NUM_CLASSES = 10
FP32_BASELINE = "weights/minivgg_gap.pth"
SAVE_PATH = "weights/minivgg_gap_phase4_v3_int8.pth"

# ★ Model 4 stem 对齐策略 (3×3, patch=27→32, 84.4% — 与 M2/M3 的 1×1 stem 37.5% 不同):
#   True  (默认): stem 保留 FP32 电计算 (与 M2/M3 部署原则一致, 已验证路径)
#   False        : stem 也走 int8 QAT 光计算 (需配套 osimulator 推理 keep_first_conv_electronic=False)
FIRST_CONV_FP32 = os.environ.get("MODEL4_FIRST_CONV_FP32", "1").strip().lower() not in ("0", "false", "no")

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(16)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASSES = sorted(["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                  "Industrial", "Pasture", "PermanentCrop", "Residential",
                  "River", "SeaLake"])


# ============================================================
#  模型定义: MiniVGG + GAP (与 model4_minivgg_gap.py 完全一致)
# ============================================================
class MiniVGG(nn.Module):
    """结构: stem(3→32,3×3,s2)+pool → stage1(32→48→48)+pool →
    stage2(48→72→72)+pool → stage3(72→96→96) → GAP → Linear(96,10)
    参数量 ~260K; head Linear 带 bias (原生定义)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, num_classes),
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
        return self.head(x)


# ============================================================
#  PIL 数据加载 (复刻 torchvision ImageFolder + transforms 语义)
# ============================================================
class EuroSATFolder(Dataset):
    """ImageFolder 顺序 (class 排序 × 文件名排序) + 可选训练增广 (PIL)。"""

    def __init__(self, data_dir, indices, train=False):
        self.samples = []
        for ci, cls in enumerate(CLASSES):
            d = os.path.join(data_dir, cls)
            for name in sorted(os.listdir(d)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(d, name), ci))
        self.idx = indices
        self.train = train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        path, cls = self.samples[self.idx[i]]
        with Image.open(path) as im:
            img = im.convert("RGB")
            if self.train:
                if np.random.rand() < 0.5:      # RandomHorizontalFlip p=0.5
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                ang = np.random.uniform(-10, 10)  # RandomRotation(10)
                img = img.rotate(ang, resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)          # (3,64,64)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - mean) / std, cls


def load_eurosat_data_pil(data_dir, batch_size, val_split=0.2, seed=42):
    n = 27000
    train_idx, val_idx, _ = split_indices(n, seed=seed,
                                          val_ratio=val_split, test_ratio=val_split)
    train_ds = EuroSATFolder(data_dir, train_idx, train=True)
    val_ds = EuroSATFolder(data_dir, val_idx, train=False)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)} (eurosat_split, test 留出)")
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0))


# ============================================================
#  训练 (参照 model2 phase4_v3: QAT 开启训练, Gazelle 噪声只在训练注入)
# ============================================================
def main():
    device = torch.device("cpu")
    print(f"设备: {device} | torch {torch.__version__} | threads {torch.get_num_threads()}")
    print(f"{'='*66}")
    stem_label = "FP32 电计算" if FIRST_CONV_FP32 else "int8 QAT 光计算"
    print(f"  Model 4 MiniVGG-GAP Phase4 v3: int8 QAT + Gazelle 噪声 (本地 CPU)")
    print(f"  ★ stem 策略: {stem_label} (3×3 stem patch=27→32, 对齐率 84.4% — "
          f"vs M2/M3 1×1 stem 37.5%)")
    print(f"  其余 Conv+Linear int8 (激活 uint8+zp 对齐 osimulator)")
    print(f"  策略: 从 FP32 基线 {os.path.basename(FP32_BASELINE)} 微调 {EPOCHS} epochs")
    print(f"{'='*66}")

    # ---- 数据 ----
    train_loader, val_loader = load_eurosat_data_pil(
        DATA_DIR, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED)

    # ---- 模型: 先加载 FP32 基线, 再转 QAT v4 ----
    model = MiniVGG(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {n_params:,}")
    sd = torch.load(FP32_BASELINE, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] FP32 基线 {FP32_BASELINE}: missing={len(missing)} unexpected={len(unexpected)}")

    print(f"\n[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声, "
          f"stem {'FP32' if FIRST_CONV_FP32 else 'int8 QAT'})")
    prepare_model_v4(model, weight_bits=8, act_bits=8, noise=True,
                     first_conv_fp32=FIRST_CONV_FP32,
                     quantize_linear=True, preserve_bn=True)
    alignment = print_alignment_detail(model, "MiniVGG-GAP (v4)")

    # ---- 优化器 ----
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP,
                                      total_epochs=EPOCHS, base_lr=LEARNING_RATE)

    print(f"\n[Step 2] 微调 ({EPOCHS} epochs, lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, "
          f"label_smoothing={LABEL_SMOOTHING})")
    print(f"  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4, ADC_lsb=0.0015) — "
          f"训练时注入 权重DAC + 激活TIA/ADC 噪声")
    print("-" * 78)
    print(f"  {'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
          f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'LR':>8s} | {'Time':>7s}")
    print("-" * 76)

    best_acc, best_state = 0.0, None
    total_time = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        val_result = evaluate_model_v4(model, val_loader, device, criterion)
        val_loss, val_acc = val_result["loss"], val_result["accuracy"]
        elapsed = time.time() - t0
        total_time += elapsed
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append((epoch, train_loss / total, correct / total, val_acc))
        print(f"  {epoch:>5d}  | {train_loss/total:>10.4f} {correct/total:>8.2%} | "
              f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | "
              f"{current_lr:>7.5f} | {elapsed:>6.1f}s", flush=True)

    # ---- 恢复最佳 ----
    model.load_state_dict(best_state)

    # ---- 最终评估 (int8 QAT vs fp32) ----
    print(f"\n[Step 3] 最终评估 (最佳 val {best_acc:.2%})")
    enable_qat(model)
    r_qat = evaluate_model_v4(model, val_loader, device, criterion)
    print(f"  Int8 模式 (光计算模拟) val 准确率: {r_qat['accuracy']:.2%}")
    disable_qat(model)
    r_fp32 = evaluate_model_v4(model, val_loader, device, criterion)
    print(f"  Float32 模式 val 准确率:        {r_fp32['accuracy']:.2%}")
    print(f"  Int8 量化损失:                 {r_fp32['accuracy'] - r_qat['accuracy']:.2%}")

    # ---- 独立 test 集评估 (int8) ----
    try:
        n = 27000
        _, _, test_idx = split_indices(n, seed=SEED,
                                       val_ratio=VAL_SPLIT, test_ratio=VAL_SPLIT)
        test_ds = EuroSATFolder(DATA_DIR, test_idx, train=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)
        enable_qat(model)
        r_test = evaluate_model_v4(model, test_loader, device, criterion)
        print(f"  Int8 模式 test 准确率 (独立 {len(test_idx)} 张): "
              f"{r_test['accuracy']:.2%}")
    except Exception as e:
        print(f"  [skip] test 集评估失败: {e}")

    # ---- 保存 ----
    torch.save(best_state, SAVE_PATH)
    print(f"\n  模型已保存: {SAVE_PATH}")

    print(f"\n{'='*66}")
    print(f"  训练完成 — 结果汇总")
    print(f"{'='*66}")
    print(f"  模型:           MiniVGG-GAP (Phase4 v3, int8 + Gazelle 噪声)")
    print(f"  参数量:         {n_params:,}")
    print(f"  首层 stem:      {'FP32 电计算' if FIRST_CONV_FP32 else 'int8 QAT 光计算'} "
          f"(3×3, patch=27→32, 对齐率 84.4%)")
    print(f"  其余:           Conv×7 + Linear×1 → int8 QAT (激活 uint8+zp 对齐 osimulator)")
    print(f"  噪声:           训练时 权重DAC + 激活TIA/ADC (修复注入缺失)")
    print(f"  硬件对齐率:     {alignment:.1%}")
    print(f"  训练总耗时:     {total_time:.1f}s ({total_time/60:.1f}min, {EPOCHS} epochs)")
    print(f"  Int8 最佳 val:  {best_acc:.2%}")
    print(f"  Int8 最终 val:  {r_qat['accuracy']:.2%}")
    print(f"  Float32 val:    {r_fp32['accuracy']:.2%}")
    print(f"  FP32 基线参考:  96.65% (minivgg_gap.pth, 原生 FP32)")
    print(f"  权重:           {SAVE_PATH}")
    print(f"{'='*66}")
    return best_acc


if __name__ == "__main__":
    main()
