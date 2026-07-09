"""
================================================================================
 模型二 Phase4 v2 (SpaceNet V1): Conv+Linear 全 int4 QAT + STE + 噪声注入
================================================================================
 修复 (vs 旧版 model2_spacenet_v1_phase4.py):
   - 旧版: optic_qat_v2 + first_layer_fp32 bug → Conv 层 QAT 全关 → int4=74%
   - 新版: optic_qat_v3 + Phase4Trainer → Conv+Linear 全 int4 QAT
   - 预期 int4 精度: 85-90% (大幅优于旧版 74%)

 量化策略:
   - 所有 Conv2d: int4 QAT (STE + 噪声注入), bias=False
   - 所有 Linear:  int4 QAT (STE + 噪声注入), bias=False
   - BN: float32 保留

 用法:
   python model2_spacenet_v1_phase4_v2.py
================================================================================
"""

import torch
import torch.nn as nn
import sys
import numpy as np

from train_phase4_runner import Phase4Trainer, load_eurosat_data


# ============================================================
#  模型: SpaceNet V1 — 全 int4, bias=False (匹配光计算硬件)
# ============================================================

class OpticSpaceNetV1(nn.Module):
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
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"设备: {DEVICE}")
    print(f"\n{'='*60}")
    print(f"  Model 2 Phase4 v2: Conv+Linear 全 int4, bias=False")
    print(f"  修复: Conv 层 QAT 全开 (vs 旧版全关)")
    print(f"{'='*60}")

    train_loader, val_loader = load_eurosat_data(batch_size=64)

    model = OpticSpaceNetV1(num_classes=10)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  4×Conv + 2×Linear → 全 int4 QAT, bias=False")

    config = {
        'mode': 'ste',
        'weight_bits': 4,
        'act_bits': 8,
        'noise': True,
        'noise_std_ratio': 0.02,
        'quantize_linear': True,   # ★ 关键修复: Linear 也做 int4 (Phase4 设计)
        'epochs': 100,
        'learning_rate': 0.001,
        'warmup_epochs': 5,
        'weight_decay': 5e-4,
        'label_smoothing': 0.0,
        'model_name': 'OpticSpaceNetV1 Phase4 v2 (Conv+Linear int4, bias=False)',
        'save_path': 'spacenet_v1_phase4_v2_ste.pth',
        'fp32_baseline': '90.15% (全 fp32)',
        'device': DEVICE,
        'num_classes': 10,
    }

    trainer = Phase4Trainer(model, train_loader, val_loader, config)
    best_acc = trainer.run()
    return best_acc


if __name__ == "__main__":
    main()
