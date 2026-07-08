"""
================================================================================
 模型二 Mixed (Optic-SpaceNet V1): Conv=int4 (光计算) + Linear=fp32 (电计算)
================================================================================
 量化策略:
   - 所有 Conv2d (stem/stage1-3): int4 QAT, bias=False
   - 所有 Linear (classifier):    float32, bias=True
   - Pool/BN/ReLU:                float32 原生

 硬件对齐:
   - 2×2 卷积: patch=32/64, 100% 对齐 8×2 阵列
   - 1×1 卷积: stem patch=3→8 (37.5%), 其余完美对齐
   - 综合对齐率: ~99.6%

 与全部 int4 方案 (model2_spacenet_v1_phase4.py) 对比:
   - 全部 int4: 4 Conv + 2 Linear 全 int4
   - 混合方案: 4 Conv int4 + 2 Linear fp32
   - Conv 占总 MACs ~75%, Linear 保留 fp32 提升分类精度

 用法:
   python model2_spacenet_v1_mixed.py                  # STE 模式
   python model2_spacenet_v1_mixed.py --mode lsqplus   # LSQ+ 模式
================================================================================
"""

import torch
import torch.nn as nn
import sys
import numpy as np

from train_mixed_runner import MixedPrecisionTrainer, load_eurosat_data


# ============================================================
#  模型: OpticSpaceNetV1 — Conv(int4) + Linear(fp32)
# ============================================================

class OpticSpaceNetV1(nn.Module):
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
    print(f"  Model 2 Mixed: Conv=int4 (光计算) + Linear=fp32 (电计算)")
    print(f"{'='*60}")
    train_loader, val_loader = load_eurosat_data(batch_size=64)

    model = OpticSpaceNetV1(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {param_count:,}")
    print(f"  4×Conv → int4 QAT, 2×Linear → fp32")

    config = {
        'mode': mode,
        'weight_bits': 4,
        'act_bits': act_bits,
        'noise': (mode == 'ste'),
        'noise_std_ratio': 0.02,
        'epochs': 100,
        'learning_rate': 0.001,
        'warmup_epochs': 5,
        'weight_decay': 5e-4 if mode == 'ste' else 1e-4,
        'label_smoothing': 0.0,
        'model_name': f'OpticSpaceNetV1 Mixed (Conv=int4, Linear=fp32, {mode})',
        'save_path': f'spacenet_v1_mixed_{mode}.pth',
        'fp32_baseline': '90.15% (全 fp32)',
        'device': DEVICE,
        'num_classes': 10,
    }

    trainer = MixedPrecisionTrainer(model, train_loader, val_loader, config)
    best_acc = trainer.run()
    return best_acc


if __name__ == "__main__":
    main()
