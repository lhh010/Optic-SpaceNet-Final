"""
================================================================================
 模型一 Mixed (Baseline VGG): Conv=int4 (光计算) + Linear=fp32 (电计算)
================================================================================
 量化策略:
   - 所有 Conv2d:  int4 QAT (STE + 噪声注入), bias=False
   - 所有 Linear:  float32 常规训练, bias=True
   - Pool/BN/ReLU: float32 原生 (BN 稳定训练)

 与全部 int4 方案 (model1_baseline_phase4.py) 对比:
   - 全部 int4: Conv + Linear 都量化为 int4, 精度损失大
   - 混合方案: Linear 保留 fp32, 仅 Conv 做 int4
   - 光计算加速 Conv (占总 MACs 98.7%), Linear 在电域高精度计算

 用法:
   python model1_baseline_mixed.py                  # STE 模式 (推荐)
   python model1_baseline_mixed.py --mode lsqplus   # LSQ+ 模式
   python model1_baseline_mixed.py --act-bits 4     # 激活也 int4 (激进)
================================================================================
"""

import torch
import torch.nn as nn
import sys
import numpy as np

from train_mixed_runner import MixedPrecisionTrainer, load_eurosat_data


# ============================================================
#  模型: BaselineVGG — Conv(int4, bias=False) + Linear(fp32, bias=True)
# ============================================================

class BaselineVGG(nn.Module):
    """Mini-VGG: Conv → 光计算 int4, Linear → 电计算 fp32"""

    def __init__(self, num_classes=10):
        super().__init__()

        # === 卷积部分 (光计算 int4, bias=False) ===
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # === 全连接部分 (电计算 fp32, bias=True) ===
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=True)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

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
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
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
    print(f"  Model 1 Mixed: Conv=int4 (光计算) + Linear=fp32 (电计算)")
    print(f"{'='*60}")
    train_loader, val_loader = load_eurosat_data(batch_size=64)

    model = BaselineVGG(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {param_count:,}")
    print(f"  6×Conv → int4 QAT, 2×Linear → fp32")

    config = {
        'mode': mode,
        'weight_bits': 4,
        'act_bits': act_bits,
        'noise': (mode == 'ste'),
        'noise_std_ratio': 0.02,
        'epochs': 80,
        'learning_rate': 0.001,
        'warmup_epochs': 5,
        'weight_decay': 5e-4 if mode == 'ste' else 1e-4,
        'label_smoothing': 0.0,
        'model_name': f'BaselineVGG Mixed (Conv=int4, Linear=fp32, {mode})',
        'save_path': f'baseline_vgg_mixed_{mode}.pth',
        'fp32_baseline': '97.17% (全 fp32)',
        'device': DEVICE,
        'num_classes': 10,
    }

    trainer = MixedPrecisionTrainer(model, train_loader, val_loader, config)
    best_acc = trainer.run()
    return best_acc


if __name__ == "__main__":
    main()
