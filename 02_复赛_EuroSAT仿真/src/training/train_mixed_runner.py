"""
================================================================================
 train_mixed_runner.py — Conv=int4 / Linear=fp32 混合精度训练公共模块

 量化策略:
   - Conv 层: int4 QAT (光计算) — 全部启用, 含首层
   - Linear 层: float32 (电计算) — 不包装 QAT, 可带 bias
   - Pool/BN/ReLU/Dropout: float32 原生

 提供:
   - WarmupCosineScheduler: Warmup + CosineAnnealing LR 调度
   - MixedPrecisionTrainer: 混合精度 QAT 训练器
   - KDPhase4Trainer:      KD + 混合精度联合训练器
   - load_eurosat_data():   EuroSAT 数据加载

 依赖:
   optic_qat_v3.py (Conv=int4 / Linear=fp32 专用 QAT 模块)
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np

from optic_qat_v3 import (
    prepare_model_v3, enable_qat, disable_qat,
    evaluate_model_v3, compute_alignment_ratio,
    print_alignment_detail, set_quant_lr,
)


# ============================================================
#  学习率调度
# ============================================================

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing"""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            if param_group.get('is_quant', False):
                param_group['lr'] = lr * 0.1
            else:
                param_group['lr'] = lr

    def _get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            return self.base_lr * self.current_epoch / max(1, self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(1, self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                   (1 + torch.cos(torch.tensor(torch.pi * progress)).item())


# ============================================================
#  数据加载
# ============================================================

def load_eurosat_data(data_dir="data/EuroSAT_RGB", batch_size=64,
                      val_split=0.2, seed=42):
    """加载 EuroSAT RGB 数据集"""
    from torchvision import datasets, transforms

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

    train_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_full = datasets.ImageFolder(data_dir, transform=val_transform)

    # 三分划分 — 单一数据源 eurosat_split (与所有推理 load_test_data 完全一致)。
    # test 段剔除不参与训练, 杜绝 test<train 泄漏 (Bug #11)。
    from eurosat_split import split_indices
    n = len(train_full)
    train_idx, val_idx, _ = split_indices(n, seed=seed,
                                          val_ratio=val_split, test_ratio=val_split)
    train_dataset = torch.utils.data.Subset(train_full, train_idx)
    val_dataset = torch.utils.data.Subset(val_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 留出测试: {len(val_dataset)} (见 eurosat_split)")
    print(f"类别: {train_full.classes}")
    return train_loader, val_loader


# ============================================================
#  混合精度训练器 (Conv=int4, Linear=fp32)
# ============================================================

class MixedPrecisionTrainer:
    """Conv=int4 QAT + Linear=fp32 训练器"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get('device', torch.device('cpu'))

    def run(self):
        cfg = self.config
        model = self.model

        # === Step 1: 转换 (Conv→int4 QAT, Linear→fp32) ===
        print(f"\n[Step 1] 转换为混合精度 QAT: Conv=int4, Linear=fp32")
        prepare_model_v3(
            model,
            mode=cfg['mode'],
            weight_bits=cfg.get('weight_bits', 4),
            act_bits=cfg.get('act_bits', 8),
            noise=cfg.get('noise', True),
            noise_std_ratio=cfg.get('noise_std_ratio', 0.02),
            quantize_linear=False,
            preserve_bn=True,
        )

        qc = sum(1 for m in model.modules()
                 if type(m).__name__ == 'QATConv2d_v3' and m.qat_enabled)
        lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        print(f"  int4 QAT Conv: {qc}, fp32 Linear: {lin}, BN(float32): {bn}")

        alignment = print_alignment_detail(model, cfg.get('model_name', 'Model'))

        # === Step 2: 训练 ===
        epochs = cfg.get('epochs', 60)
        base_lr = cfg.get('learning_rate', 0.001)
        warmup = cfg.get('warmup_epochs', 5)
        weight_decay = cfg.get('weight_decay', 5e-4)
        label_smoothing = cfg.get('label_smoothing', 0.0)

        print(f"\n[Step 2] 训练 ({epochs} epochs, lr={base_lr}, "
              f"warmup={warmup}, wd={weight_decay})")
        if cfg.get('noise', True):
            print(f"  噪声注入: std={cfg.get('noise_std_ratio', 0.02)}*scale "
                  f"(仅 int4 Conv 权重)")

        model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if cfg['mode'] == 'lsqplus':
            param_groups = set_quant_lr(model, base_lr=base_lr)
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=base_lr,
                                    weight_decay=weight_decay)

        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup,
            total_epochs=epochs, base_lr=base_lr
        )

        best_acc, best_state = 0.0, None
        total_time = 0.0

        print("-" * 70)
        print(f"  {'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | "
              f"{'Val Loss':>9s} {'Val Acc':>8s} | {'Best':>8s} | {'LR':>8s} | {'Time':>7s}")
        print("  " + "-" * 68)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(model, criterion, optimizer)
            val_result = evaluate_model_v3(model, self.val_loader, self.device, criterion)
            val_loss, val_acc = val_result['loss'], val_result['accuracy']
            elapsed = time.time() - t0
            total_time += elapsed
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>5d}  | {train_loss:>10.4f} {train_acc:>8.2%} | "
                      f"{val_loss:>9.4f} {val_acc:>7.2%} | {best_acc:>7.2%} | "
                      f"{current_lr:>7.5f} | {elapsed:>6.1f}s")

        model.load_state_dict(best_state)

        # === Step 3: 评估 ===
        print(f"\n[Step 3] 最终评估")
        enable_qat(model)
        result_qat = evaluate_model_v3(model, self.val_loader, self.device, criterion)
        print(f"  Int4 模式 (Conv=int4 光计算) 准确率: {result_qat['accuracy']:.2%}")

        disable_qat(model)
        result_fp32 = evaluate_model_v3(model, self.val_loader, self.device, criterion)
        print(f"  Float32 模式准确率:                {result_fp32['accuracy']:.2%}")
        print(f"  Int4 量化损失:                     "
              f"{result_fp32['accuracy'] - result_qat['accuracy']:.2%}")

        fname = cfg.get('save_path', 'model_mixed.pth')
        torch.save(model.state_dict(), fname)
        print(f"\n  模型已保存: {fname}")

        print(f"\n{'='*60}")
        print(f"  训练完成 — 结果汇总")
        print(f"{'='*60}")
        print(f"  策略:              Conv=int4 (光计算) + Linear=fp32 (电计算)")
        print(f"  模型:              {cfg.get('model_name', 'Unknown')}")
        print(f"  参数量:            {sum(p.numel() for p in model.parameters()):,}")
        print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  硬件对齐率:        {alignment:.1%}")
        print(f"  Int4 最佳准确率:   {best_acc:.2%}")
        print(f"  Float32 准确率:    {result_fp32['accuracy']:.2%}")
        print(f"  FP32 基准 (参考):  {cfg.get('fp32_baseline', 'N/A')}")
        print(f"{'='*60}")

        return best_acc

    def _train_epoch(self, model, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
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
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  train_mixed_runner.py — Self-Test")
    print("=" * 60)

    from torch.utils.data import DataLoader, TensorDataset

    class TinyTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 10, bias=True)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    X = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    train_dl = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    val_dl = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    config = {
        'mode': 'ste', 'weight_bits': 4, 'act_bits': 8,
        'noise': True, 'noise_std_ratio': 0.02,
        'epochs': 3, 'learning_rate': 0.001, 'warmup_epochs': 1,
        'weight_decay': 5e-4, 'label_smoothing': 0.0,
        'model_name': 'TinyTest (Conv=int4, Linear=fp32)',
        'save_path': 'tiny_test.pth', 'fp32_baseline': 'N/A',
        'device': torch.device('cpu'), 'num_classes': 10,
    }

    model = TinyTestModel()
    trainer = MixedPrecisionTrainer(model, train_dl, val_dl, config)
    acc = trainer.run()
    print(f"\n  [OK] Mixed precision pipeline works! Best acc: {acc:.2%}")

    import os
    os.remove('tiny_test.pth')
