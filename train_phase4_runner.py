"""
================================================================================
 train_phase4_runner.py — Phase 4 训练公共模块

 为 3 个模型提供统一的训练管线:
   - Warmup + CosineAnnealing 学习率调度
   - 混合精度 QAT (首层/末层 fp32, 中间 int4)
   - 可配置的量化参数 (weight_bits, act_bits, noise)
   - 统一的日志输出和模型保存

 用法:
   from train_phase4_runner import Phase4Trainer
   trainer = Phase4Trainer(model, train_loader, val_loader, config)
   best_acc = trainer.run()
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy

from optic_qat_v3 import (
    prepare_model_v3, enable_qat, disable_qat,
    evaluate_model_v3, compare_qat_vs_float,
    compute_alignment_ratio, print_alignment_detail,
    set_quant_lr,
)


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 学习率调度"""

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
            # LSQ+ 量化参数使用独立 lr (已在 param_groups 中设置)
            if param_group.get('is_quant', False):
                param_group['lr'] = lr * 0.1
            else:
                param_group['lr'] = lr

    def _get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            return self.base_lr * self.current_epoch / max(1, self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(1, self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                   (1 + torch.cos(torch.tensor(torch.pi * progress)).item())


class Phase4Trainer:
    """Phase 4 QAT 训练器"""

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model:        标准 PyTorch 模型 (随机初始化)
            train_loader: 训练数据
            val_loader:   验证数据
            config:       dict, 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = config.get('device', torch.device('cpu'))
        self.num_classes = config.get('num_classes', 10)

    def run(self):
        cfg = self.config
        model = self.model

        # === Step 1: 转换模型 ===
        quant_linear = cfg.get('quantize_linear', True)  # Phase4=True, Mixed=False
        strategy = "Conv+Linear→int4 QAT" if quant_linear else "Conv→int4 QAT, Linear→fp32"
        print(f"\n[Step 1] 转换为 QAT v3: {strategy}")
        prepare_model_v3(
            model,
            mode=cfg['mode'],
            weight_bits=cfg.get('weight_bits', 4),
            act_bits=cfg.get('act_bits', 8),
            noise=cfg.get('noise', True),
            noise_std_ratio=cfg.get('noise_std_ratio', 0.02),
            quantize_linear=quant_linear,
            preserve_bn=True,
        )

        # 统计
        qc = sum(1 for m in model.modules()
                 if type(m).__name__ == 'QATConv2d_v3' and m.qat_enabled)
        ql = sum(1 for m in model.modules()
                 if type(m).__name__ == 'QATLinear_v3' and m.qat_enabled)
        lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        print(f"  int4 QAT Conv: {qc}, int4 QAT Linear: {ql}, "
              f"fp32 Linear: {lin - ql}, BN: {bn} (float32)")

        # 硬件对齐率
        alignment = print_alignment_detail(model, cfg.get('model_name', 'Model'))

        # === Step 2: 训练 ===
        epochs = cfg.get('epochs', 60)
        base_lr = cfg.get('learning_rate', 0.001)
        warmup = cfg.get('warmup_epochs', 5)
        weight_decay = cfg.get('weight_decay', 5e-4)
        label_smoothing = cfg.get('label_smoothing', 0.0)

        print(f"\n[Step 2] 训练 ({epochs} epochs, lr={base_lr}, "
              f"warmup={warmup}, wd={weight_decay})")
        if cfg.get('noise', True) and cfg['mode'] == 'ste':
            print(f"  噪声注入: std={cfg.get('noise_std_ratio', 0.02)}*scale "
                  f"(仅 int4 权重层)")

        model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 优化器
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

            # 训练 + 验证
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

        # 加载最佳权重
        model.load_state_dict(best_state)

        # === Step 3: 最终评估 ===
        print(f"\n[Step 3] 最终评估")

        # Int4 模式
        enable_qat(model)
        result_qat = evaluate_model_v3(model, self.val_loader, self.device, criterion)
        print(f"  Int4 模式 (光计算模拟) 准确率: {result_qat['accuracy']:.2%}")

        # Float32 模式
        disable_qat(model)
        result_fp32 = evaluate_model_v3(model, self.val_loader, self.device, criterion)
        print(f"  Float32 模式准确率:         {result_fp32['accuracy']:.2%}")
        print(f"  Int4 量化损失:              "
              f"{result_fp32['accuracy'] - result_qat['accuracy']:.2%}")

        # === Step 4: 保存 ===
        fname = cfg.get('save_path', 'model_phase4.pth')
        torch.save(model.state_dict(), fname)
        print(f"\n  模型已保存: {fname}")

        # === 结果汇总 ===
        strategy_desc = ("Conv+Linear→int4 (全光计算)" if quant_linear
                        else "Conv→int4 (光计算) + Linear→fp32 (电计算)")
        print(f"\n{'='*60}")
        print(f"  训练完成 — 结果汇总")
        print(f"{'='*60}")
        print(f"  模型:              {cfg.get('model_name', 'Unknown')}")
        print(f"  参数量:            {sum(p.numel() for p in model.parameters()):,}")
        print(f"  量化策略:          {strategy_desc}")
        print(f"  模式:              {cfg['mode']}, "
              f"w{cfg.get('weight_bits', 4)}/a{cfg.get('act_bits', 8)}")
        print(f"  训练总耗时:        {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  硬件对齐率:        {alignment:.1%}")
        print(f"  Int4 最佳准确率:   {best_acc:.2%}")
        print(f"  Float32 准确率:    {result_fp32['accuracy']:.2%}")
        print(f"  FP32 基准 (参考):  {cfg.get('fp32_baseline', 'N/A')}")
        print(f"  量化损失:          {result_fp32['accuracy'] - result_qat['accuracy']:.2%}")
        print(f"{'='*60}")

        return best_acc

    def _train_epoch(self, model, criterion, optimizer):
        """训练一个 epoch — QAT 层自动施加伪量化"""
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
#  通用数据加载 (EuroSAT)
# ============================================================

def load_eurosat_data(data_dir="data/EuroSAT_RGB", batch_size=64,
                      val_split=0.2, seed=42):
    """加载 EuroSAT RGB 数据集"""
    import numpy as np
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

    n = len(train_full)
    val_size = int(n * val_split)
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
    val_dataset = torch.utils.data.Subset(val_full, indices[:val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
    print(f"类别: {train_full.classes}")
    return train_loader, val_loader


# ============================================================
#  自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  train_phase4_runner.py — Self-Test")
    print("=" * 60)

    # 创建一个简单模型和假数据来验证训练管线
    import numpy as np

    class TinyTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 10)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    # 假数据
    from torch.utils.data import DataLoader, TensorDataset
    X = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    train_ds = TensorDataset(X, y)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(train_ds, batch_size=16, shuffle=False)

    # 配置
    config = {
        'mode': 'ste',
        'weight_bits': 4,
        'act_bits': 8,
        'noise': True,
        'noise_std_ratio': 0.02,
        'epochs': 3,
        'learning_rate': 0.001,
        'warmup_epochs': 1,
        'weight_decay': 5e-4,
        'label_smoothing': 0.0,
        'model_name': 'TinyTest (Conv=int4, Linear=fp32)',
        'save_path': 'tiny_test.pth',
        'fp32_baseline': 'N/A',
        'device': torch.device('cpu'),
        'num_classes': 10,
    }

    model = TinyTestModel()
    trainer = Phase4Trainer(model, train_dl, val_dl, config)
    acc = trainer.run()
    print(f"\n  [OK] Training pipeline works! Best acc: {acc:.2%}")
