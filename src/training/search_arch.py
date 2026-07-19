"""
并行架构搜索: 5 个 channel 缩比候选 vs EuroSAT

每个候选共享同一架构模板 (stem + 3 stages × 2 convs + GAP),
仅 channel 配置不同。并行跑在容器 GPU 上。

用法:
  /local/miniconda/envs/moca_llm/bin/python src/training/search_arch.py --gpu 0 --candidate A
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
import json

from train_phase4_runner import load_eurosat_data

DATA_DIR = "data/EuroSAT_RGB"
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42

# 架构候选: {name: (C0, C1, C2, C3)}
CANDIDATES = {
    "A": (8, 12, 18, 24),
    "B": (12, 20, 30, 42),
    "C": (18, 28, 42, 56),
    "D": (24, 36, 54, 72),
    "E": (32, 48, 72, 96),
}

MAC_ESTIMATES = {
    "A": 1.23, "B": 3.11, "C": 5.95, "D": 9.75, "E": 17.03,
}


class MiniVGGArch(nn.Module):
    """参数化 MiniVGG: 可变 channel 配置"""

    def __init__(self, C0, C1, C2, C3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(C0, C1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C1), nn.ReLU(inplace=True),
            nn.Conv2d(C1, C1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(C1, C2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C2), nn.ReLU(inplace=True),
            nn.Conv2d(C2, C2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(C2, C3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3), nn.ReLU(inplace=True),
            nn.Conv2d(C3, C3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C3, num_classes),
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
        x = self.head(x)
        return x


def compute_macs(model):
    """计算单图 MACs"""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
    return total


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--candidate", type=str, required=True, choices=list(CANDIDATES.keys()),
                        help="Candidate to train")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    C0, C1, C2, C3 = CANDIDATES[args.candidate]
    name = args.candidate

    print(f"[{name}] GPU: {device}, Channels: ({C0},{C1},{C2},{C3})")

    train_loader, val_loader = load_eurosat_data(
        DATA_DIR, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED)

    model = MiniVGGArch(C0, C1, C2, C3, num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    est_macs = compute_macs(model)
    print(f"[{name}] Params: {n_params:,}  Est MACs: {est_macs:,} ({est_macs/1e6:.2f}M)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    log_lines = []

    header = f"[{name}] {'Epoch':>4s} {'TrLoss':>8s} {'TrAcc':>7s} {'VaLoss':>8s} {'ValAcc':>7s} {'Best':>7s} {'Time':>6s}"
    print(header)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 10 == 0 or epoch == 1:
            line = f"[{name}] {epoch:>4d} {train_loss:>8.4f} {train_acc:>6.2%} {val_loss:>8.4f} {val_acc:>6.2%} {best_acc:>6.2%} {elapsed:>5.1f}s"
            print(line)
        log_lines.append(f"{epoch},{train_loss:.4f},{train_acc:.6f},{val_loss:.4f},{val_acc:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    _, final_val_acc = evaluate(model, val_loader, criterion, device)

    save_path = f"weights/search_{name}.pth"
    torch.save(model.state_dict(), save_path)

    result = {
        "candidate": name,
        "channels": [C0, C1, C2, C3],
        "params": n_params,
        "est_macs": est_macs,
        "est_macs_M": round(est_macs / 1e6, 2),
        "best_val_acc": round(best_acc * 100, 2),
        "final_val_acc": round(final_val_acc * 100, 2),
        "best_epoch": best_epoch,
        "weight_path": save_path,
        "device": str(device),
    }

    # 保存 JSON
    with open(f"weights/search_{name}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{name}] 完成! Best: {best_acc:.2%} (epoch {best_epoch}) -> {save_path}")
    return result


if __name__ == "__main__":
    main()
