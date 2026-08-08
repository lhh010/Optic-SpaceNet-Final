#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
 Model 6 — J1 + head256 · Gazelle 光计算 QAT 训练脚本 (≤2M 严格预算档)
===============================================================================
 设计依据:
   - 架构: R3 J1 冠军 (1.38M MACs / 50.3K params / clean 96.30, 160ep)
     + head256 (R7: head 为 FP32 电计算, 宽度免费, r7_final 配方)
   - R7 严格 ≤2M 结论: J1 配平是最优 (C0 瓶颈, 深度 (1,2,2) 恰好, 宽度/池化/旁路
     探索全部不赚) → Model 6 = J1 + head256, 不越预算
   - QAT: v5 干净阶段 → v8 漂移鲁棒阶段 (与 c3d 同架构同 probe 参数, 上板预期 ≈93.8+)

 详细日志: 每 epoch 打印 + runs/<name>_<phase>/metrics.jsonl (结构化)
 用法:
   python train_m6_j1_head256.py                       # 干净阶段 160ep
   python train_m6_j1_head256.py --phase v8 --init <clean best.pth>
   python train_m6_j1_head256.py --epochs 2            # 冒烟测试
===============================================================================
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, _HERE)

from models import build_model, compute_macs
from data import load_eurosat
from metrics import evaluate_full

DATA_DIR = "E:/LT-Simulator/train-test/data/EuroSAT_RGB"
SEED = 42
NUM_CLASSES = 10
NAME = "m8_j1w075_stem5_head256"

# ---- Model 6 架构 (J1 + head256) ----
ARCH = dict(
    channels=[12, 24, 48, 96],
    stem_stride=2,
    fast_downsample=True,
    kernels=[1, 1, 1],        # 全 1×1 (与 8×2 tile 天然对齐, 展平=通道数)
    stem_kernel=5,
    head_dims=[256],          # head 免费 (FP32 电计算)
    bias=False,
)

# ---- v8 漂移鲁棒噪声参数 (c3d 冠军配方, 与 J1 架构完全一致) ----
V8_NOISE = dict(
    layer_noise_sigmas={p: 260.0 for p in
                        ["stage1.0", "stage2.0", "stage2.3", "stage3.0", "stage3.3"]},
    layer_col_off={"stage1.0": 421.5, "stage2.0": 717.0, "stage2.3": 688.5,
                   "stage3.0": 1308.0, "stage3.3": 396.0},
    layer_col_gain={"stage1.0": 0.03765, "stage2.0": 0.036, "stage2.3": 0.018,
                    "stage3.0": 0.02355, "stage3.3": 0.0189},
    layer_dw_rms={"stage1.0": 5.58, "stage2.0": 9.735, "stage2.3": 7.5,
                  "stage3.0": 7.005, "stage3.3": 10.845},
)


class WarmupCosine:
    def __init__(self, optimizer, warmup, total, base_lr, min_ratio=0.01):
        self.opt, self.warmup, self.total = optimizer, warmup, total
        self.base_lr, self.min_lr = base_lr, base_lr * min_ratio
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        lr = (self.base_lr * e / max(1, self.warmup)) if e <= self.warmup else \
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
            (1 + np.cos(np.pi * (e - self.warmup) / max(1, self.total - self.warmup)))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="clean", choices=["clean", "v8"])
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--init", default=None, help="v8 阶段: 干净阶段 best.pth")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phase, epochs = args.phase, args.epochs
    run_dir = f"runs/{NAME}_{phase}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    save_path = f"weights/{NAME}_{phase}.pth"

    print("=" * 78)
    print(f"  Model 6 (J1+head256): ≤2M 严格预算 | 光计算 QAT | phase={phase}")
    print(f"  链路: v5 干净 → v8 漂移鲁棒 → 容器 osim 验证 → 真机验证")
    print(f"  日志: {jsonl_path}")
    print("=" * 78)

    # ---- 数据 ----
    train_loader, val_loader, test_loader = load_eurosat(
        DATA_DIR, batch_size=args.batch, aug="standard", val_split=0.2,
        seed=SEED, num_workers=args.workers)

    # ---- 模型 ----
    model = build_model(dict(ARCH, num_classes=NUM_CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    macs = compute_macs(model)
    print(f"params={n_params:,} | MACs={macs:,} ({macs/1e6:.2f}M)")

    # ---- QAT 转换 ----
    if phase == "clean":
        from qat_v5 import prepare_model_v5
        prepare_model_v5(model, weight_bits=8, output_noise=True,
                         output_noise_ratio=0.0392, output_quant=True,
                         weight_noise=False, activation_style="osim")
        lr, warmup = args.lr or 0.05, 5
    else:  # v8 漂移鲁棒
        if not args.init:
            sys.exit("[v8] 需要 --init <clean best.pth>")
        from qat_v8 import prepare_model_v8
        prepare_model_v8(model, stem_fp32=True, head_fp32=True,
                         weight_bits=8, output_noise=True, output_quant=True,
                         weight_noise=False, activation_style="osim",
                         **V8_NOISE)
        st = torch.load(args.init, map_location="cpu")
        msd = model.state_dict()
        filt = {k: v for k, v in st.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
        print(f"  init_from {args.init}: {len(filt)}/{len(msd)} tensors")
        lr, warmup = args.lr or 0.01, 2

    # ---- 训练 ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = WarmupCosine(optimizer, warmup, epochs, lr)
    best_acc, best_state, history = 0.0, None, []

    print(f"\n[训练] {epochs} ep | lr={lr} warmup={warmup} | SGD+momentum "
          f"| batch={args.batch} workers={args.workers}")
    print("-" * 78)
    print("  Epoch | Train loss/acc        | Val loss/acc         | "
          "Best     | LR      | Time")
    print("-" * 78)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_c, tr_n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_c += (out.argmax(1) == labels).sum().item()
            tr_n += imgs.size(0)
        tr_loss /= max(1, tr_n)
        tr_acc = tr_c / max(1, tr_n)

        val = evaluate_full(model, val_loader, device)
        v_acc = val["accuracy"]
        cur_lr = scheduler.step()
        if v_acc > best_acc:
            best_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            val_loss=val["loss"], val_acc=v_acc,
                            best_acc=best_acc, lr=cur_lr, time_s=elapsed))
        print(f"  {epoch:>5d} | {tr_loss:.4f} {tr_acc:7.2%}        | "
              f"{val['loss']:.4f} {v_acc:7.2%}         | {best_acc:7.2%} | "
              f"{cur_lr:.5f} | {elapsed:5.0f}s", flush=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(history[-1]) + "\n")

    model.load_state_dict(best_state)

    # ---- 最终评估 (test 独立集) ----
    print("\n[最终评估]")
    r_test = evaluate_full(model, test_loader, device)
    print(f"  test acc   = {r_test['accuracy']:.2%} (n={r_test['n']})")
    print(f"  macro_f1   = {r_test['macro_f1']:.4f}")
    print(f"  ece        = {r_test['ece']:.4f}")
    print(f"  per-class  = "
          f"{' '.join(f'{x:.1%}' for x in r_test['per_class_f1'])}")
    summary = dict(name=NAME, phase=phase, params=n_params, macs=macs,
                   best_val=best_acc, test_acc=r_test["accuracy"],
                   macro_f1=r_test["macro_f1"], ece=r_test["ece"],
                   epochs=epochs, lr=lr, warmup=warmup, seed=SEED)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), save_path)
    print(f"  已保存: {save_path}")
    return r_test["accuracy"]


if __name__ == "__main__":
    main()
