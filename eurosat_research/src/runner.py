"""
===============================================================================
 runner.py — 统一训练/评测入口 (config-driven)
===============================================================================
 用法:
   python runner.py --config configs/xxx.json [--gpu 0]
   python runner.py --config configs/xxx.json --list-only   # 只显示 run_dir

 输出 (runs/<name>_<hash8>/):
   config.json       — 完整配置 (含 hash)
   metrics.jsonl     — 每 epoch 指标 (train/val loss, acc, f1, ece, lr, time)
   summary.json      — 最终评测 (val/test acc, macro_f1, per_class_f1, ece, confusion)
   best.pth          — 最佳 val 权重
   last.pth          — 最后 epoch 权重
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
sys.path.insert(0, _HERE)

from config import load_config, make_run_dir
from models import build_model, compute_macs
from data import load_eurosat
from metrics import evaluate_full
from muon import Muon


def build_optimizer(model, cfg):
    name = cfg["optimizer"]
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                         weight_decay=wd, nesterov=True)
    if name == "muon":
        # Muon: 2D 参数正交化, 其余 AdamW (低 lr)
        muon_params, adamw_params = [], []
        for p in model.parameters():
            if p.dim() >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return Muon([
            {"params": muon_params, "lr": lr, "weight_decay": wd},
            {"params": adamw_params, "lr": lr * 0.1, "weight_decay": wd},
        ])
    raise ValueError(f"Unknown optimizer: {name}")


class WarmupCosine:
    def __init__(self, optimizer, warmup, total, base_lr, min_ratio=0.01):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total
        self.base_lr = base_lr
        self.min_lr = base_lr * min_ratio
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup:
            lr = self.base_lr * e / max(1, self.warmup)
        else:
            prog = (e - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * prog))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def run_experiment(cfg, run_dir, gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    print(f"[{cfg['name']}] device={device} run_dir={run_dir}")

    # --- 数据 ---
    train_loader, val_loader, test_loader = load_eurosat(
        cfg["data_dir"], batch_size=cfg["batch_size"], aug=cfg["aug"],
        val_split=cfg["val_split"], seed=cfg["seed"],
        num_workers=cfg["num_workers"])

    # --- 模型 ---
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    macs = compute_macs(model)
    print(f"params={n_params:,} MACs={macs:,} ({macs/1e6:.2f}M)")
    if macs > 2e6 and cfg.get("macs_ok") is False:
        print(f"  [WARN] MACs {macs/1e6:.1f}M > 2M budget")

    # --- 初始化 (C2 微调: 从已有 checkpoint 出发) ---
    if cfg.get("init_from"):
        state = torch.load(cfg["init_from"], map_location="cpu")
        msd = model.state_dict()
        filt = {k: v for k, v in state.items()
                if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
        print(f"  init_from {cfg['init_from']}: "
              f"{len(filt)}/{len(msd)} tensors loaded")

    # --- QAT ---
    if cfg["qat"]:
        if cfg.get("qat_version", "v5") == "v9":
            from qat_v9 import prepare_model_v9
            prepare_model_v9(model,
                             layer_sigmas=cfg.get("layer_noise_sigmas", {}),
                             layer_col_off=cfg.get("layer_col_off", {}),
                             layer_col_gain=cfg.get("layer_col_gain", {}),
                             layer_dw_rms=cfg.get("layer_dw_rms", {}),
                             layer_rff_std=cfg.get("layer_rff_std", {}),
                             stem_fp32=cfg.get("stem_fp32", True),
                             head_fp32=cfg.get("head_fp32", True),
                             weight_bits=cfg["weight_bits"],
                             output_noise=cfg["output_noise"],
                             output_quant=cfg["output_quant"],
                             weight_noise=cfg["weight_noise"],
                             weight_noise_ratio=cfg["noise_std_ratio"],
                             activation_style=cfg.get("activation_style", "osim"))
        elif cfg.get("qat_version", "v5") == "v8":
            from qat_v8 import prepare_model_v8
            prepare_model_v8(model,
                             layer_sigmas=cfg.get("layer_noise_sigmas", {}),
                             layer_col_off=cfg.get("layer_col_off", {}),
                             layer_col_gain=cfg.get("layer_col_gain", {}),
                             layer_dw_rms=cfg.get("layer_dw_rms", {}),
                             stem_fp32=cfg.get("stem_fp32", True),
                             head_fp32=cfg.get("head_fp32", True),
                             weight_bits=cfg["weight_bits"],
                             output_noise=cfg["output_noise"],
                             output_quant=cfg["output_quant"],
                             weight_noise=cfg["weight_noise"],
                             weight_noise_ratio=cfg["noise_std_ratio"],
                             activation_style=cfg.get("activation_style", "osim"))
        elif cfg.get("qat_version", "v5") == "v7":
            from qat_v7 import prepare_model_v7
            prepare_model_v7(model,
                             layer_sigmas=cfg.get("layer_noise_sigmas", {}),
                             layer_static_sigmas=cfg.get("layer_static_sigmas", {}),
                             stem_fp32=cfg.get("stem_fp32", True),
                             head_fp32=cfg.get("head_fp32", True),
                             gain_jitter_ch=cfg.get("gain_jitter_ch", 0.0),
                             gain_jitter=cfg.get("gain_jitter", 0.0),
                             off_jitter_raw=cfg.get("off_jitter_raw", 0.0),
                             weight_bits=cfg["weight_bits"],
                             output_noise=cfg["output_noise"],
                             output_quant=cfg["output_quant"],
                             weight_noise=cfg["weight_noise"],
                             weight_noise_ratio=cfg["noise_std_ratio"],
                             activation_style=cfg.get("activation_style", "osim"))
        elif cfg.get("qat_version", "v5") == "v6":
            from qat_v6 import prepare_model_v6
            prepare_model_v6(model,
                             layer_sigmas=cfg.get("layer_noise_sigmas", {}),
                             stem_fp32=cfg.get("stem_fp32", True),
                             head_fp32=cfg.get("head_fp32", True),
                             gain_jitter=cfg.get("gain_jitter", 0.0),
                             off_jitter_raw=cfg.get("off_jitter_raw", 0.0),
                             weight_bits=cfg["weight_bits"],
                             output_noise=cfg["output_noise"],
                             output_quant=cfg["output_quant"],
                             weight_noise=cfg["weight_noise"],
                             weight_noise_ratio=cfg["noise_std_ratio"],
                             activation_style=cfg.get("activation_style", "osim"))
        else:
            from qat_v5 import prepare_model_v5
            prepare_model_v5(model,
                             weight_bits=cfg["weight_bits"],
                             output_noise=cfg["output_noise"],
                             output_noise_ratio=cfg["output_noise_ratio"],
                             output_quant=cfg["output_quant"],
                             weight_noise=cfg["weight_noise"],
                             weight_noise_ratio=cfg["noise_std_ratio"],
                             activation_style=cfg.get("activation_style", "osim"))

    # --- Tier: T0 用短训 + FP32 (代理) ---
    epochs = cfg["t0_epochs"] if cfg["tier"] == "T0" else cfg["epochs"]
    if cfg["tier"] == "T0":
        cfg_qat = cfg["qat"]
        cfg["qat"] = False  # T0 用 FP32 快速代理
    else:
        cfg_qat = None

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = build_optimizer(model, cfg)
    scheduler = WarmupCosine(optimizer, cfg["warmup_epochs"], epochs,
                             cfg["lr"], cfg["min_lr_ratio"])

    # --- EMA (可选) ---
    ema_model = None
    if cfg["ema_decay"] > 0:
        ema_model = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # --- SWA (可选): 维护权重平均 ---
    swa_accum, swa_count = None, 0

    best_acc, best_state, best_epoch = 0.0, None, 0
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    f_log = open(jsonl_path, "w")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, device)
        lr = scheduler.step()
        elapsed = time.time() - t0

        # EMA 更新
        if cfg["ema_decay"] > 0:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_model[k].mul_(cfg["ema_decay"]).add_(v, alpha=1 - cfg["ema_decay"])

        # SWA 累积 (后 swa_start_frac 部分)
        if cfg["swa"] and epoch / epochs >= cfg["swa_start_frac"]:
            if swa_accum is None:
                swa_accum = {k: v.detach().clone() for k, v in model.state_dict().items()
                             if v.dtype.is_floating_point}
            else:
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        swa_accum[k].add_(v.detach())
            swa_count += 1

        # 每 epoch 轻量 val (acc only, 快)
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss += criterion(out, labels).item() * images.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total
        val_loss /= val_total

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        record = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                  "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
                  "time": round(elapsed, 1)}
        f_log.write(json.dumps(record) + "\n")
        f_log.flush()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  ep {epoch:>3d} | tr {tr_loss:.3f}/{tr_acc:.2%} | "
                  f"val {val_loss:.3f}/{val_acc:.2%} | lr {lr:.2e} | {elapsed:.0f}s")

    f_log.close()

    # --- 恢复最佳权重并完整评测 ---
    if cfg["tier"] == "T0":
        cfg["qat"] = cfg_qat  # 还原

    summary = {"name": cfg["name"], "hash": cfg["_hash"], "params": n_params,
               "macs": int(macs), "best_epoch": best_epoch,
               "best_val_acc": best_acc, "epochs_run": epochs}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 完整指标 (val + test)
    val_metrics = evaluate_full(model, val_loader, device, cfg["num_classes"])
    test_metrics = evaluate_full(model, test_loader, device, cfg["num_classes"])
    summary["val"] = {k: v for k, v in val_metrics.items() if k != "confusion"}
    summary["test"] = {k: v for k, v in test_metrics.items() if k != "confusion"}
    summary["val_confusion"] = val_metrics["confusion"]
    summary["test_confusion"] = test_metrics["confusion"]

    # SWA 权重平均评测 (若启用)
    if cfg["swa"] and swa_count > 0:
        for k in swa_accum:
            swa_accum[k] /= swa_count
        model.load_state_dict(swa_accum)
        swa_val = evaluate_full(model, val_loader, device, cfg["num_classes"])
        swa_test = evaluate_full(model, test_loader, device, cfg["num_classes"])
        summary["swa_val"] = {k: v for k, v in swa_val.items() if k != "confusion"}
        summary["swa_test"] = {k: v for k, v in swa_test.items() if k != "confusion"}
        print(f"  SWA: val {swa_val['accuracy']:.2%} test {swa_test['accuracy']:.2%}")

    # EMA 评测 (若启用)
    if cfg["ema_decay"] > 0:
        model.load_state_dict(ema_model)
        ema_val = evaluate_full(model, val_loader, device, cfg["num_classes"])
        ema_test = evaluate_full(model, test_loader, device, cfg["num_classes"])
        summary["ema_val"] = {k: v for k, v in ema_val.items() if k != "confusion"}
        summary["ema_test"] = {k: v for k, v in ema_test.items() if k != "confusion"}
        print(f"  EMA: val {ema_val['accuracy']:.2%} test {ema_test['accuracy']:.2%}")

    # 恢复最佳权重保存
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(run_dir, "best.pth"))
    torch.save(model.state_dict(), os.path.join(run_dir, "last.pth"))

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] {cfg['name']} best_val={best_acc:.2%} "
          f"test_acc={test_metrics['accuracy']:.2%} "
          f"test_f1={test_metrics['macro_f1']:.2%} "
          f"ece={test_metrics['ece']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    cfg, h = load_config(args.config)
    if args.base_dir:
        cfg["run_dir"] = args.base_dir
    run_dir = make_run_dir(cfg)
    print(f"run_dir: {run_dir}")
    if args.list_only:
        return
    run_experiment(cfg, run_dir, gpu=args.gpu)


if __name__ == "__main__":
    main()
