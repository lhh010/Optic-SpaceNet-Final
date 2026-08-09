"""
===============================================================================
 eval_ckpt.py — 用当前代码对已有 checkpoint 做确定性 re-eval
===============================================================================
 用法:
   python src/eval_ckpt.py --config configs/r3_J1_long.json \
       --ckpt runs/r3_J1_long_3b6c03f6/best.pth --data-dir ../data/EuroSAT_RGB

 QAT 噪声仅在 training=True 时注入, eval 确定性, 可逐位复现历史 summary 口径。
===============================================================================
"""
import os
import sys
import json
import argparse
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import load_config
from models import build_model
from data import load_eurosat
from metrics import evaluate_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    cfg, _ = load_config(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    torch.manual_seed(cfg["seed"])

    _, val_loader, test_loader = load_eurosat(
        cfg["data_dir"], batch_size=256, aug="none",
        val_split=cfg["val_split"], seed=cfg["seed"], num_workers=0)

    model = build_model(cfg)
    from qat_v5 import prepare_model_v5
    prepare_model_v5(model,
                     weight_bits=cfg["weight_bits"],
                     output_noise=cfg["output_noise"],
                     output_noise_ratio=cfg["output_noise_ratio"],
                     output_quant=cfg["output_quant"],
                     weight_noise=cfg["weight_noise"],
                     weight_noise_ratio=cfg["noise_std_ratio"],
                     activation_style=cfg.get("activation_style", "osim"))

    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:5]:
            print(f"  missing: {k}")
        for k in list(unexpected)[:5]:
            print(f"  unexpected: {k}")

    device = torch.device("cpu")
    out = {"ckpt": args.ckpt, "config": args.config}
    for rep in range(args.repeats):
        vm = evaluate_full(model, val_loader, device, cfg["num_classes"])
        tm = evaluate_full(model, test_loader, device, cfg["num_classes"])
        tag = f"[rep{rep}] " if args.repeats > 1 else ""
        print(f"{tag}val  acc={vm['accuracy']:.4%} loss={vm['loss']:.4f} (n={vm['n']})")
        print(f"{tag}test acc={tm['accuracy']:.4%} loss={tm['loss']:.4f} "
              f"macro_f1={tm['macro_f1']:.4f} ece={tm['ece']:.4f} (n={tm['n']})")
        out.setdefault("evals", []).append({
            "val_acc": vm["accuracy"], "test_acc": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"], "test_ece": tm["ece"]})
    return out


if __name__ == "__main__":
    main()
