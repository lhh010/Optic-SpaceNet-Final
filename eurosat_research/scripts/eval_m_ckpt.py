"""
===============================================================================
 eval_m_ckpt.py — M-validate: 独立复测 ckpt 的 test acc (X0 标准口径)
===============================================================================
 口径 = x0r_* run 的 summary test acc:
   build_model(cfg) -> prepare_model_v8 (与 config 相同噪声标定; eval 模式 noise off)
   -> 载入 ckpt -> model.eval() -> evaluate_full(test_loader)  (test 5400 clean,
   quant on: weight int8 per-channel + 输入 uint8 + ADC 12-bit 输出量化)
 用法:
   python scripts/eval_m_ckpt.py --config configs/m6_j1_v8probe15.json \
       --ckpt weights/m6_j1_v8probe15.pth --gpu 0
===============================================================================
"""
import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from config import load_config          # noqa: E402
from models import build_model, compute_macs  # noqa: E402
from data import load_eurosat           # noqa: E402
from metrics import evaluate_full       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-dir", default=None, help="覆盖 config 的 data_dir")
    args = ap.parse_args()

    cfg, _h = load_config(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    _tr, _val, test_loader = load_eurosat(
        cfg["data_dir"], batch_size=cfg["batch_size"], aug=cfg["aug"],
        val_split=cfg["val_split"], seed=cfg["seed"],
        num_workers=cfg["num_workers"])

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    macs = compute_macs(model)

    assert cfg["qat"] and cfg.get("qat_version") == "v8", "M-validate 只覆盖 v8"
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

    state = torch.load(args.ckpt, map_location="cpu")
    msd = model.state_dict()
    filt = {k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd if k not in filt]
    model.load_state_dict(filt, strict=False)
    print(f"[ckpt] {args.ckpt}: {len(filt)}/{len(msd)} tensors loaded; "
          f"missing={missing if missing else 'none'}")

    model.eval()
    test_metrics = evaluate_full(model, test_loader, device, cfg["num_classes"])
    out = {
        "config": args.config, "ckpt": args.ckpt,
        "params": n_params, "macs": int(macs),
        "test_acc": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_n": len(test_loader.dataset),
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
