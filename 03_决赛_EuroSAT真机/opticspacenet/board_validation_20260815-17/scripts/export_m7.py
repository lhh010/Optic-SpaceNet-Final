"""
===============================================================================
 export_m7.py — 导出 M7 (J1-w075, 全 1x1, 无 conv3s2) QAT 权重 → 真机部署格式
 基于 export_ds3.py, 移除 conv3s2 下采样层 (M7 无 s1ds/s2ds)。
 层命名 (runner/probe/calib 共用, 与 run_j1_gazelle.py 一致):
   stem — stem.0 conv3x3 s2 (电) + stem.1 BN + MaxPool2
   s1a  — stage1.0 1x1 + stage1.1 BN
   s2a  — stage2.0 1x1 / s2b — stage2.3 1x1
   s3a  — stage3.0 1x1 / s3b — stage3.3 1x1
   h1/h2 — head (int8 + float wf + bias)
 用法: python3 export_m7.py <ckpt> <config.json> <out_dir>
===============================================================================
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, _SRC)
from models import build_model  # noqa: E402


def quantize_w_per_channel(w_np):
    amax = np.abs(w_np).max(axis=tuple(range(1, w_np.ndim)), keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    w_int = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return w_int, scale


def export(weight_path, cfg_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(cfg_path))
    cfg.setdefault("num_classes", 10)
    model = build_model(cfg)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    sd = model.state_dict()
    meta = {}

    def get(*parts):
        k = ".".join(parts)
        return sd[k].numpy() if k in sd else None

    # ---- stem (电计算, float) ----
    stem_w = get("stem", "0", "weight")
    np.save(os.path.join(out_dir, "stem_w.npy"), stem_w)
    meta["stem_scale"] = 1.0
    stem_bn = np.stack([get("stem", "1", "weight"), get("stem", "1", "bias"),
                        get("stem", "1", "running_mean"),
                        get("stem", "1", "running_var")])
    np.save(os.path.join(out_dir, "stem_bn.npy"), stem_bn)
    meta["stem_bn_eps"] = 1e-5
    meta["stem_pool_mode"] = cfg.get("stem_pool_mode") or "max"

    def export_bn(layer_name, *parts):
        w = get(*parts, "weight")
        b = get(*parts, "bias")
        rm = get(*parts, "running_mean")
        rv = get(*parts, "running_var")
        assert w is not None, f"{parts} BN weight missing"
        np.save(os.path.join(out_dir, f"{layer_name}_bn.npy"),
                np.stack([w, b, rm, rv]))
        meta[f"{layer_name}_bn_eps"] = 1e-5

    def export_conv1x1(layer_name, *parts):
        w = get(*parts, "weight").squeeze()
        wq, s = quantize_w_per_channel(w)
        np.save(os.path.join(out_dir, f"{layer_name}_w.npy"), wq)
        meta[f"{layer_name}_scale"] = s.reshape(-1).tolist()

    # ---- stage1: 1x1 + pool ----
    export_conv1x1("s1a", "stage1", "0")
    export_bn("s1a", "stage1", "1")
    # ---- stage2: 1x1 x2 + pool ----
    export_conv1x1("s2a", "stage2", "0")
    export_bn("s2a", "stage2", "1")
    export_conv1x1("s2b", "stage2", "3")
    export_bn("s2b", "stage2", "4")
    # ---- stage3: 1x1 x2 (无 pool) ----
    export_conv1x1("s3a", "stage3", "0")
    export_bn("s3a", "stage3", "1")
    export_conv1x1("s3b", "stage3", "3")
    export_bn("s3b", "stage3", "4")
    # ---- head: GAP -> FC -> ReLU -> FC ----
    for tag, parts in [("h1", ("head", "2")), ("h2", ("head", "4"))]:
        wgt = get(*parts, "weight")
        b = get(*parts, "bias")
        assert wgt is not None and b is not None, f"{parts} missing"
        np.save(os.path.join(out_dir, f"{tag}_wf.npy"), wgt)
        np.save(os.path.join(out_dir, f"{tag}_bias.npy"), b)
        w, s = quantize_w_per_channel(wgt)
        np.save(os.path.join(out_dir, f"{tag}_w.npy"), w)
        meta[f"{tag}_scale"] = s.reshape(-1).tolist()

    meta["source"] = weight_path
    meta["config"] = cfg_path
    meta["arch"] = {k: cfg.get(k) for k in
                    ("channels", "pool_mode", "stem_pool_mode", "kernels",
                     "fast_downsample", "stem_stride")}
    meta["images"] = "test_images_j1.npy"
    meta["labels"] = "test_labels_j1.npy"
    meta["n_test"] = 5400
    meta["data_source"] = "test5400_ccic_20260816"

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"exported to {out_dir}")
    print(f"  stem: {stem_w.shape} (float, electronic)")
    print(f"  layers: {[k for k in meta if k.endswith('_scale')]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("config")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    export(a.ckpt, a.config, a.out_dir)
