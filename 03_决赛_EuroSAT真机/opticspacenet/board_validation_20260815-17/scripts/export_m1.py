"""
export_m1.py — 导出 Model 1 Baseline VGG (变体 A: conv1_1 电计算) → 真机部署格式
基于 optic_inference_int8_model1.py 的 BaselineVGG (6 Conv 3x3 + 2 Linear, bias=False, BN)。
光层 (变体 A): conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, fc1, fc2 (7 光层)
用法: python3 export_m1.py <ckpt> <out_dir>
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

_SCRIPTS = "/workspace/train-test/src/scripts"
sys.path.insert(0, _SCRIPTS)
from optic_inference_int8_model1 import BaselineVGG  # noqa: E402


def quantize_w_per_channel(w_np):
    amax = np.abs(w_np).max(axis=tuple(range(1, w_np.ndim)), keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    w_int = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return w_int, scale


def export(weight_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = BaselineVGG(num_classes=10)
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

    def save_bn(layer, *parts):
        w, b = get(*parts, "weight"), get(*parts, "bias")
        rm, rv = get(*parts, "running_mean"), get(*parts, "running_var")
        assert w is not None
        np.save(os.path.join(out_dir, f"{layer}_bn.npy"),
                np.stack([w, b, rm, rv]))
        meta[f"{layer}_bn_eps"] = 1e-5

    # ---- conv1_1: 电计算 (float) ----
    np.save(os.path.join(out_dir, "conv1_1_w.npy"), get("conv1_1", "weight"))
    save_bn("conv1_1", "bn1_1")
    meta["conv1_1_scale"] = 1.0

    # ---- 光卷积层: int8 per-channel ----
    for name, attr in [("conv1_2", "conv1_2"), ("conv2_1", "conv2_1"),
                       ("conv2_2", "conv2_2"), ("conv3_1", "conv3_1"),
                       ("conv3_2", "conv3_2")]:
        w = get(attr, "weight")  # (C_out, C_in, 3, 3)
        w2 = w.reshape(w.shape[0], -1)  # (C_out, 9*C_in)
        wq, s = quantize_w_per_channel(w2)
        np.save(os.path.join(out_dir, f"{name}_w.npy"), wq)
        meta[f"{name}_scale"] = s.reshape(-1).tolist()
        bn = attr.replace("conv", "bn")  # conv1_2 -> bn1_2
        save_bn(name, bn)

    # ---- 光 FC: int8 per-channel (bias=False) ----
    for name, attr in [("fc1", "fc1"), ("fc2", "fc2")]:
        w = get(attr, "weight")  # (C_out, C_in)
        wq, s = quantize_w_per_channel(w)
        np.save(os.path.join(out_dir, f"{name}_w.npy"), wq)
        meta[f"{name}_scale"] = s.reshape(-1).tolist()
        wf = get(attr, "weight")
        np.save(os.path.join(out_dir, f"{name}_wf.npy"), wf)

    meta["source"] = weight_path
    meta["variant"] = "A"
    meta["images"] = "test_images_j1.npy"
    meta["labels"] = "test_labels_j1.npy"
    meta["n_test"] = 5400
    meta["data_source"] = "test5400_ccic_20260816"
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"exported to {out_dir}")
    print("layers:", [k for k in meta if k.endswith("_scale")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    export(a.ckpt, a.out_dir)
