"""
===============================================================================
 export_ds3.py — 导出 X0 ds3 变体 (w075ds3 / ds3pool3) QAT 权重 → 真机部署格式
===============================================================================
 在 export_j1.py 基础上适配 X0 新旋钮:
   - pool_mode="conv3s2": stage 下采样点换为 3x3 stride2 conv (光计算层, k=9C im2col)
   - stem_pool_mode: "max" (w075ds3) / "max3" (ds3pool3), 记入 meta 供 runner 选择
   - width_mult 已折进 channels (w075ds3: [12,24,48,96]; ds3pool3: [16,32,64,128])

 层命名 (runner/probe/calib 共用):
   stem  — stem.0 conv3x3 s2 (电计算 float) + stem.1 BN
   s1a   — stage1.0 1x1 conv  + stage1.1 BN
   s1ds  — stage1.3.0 3x3 s2 conv (光) + stage1.3.1 BN
   s2a   — stage2.0 1x1 / s2b — stage2.3 1x1
   s2ds  — stage2.6.0 3x3 s2 conv (光) + stage2.6.1 BN
   s3a   — stage3.0 1x1 / s3b — stage3.3 1x1
   h1/h2 — head.2 / head.4 (int8 + float wf + bias)

 用法:
   python3 export_ds3.py <ckpt> <config.json> <out_dir> [--data-dir D] [--n-test N]
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
    """per-channel signed int8 + scale (与 QAT 训练一致)。"""
    amax = np.abs(w_np).max(axis=tuple(range(1, w_np.ndim)), keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    w_int = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return w_int, scale


def export(weight_path, cfg_path, out_dir, data_dir=None, n_test=1000):
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(cfg_path))
    cfg.setdefault("num_classes", 10)
    model = build_model(cfg)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=True), None
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
    # stem pool: conv3s2 变体 stem 下采样为 max (w075ds3) 或 max3 (ds3pool3)
    pool_mode = cfg.get("pool_mode", "max")
    stem_pool_mode = cfg.get("stem_pool_mode") or \
        ("max" if pool_mode == "conv3s2" else pool_mode)
    meta["stem_pool_mode"] = stem_pool_mode

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
        w = get(*parts, "weight").squeeze()  # (C_out, C_in)
        wq, s = quantize_w_per_channel(w)
        np.save(os.path.join(out_dir, f"{layer_name}_w.npy"), wq)
        meta[f"{layer_name}_scale"] = s.reshape(-1).tolist()

    def export_conv3s2(layer_name, *parts):
        """3x3 s2 conv (光): (C_out, C, 3, 3) → 2D (C_out, 9C) int8。
        im2col patch 展平顺序 (C, kh, kw) row-major, 与 reshape 一致。"""
        w = get(*parts, "weight")  # (C_out, C, 3, 3)
        w2 = w.reshape(w.shape[0], -1)
        wq, s = quantize_w_per_channel(w2)
        np.save(os.path.join(out_dir, f"{layer_name}_w.npy"), wq)
        meta[f"{layer_name}_scale"] = s.reshape(-1).tolist()

    # ---- stage1: 1x1 + conv3s2 下采样 ----
    export_conv1x1("s1a", "stage1", "0")
    export_bn("s1a", "stage1", "1")
    export_conv3s2("s1ds", "stage1", "3", "0")
    export_bn("s1ds", "stage1", "3", "1")
    # ---- stage2: 1x1 ×2 + conv3s2 下采样 ----
    export_conv1x1("s2a", "stage2", "0")
    export_bn("s2a", "stage2", "1")
    export_conv1x1("s2b", "stage2", "3")
    export_bn("s2b", "stage2", "4")
    export_conv3s2("s2ds", "stage2", "6", "0")
    export_bn("s2ds", "stage2", "6", "1")
    # ---- stage3: 1x1 ×2 (无 pool) ----
    export_conv1x1("s3a", "stage3", "0")
    export_bn("s3a", "stage3", "1")
    export_conv1x1("s3b", "stage3", "3")
    export_bn("s3b", "stage3", "4")
    # ---- head: GAP → FC(C3→C3) ReLU → FC(C3→10) ----
    for tag, parts in [("h1", ("head", "2")), ("h2", ("head", "4"))]:
        wgt = get(*parts, "weight")
        b = get(*parts, "bias")
        assert wgt is not None and b is not None, f"{parts} weight/bias missing"
        np.save(os.path.join(out_dir, f"{tag}_wf.npy"), wgt)
        np.save(os.path.join(out_dir, f"{tag}_bias.npy"), b)
        w, s = quantize_w_per_channel(wgt)
        np.save(os.path.join(out_dir, f"{tag}_w.npy"), w)
        meta[f"{tag}_scale"] = s.reshape(-1).tolist()

    # ---- 测试数据 (与训练/C1 同 split 同口径) ----
    if data_dir:
        sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "src", "data"))
        from eurosat_split import split_indices as _si
        from torchvision import datasets, transforms
        from torch.utils.data import Subset
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
        ds = datasets.ImageFolder(data_dir, transform=tf)
        tr, va, te = _si(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
        test_ds = Subset(ds, te[:n_test])
        images, labels = [], []
        for img, lbl in test_ds:
            images.append(img.numpy())
            labels.append(lbl)
        images = np.stack(images).astype(np.float32)
        labels = np.array(labels, dtype=np.int64)
        np.save(os.path.join(out_dir, "test_images_j1.npy"), images)
        np.save(os.path.join(out_dir, "test_labels_j1.npy"), labels)
        meta["images"] = "test_images_j1.npy"
        meta["labels"] = "test_labels_j1.npy"
        meta["n_test"] = n_test
        print(f"  test data: {images.shape} labels={labels.shape}")

    meta["source"] = weight_path
    meta["config"] = cfg_path
    meta["arch"] = {k: cfg.get(k) for k in
                    ("channels", "pool_mode", "stem_pool_mode", "kernels",
                     "fast_downsample", "stem_stride")}

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"exported to {out_dir}")
    print(f"  stem: {stem_w.shape} (float, electronic), stem_pool={stem_pool_mode}")
    print(f"  layers: {[k for k in meta if k.endswith('_scale')]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-test", type=int, default=1000)
    a = ap.parse_args()
    export(a.ckpt, a.config, a.out_dir, a.data_dir, a.n_test)
