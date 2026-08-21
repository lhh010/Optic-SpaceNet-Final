"""
===============================================================================
 export_j1.py — 导出 J1 QAT 权重 → 真机部署格式
===============================================================================
 从 torch 训练的 J1_long best.pth 提取:
   - 每层 float 权重 → per-channel int8 + scale (与 QAT 训练一致)
   - 测试集 images/labels (npy)
 输出目录结构 (weights_j1/):
   meta.json         — 层名/scale/数据文件名
   {layer}_w.npy     — int8 权重
   {layer}_scale.npy — per-channel scale
   注: stem 电计算 (float 权重直接存), 其余层光计算 (int8 + scale)
===============================================================================
"""
import os
import sys
import json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "Ltsimulator-test", "auto_research", "src"))
from models import MiniVGG
from data import load_eurosat


def quantize_w_per_channel(w_np):
    """per-channel signed int8 + scale (与 QAT v5 quant_int8_per_channel 一致)。"""
    amax = np.abs(w_np).max(axis=tuple(range(1, w_np.ndim)), keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    w_int = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return w_int, scale


def export(weight_path, out_dir, data_dir="data/EuroSAT_RGB", n_test=1000):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 构建 J1 + 加载权重 (float)
    model = MiniVGG(num_classes=10, channels=[16, 32, 64, 128],
                    stem_stride=2, fast_downsample=True,
                    kernels=(1, 1, 1), head_dims=[128])
    state = torch.load(weight_path, map_location="cpu")
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    sd = model.state_dict()
    meta = {}

    def get(*parts):
        k = ".".join(parts)
        return sd[k].numpy() if k in sd else None

    # 2. 逐层导出
    # stem (电计算, float)
    stem_w = get("stem", "0", "weight")  # (16,3,3,3)
    np.save(os.path.join(out_dir, "stem_w.npy"), stem_w)
    meta["stem_scale"] = 1.0
    # stem BN 参数 (电计算路径需要): [w, b, running_mean, running_var]
    stem_bn = np.stack([get("stem", "1", "weight"), get("stem", "1", "bias"),
                        get("stem", "1", "running_mean"),
                        get("stem", "1", "running_var")])
    np.save(os.path.join(out_dir, "stem_bn.npy"), stem_bn)
    meta["stem_bn_eps"] = 1e-5

    def export_bn(layer_name, *parts):
        """导出 BN 参数 [w, b, running_mean, running_var] → {layer_name}_bn.npy。
        BN 位于光计算 conv 之后, 推理需在反量化输出上应用。"""
        w = get(*parts, "weight")
        b = get(*parts, "bias")
        rm = get(*parts, "running_mean")
        rv = get(*parts, "running_var")
        assert w is not None, f"{parts} BN weight missing"
        arr = np.stack([w, b, rm, rv])
        np.save(os.path.join(out_dir, f"{layer_name}_bn.npy"), arr)
        meta[f"{layer_name}_bn_eps"] = 1e-5
        return arr

    # stage1.0 (16->32, 1x1) + 其后 BN (stage1.1)
    s1a_w = get("stage1", "0", "weight").squeeze()  # (32,16)
    w, s = quantize_w_per_channel(s1a_w)
    np.save(os.path.join(out_dir, "s1a_w.npy"), w)
    meta["s1a_scale"] = s.reshape(-1).tolist()
    export_bn("s1a", "stage1", "1")

    # stage2.0 (32->64) + BN (stage2.1)
    s2a_w = get("stage2", "0", "weight").squeeze()  # (64,32)
    w, s = quantize_w_per_channel(s2a_w)
    np.save(os.path.join(out_dir, "s2a_w.npy"), w)
    meta["s2a_scale"] = s.reshape(-1).tolist()
    export_bn("s2a", "stage2", "1")

    # stage2.3 (64->64) + BN (stage2.4)
    s2b_w = get("stage2", "3", "weight").squeeze()  # (64,64)
    w, s = quantize_w_per_channel(s2b_w)
    np.save(os.path.join(out_dir, "s2b_w.npy"), w)
    meta["s2b_scale"] = s.reshape(-1).tolist()
    export_bn("s2b", "stage2", "4")

    # stage3.0 (64->128) + BN (stage3.1)
    s3a_w = get("stage3", "0", "weight").squeeze()  # (128,64)
    w, s = quantize_w_per_channel(s3a_w)
    np.save(os.path.join(out_dir, "s3a_w.npy"), w)
    meta["s3a_scale"] = s.reshape(-1).tolist()
    export_bn("s3a", "stage3", "1")

    # stage3.3 (128->128) + BN (stage3.4)
    s3b_w = get("stage3", "3", "weight").squeeze()  # (128,128)
    w, s = quantize_w_per_channel(s3b_w)
    np.save(os.path.join(out_dir, "s3b_w.npy"), w)
    meta["s3b_scale"] = s.reshape(-1).tolist()
    export_bn("s3b", "stage3", "4")

    # head: GAP(128) -> FC(128->128) ReLU -> FC(128->10)
    # 权重布局: (C_out, C_in) — 保持原样, quantize_w_per_channel 沿 dim0 (输出通道)
    # 同时导出 float 权重 + bias: bias 参与 logits 必须部署; float 供 HEAD_ELEC 电计算模式
    for tag, parts in [("h1", ("head", "2")), ("h2", ("head", "4"))]:
        wgt = get(*parts, "weight")
        b = get(*parts, "bias")
        assert wgt is not None and b is not None, f"{parts} weight/bias missing"
        np.save(os.path.join(out_dir, f"{tag}_wf.npy"), wgt)
        np.save(os.path.join(out_dir, f"{tag}_bias.npy"), b)
        w, s = quantize_w_per_channel(wgt)
        np.save(os.path.join(out_dir, f"{tag}_w.npy"), w)
        meta[f"{tag}_scale"] = s.reshape(-1).tolist()

    # 3. 测试数据 (test 段, 与训练一致的 split)
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "Ltsimulator-test", "src", "data"))
    from eurosat_split import split_indices as _si

    from torchvision import datasets, transforms
    from torch.utils.data import Subset
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
    ds = datasets.ImageFolder(data_dir, transform=tf)
    n = len(ds)
    tr, va, te = _si(n, seed=42, val_ratio=0.2, test_ratio=0.2)
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
    meta["source"] = weight_path

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"exported to {out_dir}")
    print(f"  stem: {stem_w.shape} (float, electronic)")
    print(f"  test data: {images.shape} labels={labels.shape}")
    print(f"  layers: {[k for k in meta if k.endswith('_scale')]}")


if __name__ == "__main__":
    weight_path = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/r3_J1_long_3b6c03f6/best.pth"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "weights_j1"
    export(weight_path, out_dir)
