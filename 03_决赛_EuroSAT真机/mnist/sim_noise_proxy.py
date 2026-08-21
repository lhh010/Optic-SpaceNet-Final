# -*- coding: utf-8 -*-
"""sim_noise_proxy.py — numpy 噪声代理: 预测部署权重的 hw 精度下限。
用法: python3 sim_noise_proxy.py <weights_dir> [n] [seeds...]
在 FAKE 前向的 5 个光计算 conv 层 raw 域注入 iid N(0, resid_std) 噪声
(calib_j1_real.json 实测口径)。clean / noisy 两组精度都打印。
"""
import os
import sys

import numpy as np

os.environ.setdefault("J1_FAKE", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_j1_gazelle as R

# calib_j1_real.json (2026-08-07 两次校准一致) 的 per-layer resid_std
SIG = {"s1a": 683.5, "s2a": 995.0, "s2b": 1352.4, "s3a": 1619.0, "s3b": 1297.1}


def optical_noisy(x, w_int, w_scale, sigma, rng):
    """复刻 R.optical_conv1x1, raw 域注入 iid 噪声。"""
    B, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(B * H * W, C)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    y = R.optical_mm(x_int, w_int.T)
    if sigma > 0:
        y = y + rng.standard_normal(y.shape) * sigma
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int.T.astype(np.float64).sum(axis=0, keepdims=True)
    y = x_scale * w_scale_arr * y - x_scale * x_zp * w_scale_arr * col_sum
    return y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)


def forward(x, ws, meta, sigma_map, rng, eps):
    h = R.stem_forward(x, ws, meta)
    for name, pool in [("s1a", True), ("s2a", False), ("s2b", True),
                       ("s3a", False), ("s3b", False)]:
        h = optical_noisy(h, ws[name], meta[f"{name}_scale"],
                          sigma_map.get(name, 0.0), rng)
        h = R.apply_bn(h, ws[f"{name}_bn"], eps)
        h = R.relu(h)
        if pool:
            h = R.pool2d(h, 2)
    g = h.mean(axis=(2, 3))
    z = R.relu(g @ ws["h1_wf"].T + ws["h1_bias"])
    return z @ ws["h2_wf"].T + ws["h2_bias"]


def evaluate(ws, meta, images, labels, sigma_map, seed):
    rng = np.random.default_rng(seed)
    eps = float(meta.get("stem_bn_eps", 1e-5))
    correct = 0
    for s in range(0, len(labels), 8):
        e = min(s + 8, len(labels))
        pred = forward(images[s:e], ws, meta, sigma_map, rng, eps).argmax(1)
        correct += int((pred == labels[s:e]).sum())
    return correct / len(labels) * 100


def main():
    wdir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    seeds = [int(s) for s in sys.argv[3:]] or [0, 1, 2]
    os.environ["J1_WEIGHTS_DIR"] = wdir
    R.WDIR = wdir
    ws, meta = R.load_weights()
    images = np.load(os.path.join(wdir, meta["images"]))[:n]
    labels = np.load(os.path.join(wdir, meta["labels"]))[:n]
    clean = evaluate(ws, meta, images, labels, {}, 0)
    noisy = [evaluate(ws, meta, images, labels, SIG, s) for s in seeds]
    print(f"{wdir}: clean={clean:.2f}% "
          f"iid-full={[f'{a:.2f}' for a in noisy]} "
          f"mean={np.mean(noisy):.2f}%")


if __name__ == "__main__":
    main()
