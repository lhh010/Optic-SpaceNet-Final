# -*- coding: utf-8 -*-
"""sim_noise_proxy_v7.py — 结构化噪声代理 (round5 重复性实验口径)。
用法: python3 sim_noise_proxy_v7.py <weights_dir> [n] [seeds...]
噪声模型 (逼近真机):
  每个光计算 conv raw 域:
    iid 快噪声 N(0, 260.9)  [crossval 短窗口径]
    per-channel 静态偏移 off_c ~ N(0, sigma_static_l), 一次采样整跑固定
    per-channel 增益 g_c ~ N(1, 0.02), 一次采样整跑固定
  sigma_static_l 来自板上探针 calib_c2c_seg2.json (2026-08-08)。
"""
import os
import sys

import numpy as np

os.environ.setdefault("J1_FAKE", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_j1_gazelle as R

SIG_DYN = 260.9
SIG_STATIC = {"s1a": 708.3, "s2a": 1102.9, "s2b": 1373.9, "s3a": 1747.9, "s3b": 1318.8}
GAIN_CH = 0.02


def optical_noisy(x, w_int, w_scale, sigma_dyn, off_c, gain_c, rng):
    B, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(B * H * W, C)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    y = R.optical_mm(x_int, w_int.T)
    y = y * gain_c.reshape(1, -1) + off_c.reshape(1, -1)
    if sigma_dyn > 0:
        y = y + rng.standard_normal(y.shape) * sigma_dyn
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int.T.astype(np.float64).sum(axis=0, keepdims=True)
    y = x_scale * w_scale_arr * y - x_scale * x_zp * w_scale_arr * col_sum
    return y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)


def evaluate(ws, meta, images, labels, seed):
    rng = np.random.default_rng(seed)
    eps = float(meta.get("stem_bn_eps", 1e-5))
    # 整跑一次采样的结构化分量
    struct = {}
    for name in ["s1a", "s2a", "s2b", "s3a", "s3b"]:
        c_out = ws[name].shape[0]  # w_int: (C_out, C_in)
        struct[name] = (rng.standard_normal(c_out) * SIG_STATIC[name],
                        1.0 + rng.standard_normal(c_out) * GAIN_CH)
    correct = 0
    for s in range(0, len(labels), 8):
        e = min(s + 8, len(labels))
        h = R.stem_forward(images[s:e], ws, meta)
        for name, pool in [("s1a", True), ("s2a", False), ("s2b", True),
                           ("s3a", False), ("s3b", False)]:
            off_c, gain_c = struct[name]
            h = optical_noisy(h, ws[name], meta[f"{name}_scale"],
                              SIG_DYN, off_c, gain_c, rng)
            h = R.apply_bn(h, ws[f"{name}_bn"], eps)
            h = R.relu(h)
            if pool:
                h = R.pool2d(h, 2)
        g = h.mean(axis=(2, 3))
        z = R.relu(g @ ws["h1_wf"].T + ws["h1_bias"])
        pred = (z @ ws["h2_wf"].T + ws["h2_bias"]).argmax(1)
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
    accs = [evaluate(ws, meta, images, labels, s) for s in seeds]
    print(f"{wdir}: structured-proxy accs={[f'{a:.2f}' for a in accs]} "
          f"mean={np.mean(accs):.2f}%")


if __name__ == "__main__":
    main()
