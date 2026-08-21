# -*- coding: utf-8 -*-
"""sim_noise_proxy_v8.py — 组分噪声代理 (probe_dump 实测口径, 2026-08-08)。
用法: python3 sim_noise_proxy_v8.py <weights_dir> [n] [seeds...]
每个光计算 conv raw 域, 整跑一次采样 + 逐样本:
  1. per-column 常量偏移 off_c ~ N(0, COL_OFF[l])     [probe 实测 4-23% 方差]
  2. per-column 增益 g_c ~ N(1, COL_GAIN[l])          [probe 实测 1-3%]
  3. per-element 权重扰动 dW ~ N(0, DW_RMS[l]) counts  [probe 线性回归 21-50%]
  4. 残余 iid N(0, IID_RESID[l])                      [总 resid 减去 1-3 解释部分]
全部整跑一次采样 (结构化, 与 rep 实验 95.2% 结构化占比一致)。
"""
import os
import sys

import numpy as np

os.environ.setdefault("J1_FAKE", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_j1_gazelle as R

# probe_dump (2026-08-08, weights_c2c) 实测
COL_OFF = {"s1a": 281.0, "s2a": 478.0, "s2b": 459.0, "s3a": 872.0, "s3b": 264.0}
COL_GAIN = {"s1a": 0.0251, "s2a": 0.0240, "s2b": 0.0120, "s3a": 0.0157, "s3b": 0.0126}
DW_RMS = {"s1a": 3.72, "s2a": 6.49, "s2b": 5.00, "s3a": 4.67, "s3b": 7.23}
# after-linear 残差 std (含 iid 261 + 非线性结构化), 作为 iid 注入 (过悲观上界)
RESID_AL = {"s1a": 668.0, "s2a": 971.0, "s2b": 1210.0, "s3a": 1279.0, "s3b": 1140.0}
# 可选: 只注入 crossval 快噪声 260 (乐观下界)
IID_ONLY = {"s1a": 260.9, "s2a": 260.9, "s2b": 260.9, "s3a": 260.9, "s3b": 260.9}
# RFF 非线性结构化残差 (sqrt(RESID_AL^2 - 260.9^2)), PROXY_RFF=1 时启用
RFF_STD = {"s1a": 614.9, "s2a": 935.3, "s2b": 1181.5, "s3a": 1252.1, "s3b": 1109.7}

MODE = os.environ.get("PROXY_IID", "full")  # full | fast
USE_RFF = os.environ.get("PROXY_RFF", "0") == "1"


def evaluate(ws, meta, images, labels, seed):
    rng = np.random.default_rng(seed)
    eps = float(meta.get("stem_bn_eps", 1e-5))
    iid = RESID_AL if MODE == "full" else IID_ONLY
    # 整跑一次采样的结构化分量
    struct = {}
    for name in ["s1a", "s2a", "s2b", "s3a", "s3b"]:
        w = ws[name]  # (C_out, C_in) int8
        struct[name] = (
            rng.standard_normal(w.shape[0]) * COL_OFF[name],
            1.0 + rng.standard_normal(w.shape[0]) * COL_GAIN[name],
            rng.standard_normal(w.shape) * DW_RMS[name],
            (rng.standard_normal((w.shape[1], 64)) * 3.0,
             rng.standard_normal((64, w.shape[0])) / 8.0,
             rng.random(64) * 2 * np.pi) if USE_RFF else None,
        )
    correct = 0
    for s in range(0, len(labels), 8):
        e = min(s + 8, len(labels))
        h = R.stem_forward(images[s:e], ws, meta)
        for name, pool in [("s1a", True), ("s2a", False), ("s2b", True),
                           ("s3a", False), ("s3b", False)]:
            off_c, gain_c, dW, rff = struct[name]
            B, C, H, W = h.shape
            x_flat = h.transpose(0, 2, 3, 1).reshape(B * H * W, C)
            x_int, x_scale, x_zp = R.quantize_act(x_flat)
            w_int = ws[name]
            y = x_int.astype(np.float64) @ (w_int.astype(np.float64) + dW).T  # (m, C_out)
            y = y * gain_c.reshape(1, -1) + off_c.reshape(1, -1)
            if rff is not None:
                Br, Ar, phi = rff
                xn = x_int.astype(np.float64)
                xn = xn / (np.abs(xn).max(axis=1, keepdims=True) + 1e-6)
                f = np.cos(xn @ Br + phi) @ Ar
                f = f / (f.std() + 1e-6) * RFF_STD[name]
                y = y + f
            sig = iid[name]
            if sig > 0:
                y = y + rng.standard_normal(y.shape) * sig
            w_scale = np.asarray(meta[f"{name}_scale"], dtype=np.float64).reshape(1, -1)
            col_sum = (w_int + dW).T.astype(np.float64).sum(axis=0, keepdims=True)
            y = x_scale * w_scale * y - x_scale * x_zp * w_scale * col_sum
            h = y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
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
    print(f"{wdir} [iid={MODE}]: accs={[f'{a:.2f}' for a in accs]} "
          f"mean={np.mean(accs):.2f}%")


if __name__ == "__main__":
    main()
