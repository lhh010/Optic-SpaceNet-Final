# -*- coding: utf-8 -*-
"""run_m1_gazelle.py — Model 1 Baseline VGG (变体 A) 在 Gazelle 真机上的推理验证。
板上运行 (Python 3.6 + compass_sdk, 无 torch)。权重由 export_m1.py 转出。

结构 (变体 A):
  conv1_1 (电, 3x3 s1 p1) + BN + ReLU
  → conv1_2 光 (3x3, k=288) + BN + ReLU + MaxPool2
  → conv2_1 光 (k=288) + BN + ReLU
  → conv2_2 光 (k=576) + BN + ReLU + MaxPool2
  → conv3_1 光 (k=576) + BN + ReLU
  → conv3_2 光 (k=1152) + BN + ReLU + MaxPool2
  → flatten 8192 → fc1 光 (8192→256, k 拆 2×4096) + ReLU
  → fc2 光 (256→10)

环境变量 (compass_sdk 篡改 sys.argv, 禁止位置参数):
  M1_WEIGHTS_DIR  权重目录
  M1_FAKE=1       离线 np.matmul 模式
  M1_LIMIT / M1_OFFSET / M1_BATCH
  M1_CALIB        标量 calib json
  M1_CALIB_COL    逐列 calib json (优先)
  M1_LOGITS_OUT   logits 输出 npy
"""
import os
import time
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.environ.get("M1_WEIGHTS_DIR", os.path.join(HERE, "weights_m1_5400"))
FAKE = os.environ.get("M1_FAKE", "0") == "1"
LIMIT = int(os.environ.get("M1_LIMIT", "50"))
OFFSET = int(os.environ.get("M1_OFFSET", "0"))
BATCH = int(os.environ.get("M1_BATCH", "1"))
FC1_KSPLIT = 2  # fc1 k=8192 拆 2×4096 (compass weight 上限 4096)

if FAKE:
    def _compass_matmul(vec, wgt):
        return vec.astype(np.float64) @ wgt.astype(np.float64)
else:
    from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
    _compass_matmul = compass_matmul

CALIB_FILE = os.environ.get("M1_CALIB", os.path.join(WDIR, "..", "calib_m1.json"))
CALIB_COL_FILE = os.environ.get("M1_CALIB_COL", "")
_CALIB = None
_CALIB_COL = None


def _load_calib():
    global _CALIB
    if _CALIB is None:
        _CALIB = {}
        if not FAKE and os.path.exists(CALIB_FILE):
            _CALIB = json.load(open(CALIB_FILE))
    return _CALIB


def _load_calib_col():
    global _CALIB_COL
    if _CALIB_COL is None:
        _CALIB_COL = {}
        if not FAKE and CALIB_COL_FILE and os.path.exists(CALIB_COL_FILE):
            _CALIB_COL = json.load(open(CALIB_COL_FILE))
    return _CALIB_COL


def col_calib_params(layer):
    c = _load_calib_col().get(layer)
    if c is None:
        return None
    return (np.asarray(c["alpha"], dtype=np.float64).reshape(1, -1),
            np.asarray(c["beta"], dtype=np.float64).reshape(1, -1))


def calib_correct(y, layer):
    c = _load_calib().get(layer)
    if c is None:
        return y
    return (y - c["beta"]) / c["alpha"]


def optical_mm(x_u8, w_i8):
    """x: (m,k) uint8, w: (k,n) int8 → (m,n) float。FPGA m>=3 行回绕 → m<=2 tiling。"""
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        chunk = x_u8[i:i + 2]
        outs.append(_compass_matmul(chunk, w).astype(np.float64))
    return np.vstack(outs)


def load_weights():
    layers = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1",
              "conv3_2", "fc1", "fc2"]
    meta = json.load(open(os.path.join(WDIR, "meta.json")))
    ws = {}
    for name in layers:
        ws[name] = np.load(os.path.join(WDIR, "%s_w.npy" % name))
    for name in ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2"]:
        p = os.path.join(WDIR, "%s_bn.npy" % name)
        if os.path.exists(p):
            ws["%s_bn" % name] = np.load(p)
    return ws, meta


def apply_bn(x, bn, eps):
    bn_w, bn_b, bn_m, bn_v = bn[0], bn[1], bn[2], bn[3]
    shape = (1, -1, 1, 1)
    return (x - bn_m.reshape(shape)) / np.sqrt(bn_v.reshape(shape) + eps) \
        * bn_w.reshape(shape) + bn_b.reshape(shape)


def relu(x):
    return np.maximum(0.0, x)


def pool2d(x, k):
    B, C, H, W = x.shape
    oh, ow = H // k, W // k
    return x.reshape(B, C, oh, k, ow, k).max(axis=(3, 5))


def quantize_act(x):
    xmin, xmax = float(x.min()), float(x.max())
    span = max(xmax - xmin, 1e-8)
    scale = span / 255.0
    zp = int(round(-xmin / scale))
    zp = max(0, min(255, zp))
    x_int = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.uint8)
    return x_int, scale, zp


def conv3x3_np(x, w, stride=1, pad=1):
    """电计算 conv3x3 (仅 conv1_1 用)。"""
    B, C, H, W = x.shape
    oh = (H + 2 * pad - 3) // stride + 1
    ow = (W + 2 * pad - 3) // stride + 1
    out = np.zeros((B, w.shape[0], oh, ow), dtype=np.float64)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i:i + 3, j:j + 3]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out


def im2col_3x3(x, stride=1, pad=1):
    """3x3 stride1 pad1 im2col: (B,C,H,W) → cols (B*oh*ow, 9C)。"""
    B, C, H, W = x.shape
    oh = (H + 2 * pad - 3) // stride + 1
    ow = (W + 2 * pad - 3) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    cols = np.empty((B * oh * ow, C * 9), dtype=np.float64)
    r = 0
    for b in range(B):
        for i in range(oh):
            for j in range(ow):
                cols[r] = xp[b, :, i:i + 3, j:j + 3].reshape(-1)
                r += 1
    return cols, oh, ow


def _dequant(y, x_scale, x_zp, w_int_2dT, w_scale, cc):
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int_2dT.astype(np.float64).sum(axis=0, keepdims=True)
    off = x_zp * col_sum
    if cc is not None:
        w_scale_arr = w_scale_arr / cc[0]
        off = cc[0] * off + cc[1]
    return x_scale * w_scale_arr * (y - off)


def optical_conv3x3(x, w_int2d, w_scale, layer=None):
    """3x3 conv 光: im2col (m, 9C) → (m, 9C) @ (9C, C_out)。"""
    B, C, H, W = x.shape
    cols, oh, ow = im2col_3x3(x)
    x_int, x_scale, x_zp = quantize_act(cols)
    y = optical_mm(x_int, w_int2d.T)
    cc = col_calib_params(layer) if layer else None
    if cc is None and layer:
        y = calib_correct(y, layer)
    y = _dequant(y, x_scale, x_zp, w_int2d.T, w_scale, cc)
    return y.reshape(B, oh, ow, -1).transpose(0, 3, 1, 2)


def optical_fc_splitk(x, w_int, w_scale, layer=None, ksplit=FC1_KSPLIT):
    """FC 光, k 拆块累加 (fc1 k=8192)。x: (B, k)。"""
    B, k = x.shape
    x_int, x_scale, x_zp = quantize_act(x)
    ks = k // ksplit
    y = np.zeros((B, w_int.shape[0]), dtype=np.float64)
    for s in range(ksplit):
        xb = x_int[:, s * ks:(s + 1) * ks]
        wb = w_int[:, s * ks:(s + 1) * ks].T  # (ks, C_out)
        y += optical_mm(xb, wb)
    cc = col_calib_params(layer) if layer else None
    if cc is None and layer:
        y = calib_correct(y, layer)
    # 反量化: y = x_scale*w_scale*(y - x_zp*col_sum_total)
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int.astype(np.float64).sum(axis=1, keepdims=True).T
    off = x_zp * col_sum
    if cc is not None:
        w_scale_arr = w_scale_arr / cc[0]
        off = cc[0] * off + cc[1]
    return x_scale * w_scale_arr * (y - off)


def forward(x, ws, meta):
    eps = float(meta.get("conv1_1_bn_eps", 1e-5))
    # conv1_1 电
    h = conv3x3_np(x, ws["conv1_1"].astype(np.float64))
    h = apply_bn(h, ws["conv1_1_bn"], eps)
    h = relu(h)
    # conv1_2 光
    h = optical_conv3x3(h, ws["conv1_2"], meta["conv1_2_scale"], layer="conv1_2")
    h = apply_bn(h, ws["conv1_2_bn"], eps)
    h = relu(h)
    h = pool2d(h, 2)
    # conv2_1 光
    h = optical_conv3x3(h, ws["conv2_1"], meta["conv2_1_scale"], layer="conv2_1")
    h = apply_bn(h, ws["conv2_1_bn"], eps)
    h = relu(h)
    # conv2_2 光
    h = optical_conv3x3(h, ws["conv2_2"], meta["conv2_2_scale"], layer="conv2_2")
    h = apply_bn(h, ws["conv2_2_bn"], eps)
    h = relu(h)
    h = pool2d(h, 2)
    # conv3_1 光
    h = optical_conv3x3(h, ws["conv3_1"], meta["conv3_1_scale"], layer="conv3_1")
    h = apply_bn(h, ws["conv3_1_bn"], eps)
    h = relu(h)
    # conv3_2 光
    h = optical_conv3x3(h, ws["conv3_2"], meta["conv3_2_scale"], layer="conv3_2")
    h = apply_bn(h, ws["conv3_2_bn"], eps)
    h = relu(h)
    h = pool2d(h, 2)
    # flatten + fc1 光 (k 拆块)
    g = h.reshape(h.shape[0], -1)
    z = optical_fc_splitk(g, ws["fc1"], meta["fc1_scale"], layer="fc1")
    z = relu(z)
    # fc2 光
    z = optical_fc_splitk(z, ws["fc2"], meta["fc2_scale"], layer="fc2", ksplit=1)
    return z


def main():
    if not FAKE:
        compass_init(150)
    ws, meta = load_weights()
    images = np.load(os.path.join(WDIR, meta["images"]))
    labels = np.load(os.path.join(WDIR, meta["labels"]))

    n_test = min(LIMIT, len(labels) - OFFSET)
    print("m1 on HW: wdir=%s offset=%d n=%d, batch=%d, FAKE=%s"
          % (WDIR, OFFSET, n_test, BATCH, FAKE), flush=True)
    correct = 0
    t0 = time.time()
    all_logits = []
    for start in range(0, n_test, BATCH):
        end = min(start + BATCH, n_test)
        x = images[OFFSET + start:OFFSET + end]
        logits = forward(x, ws, meta)
        all_logits.append(logits)
        pred = np.argmax(logits, axis=1)
        correct += int(np.sum(pred == labels[OFFSET + start:OFFSET + end]))
        print("[%5d/%d] acc=%.2f%% elapsed=%.1fs"
              % (end, n_test, correct * 100.0 / end, time.time() - t0), flush=True)
    print("FINAL: %.2f%%" % (correct * 100.0 / n_test), flush=True)
    logits_out = os.environ.get("M1_LOGITS_OUT")
    if logits_out:
        np.save(logits_out, np.vstack(all_logits))
        print("logits saved: %s" % logits_out, flush=True)


if __name__ == "__main__":
    main()
