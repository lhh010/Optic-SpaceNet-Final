# -*- coding: utf-8 -*-
"""X0 ds3 变体 (w075ds3 / ds3pool3) 在 Gazelle 真机上的推理验证。

板上运行 (Python 3.6 + compass_sdk, 无 torch)。权重由 export_ds3.py 转出。
由 C1 patched run_j1_gazelle.py 适配: 新增 conv3s2 光计算下采样层 + stem max3 池化,
逐列 calib (DS3_CALIB_COL) 折叠逻辑与 C1 patch 完全一致。

结构:
  stem(3x3 s2 + BN + ReLU + pool[max|max3], 电计算)
  → stage1: 1x1 光 + BN + ReLU → s1ds conv3x3s2 光 + BN + ReLU
  → stage2: 1x1 光 ×2 (各 +BN+ReLU) → s2ds conv3x3s2 光 + BN + ReLU
  → stage3: 1x1 光 ×2 (各 +BN+ReLU)
  → GAP → head FC×2 (光, bias 反量化后加)

环境变量 (compass_sdk 篡改 sys.argv, 禁止位置参数):
  DS3_WEIGHTS_DIR  权重目录 (default weights_w075ds3)
  DS3_FAKE=1       离线 np.matmul 模式
  DS3_LIMIT        测试样本数 (default 500)
  DS3_OFFSET       分段跑批起点
  DS3_BATCH        批大小 (default 8)
  DS3_CALIB        标量 calib json (default ../calib_ds3.json)
  DS3_CALIB_COL    逐列 calib json (设置后优先于标量 calib)
  DS3_LOGITS_OUT   logits 输出 npy
  DS3_HEAD_ELEC=1  head 电计算 (float 权重, 不经光通路)
"""
import os
import time
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.environ.get("DS3_WEIGHTS_DIR", os.path.join(HERE, "weights_w075ds3"))
FAKE = os.environ.get("DS3_FAKE", "0") == "1"
LIMIT = int(os.environ.get("DS3_LIMIT", "500"))
OFFSET = int(os.environ.get("DS3_OFFSET", "0"))
BATCH = int(os.environ.get("DS3_BATCH", "8"))
HEAD_ELEC = os.environ.get("DS3_HEAD_ELEC", "0") == "1"

if FAKE:
    def _compass_matmul(vec, wgt):
        return vec.astype(np.float64) @ wgt.astype(np.float64)
else:
    from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
    _compass_matmul = compass_matmul

CALIB_FILE = os.environ.get("DS3_CALIB", os.path.join(WDIR, "..", "calib_ds3.json"))
_CALIB = None

# 逐列 (per-output-channel) 校准: calibrate_col.py 产物。设 DS3_CALIB_COL 后
# 优先于标量 calib, 折叠进反量化 (零额外算子); 不设置则回退标量路径, 行为不变。
CALIB_COL_FILE = os.environ.get("DS3_CALIB_COL", "")
_CALIB_COL = None


def _load_calib_col():
    global _CALIB_COL
    if _CALIB_COL is None:
        _CALIB_COL = {}
        if not FAKE and CALIB_COL_FILE and os.path.exists(CALIB_COL_FILE):
            _CALIB_COL = json.load(open(CALIB_COL_FILE))
    return _CALIB_COL


def col_calib_params(layer):
    """该层逐列校准 (alpha_vec, beta_vec), 形状 (1, C_out); 无则 None。"""
    c = _load_calib_col().get(layer)
    if c is None:
        return None
    a = np.asarray(c["alpha"], dtype=np.float64).reshape(1, -1)
    b = np.asarray(c["beta"], dtype=np.float64).reshape(1, -1)
    return a, b


def _load_calib():
    global _CALIB
    if _CALIB is None:
        _CALIB = {}
        if not FAKE and os.path.exists(CALIB_FILE):
            _CALIB = json.load(open(CALIB_FILE))
    return _CALIB


def calib_correct(y, layer):
    """应用 per-layer alpha/beta 校准 (仅真机; FAKE 无硬件增益)。"""
    c = _load_calib().get(layer)
    if c is None:
        return y
    return (y - c["beta"]) / c["alpha"]


def optical_mm(x_u8, w_i8):
    """x: (m,k) uint8, w: (k,n) int8 → (m,n) float (MAC units)。
    FPGA m>=3 行回绕 bug → m<=2 分块 tiling。"""
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        chunk = x_u8[i:i + 2]
        outs.append(_compass_matmul(chunk, w).astype(np.float64))
    return np.vstack(outs)


CONV_LAYERS = ["s1a", "s1ds", "s2a", "s2b", "s2ds", "s3a", "s3b"]


def load_weights():
    """加载全部层权重 (int8) + per-channel scale + BN 参数。"""
    layers = ["stem"] + CONV_LAYERS + ["h1", "h2"]
    meta = json.load(open(os.path.join(WDIR, "meta.json")))
    ws = {}
    for name in layers:
        ws[name] = np.load(os.path.join(WDIR, "%s_w.npy" % name))
    for name in ["stem"] + CONV_LAYERS:
        p = os.path.join(WDIR, "%s_bn.npy" % name)
        if os.path.exists(p):
            ws["%s_bn" % name] = np.load(p)
    for name in ["h1", "h2"]:
        for suffix in ["bias", "wf"]:
            p = os.path.join(WDIR, "%s_%s.npy" % (name, suffix))
            if os.path.exists(p):
                ws["%s_%s" % (name, suffix)] = np.load(p)
    return ws, meta


def apply_bn(x, bn, eps):
    """BN 推理: y = (x - mean)/sqrt(var+eps) * w + b。x: (B, C, H, W)。"""
    bn_w, bn_b, bn_m, bn_v = bn[0], bn[1], bn[2], bn[3]
    shape = (1, -1, 1, 1)
    return (x - bn_m.reshape(shape)) / np.sqrt(bn_v.reshape(shape) + eps) \
        * bn_w.reshape(shape) + bn_b.reshape(shape)


def main():
    if not FAKE:
        compass_init(150)
    ws, meta = load_weights()
    images = np.load(os.path.join(WDIR, meta["images"]))
    labels = np.load(os.path.join(WDIR, meta["labels"]))

    n_test = min(LIMIT, len(labels) - OFFSET)
    print("ds3 on HW: wdir=%s offset=%d n=%d, batch=%d, FAKE=%s, HEAD_ELEC=%s"
          % (WDIR, OFFSET, n_test, BATCH, FAKE, HEAD_ELEC), flush=True)

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
    logits_out = os.environ.get("DS3_LOGITS_OUT")
    if logits_out:
        np.save(logits_out, np.vstack(all_logits))
        print("logits saved: %s" % logits_out, flush=True)


def stem_forward(x, ws, meta):
    """stem (电计算): conv3x3 s2 + BN + ReLU + pool。
    pool 由 meta['stem_pool_mode'] 决定: max → MaxPool2d(2); max3 → MaxPool2d(3,s2,p1)。"""
    w = ws["stem"].astype(np.float64)
    bn = ws["stem_bn"]
    h = conv2d_np(x, w, 1.0, 1.0, stride=2, pad=1)
    bn_w, bn_b, bn_m, bn_v = bn[0], bn[1], bn[2], bn[3]
    eps = float(meta.get("stem_bn_eps", 1e-5))
    h = (h - bn_m.reshape(1, -1, 1, 1)) / np.sqrt(bn_v.reshape(1, -1, 1, 1) + eps) \
        * bn_w.reshape(1, -1, 1, 1) + bn_b.reshape(1, -1, 1, 1)
    h = np.maximum(0, h)
    if meta.get("stem_pool_mode", "max") == "max3":
        h = pool3s2(h)
    else:
        h = pool2d(h, 2)
    return h


def forward(x, ws, meta):
    """ds3 前向 (电 stem + 光 1x1/conv3s2 + 光 FC)。
    torch 结构: 每 conv 后跟 BN → ReLU (光计算层反量化输出上应用 BN)。"""
    eps = float(meta.get("stem_bn_eps", 1e-5))
    h = stem_forward(x, ws, meta)
    # stage1: 1x1 (光) + BN + ReLU → conv3s2 (光) + BN + ReLU
    h = optical_conv1x1(h, ws["s1a"], meta["s1a_scale"], layer="s1a")
    h = apply_bn(h, ws["s1a_bn"], eps)
    h = relu(h)
    h = optical_conv3s2(h, ws["s1ds"], meta["s1ds_scale"], layer="s1ds")
    h = apply_bn(h, ws["s1ds_bn"], eps)
    h = relu(h)
    # stage2: 1x1 ×2 (光) → conv3s2 (光)
    h = optical_conv1x1(h, ws["s2a"], meta["s2a_scale"], layer="s2a")
    h = apply_bn(h, ws["s2a_bn"], eps)
    h = relu(h)
    h = optical_conv1x1(h, ws["s2b"], meta["s2b_scale"], layer="s2b")
    h = apply_bn(h, ws["s2b_bn"], eps)
    h = relu(h)
    h = optical_conv3s2(h, ws["s2ds"], meta["s2ds_scale"], layer="s2ds")
    h = apply_bn(h, ws["s2ds_bn"], eps)
    h = relu(h)
    # stage3: 1x1 ×2 (光)
    h = optical_conv1x1(h, ws["s3a"], meta["s3a_scale"], layer="s3a")
    h = apply_bn(h, ws["s3a_bn"], eps)
    h = relu(h)
    h = optical_conv1x1(h, ws["s3b"], meta["s3b_scale"], layer="s3b")
    h = apply_bn(h, ws["s3b_bn"], eps)
    h = relu(h)
    # GAP
    g = h.mean(axis=(2, 3))
    if HEAD_ELEC and "h1_wf" in ws:
        z = relu(g @ ws["h1_wf"].T + ws["h1_bias"])
        return z @ ws["h2_wf"].T + ws["h2_bias"]
    z = optical_fc(g, ws["h1"], meta["h1_scale"], layer="h1",
                   bias=ws.get("h1_bias"))
    z = relu(z)
    z = optical_fc(z, ws["h2"], meta["h2_scale"], layer="h2",
                   bias=ws.get("h2_bias"))
    return z


def quantize_act(x):
    """per-tensor unsigned affine → uint8 + zp (与 QAT 训练一致)。"""
    xmin, xmax = float(x.min()), float(x.max())
    span = max(xmax - xmin, 1e-8)
    scale = span / 255.0
    zp = int(round(-xmin / scale))
    zp = max(0, min(255, zp))
    x_int = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.uint8)
    return x_int, scale, zp


def conv2d_np(x, w, w_scale, in_scale, stride=2, pad=1):
    """电计算 conv (numpy 实现, 仅 stem 用)。"""
    B, C, H, W = x.shape
    kh = kw = w.shape[2]
    oh = (H + 2 * pad - kh) // stride + 1
    ow = (W + 2 * pad - kw) // stride + 1
    out = np.zeros((B, w.shape[0], oh, ow), dtype=np.float64)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out


def im2col_3x3s2(x):
    """3x3 stride2 pad1 im2col: x (B,C,H,W) → cols (B*oh*ow, 9C)。
    patch 展平顺序 (C, kh, kw) row-major, 与 export 的 weight.reshape(C_out, 9C) 一致。
    行顺序: (b, i, j) — b 最外层, 与 reshape 回 (B,oh,ow,C_out) 一致。"""
    B, C, H, W = x.shape
    pad = 1
    oh = (H + 2 * pad - 3) // 2 + 1
    ow = (W + 2 * pad - 3) // 2 + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    cols = np.empty((B * oh * ow, C * 9), dtype=np.float64)
    r = 0
    for b in range(B):
        for i in range(oh):
            for j in range(ow):
                cols[r] = xp[b, :, 2*i:2*i+3, 2*j:2*j+3].reshape(-1)
                r += 1
    return cols, oh, ow


def pool2d(x, k):
    B, C, H, W = x.shape
    oh, ow = H // k, W // k
    return x.reshape(B, C, oh, k, ow, k).max(axis=(3, 5))


def pool3s2(x):
    """MaxPool2d(3, stride=2, padding=1) (torch 口径)。"""
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant",
                constant_values=-np.inf)
    oh = (H + 2 - 3) // 2 + 1
    ow = (W + 2 - 3) // 2 + 1
    out = np.empty((B, C, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = xp[:, :, 2*i:2*i+3, 2*j:2*j+3].max(axis=(2, 3))
    return out


def relu(x):
    return np.maximum(0.0, x)


def _dequant(y, x_scale, x_zp, w_int_2dT, w_scale, cc):
    """公共反量化 (+逐列 calib 折叠)。w_int_2dT: (k, C_out) int8。
    逐列: y_f = x_scale·(w_scale_c/α_c)·(y − β_c − α_c·x_zp·col_sum_c)"""
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int_2dT.astype(np.float64).sum(axis=0, keepdims=True)  # (1, C_out)
    off = x_zp * col_sum
    if cc is not None:
        w_scale_arr = w_scale_arr / cc[0]
        off = cc[0] * off + cc[1]
    return x_scale * w_scale_arr * (y - off)


def optical_conv1x1(x, w_int, w_scale, layer=None):
    """1x1 conv 走光计算: im2col (m, C) → (m, C) @ (C, C_out)。
    反量化: x≈scale·(x_int−zp) → y = x_scale·w_scale·y_int − x_scale·zp·w_scale·col_sum"""
    B, C, H, W = x.shape
    m = B * H * W
    x_flat = x.transpose(0, 2, 3, 1).reshape(m, C)
    x_int, x_scale, x_zp = quantize_act(x_flat)
    y = optical_mm(x_int, w_int.T)  # (m, C_out)
    cc = col_calib_params(layer) if layer else None
    if cc is None and layer:
        y = calib_correct(y, layer)
    y = _dequant(y, x_scale, x_zp, w_int.T, w_scale, cc)
    return y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)


def optical_conv3s2(x, w_int2d, w_scale, layer=None):
    """3x3 stride2 conv 走光计算: im2col (m, 9C) → (m, 9C) @ (9C, C_out)。
    w_int2d: (C_out, 9C) int8 (export 时 reshape); 量化/反量化/calib 与 conv1x1 同口径。"""
    B, C, H, W = x.shape
    cols, oh, ow = im2col_3x3s2(x)  # (m, 9C)
    x_int, x_scale, x_zp = quantize_act(cols)
    y = optical_mm(x_int, w_int2d.T)  # (m, C_out)
    cc = col_calib_params(layer) if layer else None
    if cc is None and layer:
        y = calib_correct(y, layer)
    y = _dequant(y, x_scale, x_zp, w_int2d.T, w_scale, cc)
    return y.reshape(B, oh, ow, -1).transpose(0, 3, 1, 2)


def optical_fc(x, w_int, w_scale, layer=None, bias=None):
    """FC 走光计算: x (B, C) uint8 → (B, C) @ (C, C_out)。bias 反量化后加。"""
    x_int, x_scale, x_zp = quantize_act(x)
    y = optical_mm(x_int, w_int.T)  # (B, C_out)
    cc = col_calib_params(layer) if layer else None
    if cc is None and layer:
        y = calib_correct(y, layer)
    y = _dequant(y, x_scale, x_zp, w_int.T, w_scale, cc)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64).reshape(1, -1)
    return y


if __name__ == "__main__":
    main()
