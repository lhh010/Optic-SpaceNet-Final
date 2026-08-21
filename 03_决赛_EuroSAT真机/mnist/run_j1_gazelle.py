# -*- coding: utf-8 -*-
"""J1 (MiniVGG-GAP 变体) 在 Gazelle 真机上的推理验证。

板上运行 (Python 3.6 + compass_sdk, 无 torch)。权重由 torch 训练产物转出。

结构:
  stem(3x3 s2 + pool, 电计算) → stage1(1x1) → stage2(1x1 ×2) → stage3(1x1 ×2)
  → GAP → head(FC ×3) — 全层光计算 (除 stem)

环境变量 (compass_sdk 篡改 sys.argv, 禁止位置参数):
  J1_WEIGHTS_DIR   权重目录 (default weights_j1)
  J1_FAKE=1        离线 np.matmul 模式
  J1_LIMIT         测试样本数 (default 500)
  J1_BATCH         批大小 (default 8)
  J1_DATA          测试数据 npy (default test_images_j1.npy)

输出格式: {layer}_w{n}.npy (int8 权重, per-channel scale 存 {layer}_scale.npy)
          {layer}_in_scale.npy / _in_zp.npy (输入量化参数)
"""
import os
import time
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.environ.get("J1_WEIGHTS_DIR", os.path.join(HERE, "weights_j1"))
FAKE = os.environ.get("J1_FAKE", "0") == "1"
LIMIT = int(os.environ.get("J1_LIMIT", "500"))
OFFSET = int(os.environ.get("J1_OFFSET", "0"))  # 分段跑批起点 (对抗跑批漂移)
BATCH = int(os.environ.get("J1_BATCH", "8"))
# head (h1/h2 FC) 电计算模式: float 权重 + bias, logits 层不吃光噪声
HEAD_ELEC = os.environ.get("J1_HEAD_ELEC", "0") == "1"
# test-time BN: 用当前 batch 统计量替代存储的 running stats (对抗漂移 OOD)
TTBN = os.environ.get("J1_TTBN", "0") == "1"

if FAKE:
    def _compass_matmul(vec, wgt):
        return vec.astype(np.float64) @ wgt.astype(np.float64)
else:
    from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
    _compass_matmul = compass_matmul

# 真机 per-layer 校准: hw = alpha*ideal + beta (crossval 拟合 + calibrate_j1.py 实测定标)
# 反量化前修正: ideal = (hw - beta) / alpha
CALIB_FILE = os.environ.get("J1_CALIB", os.path.join(WDIR, "..", "calib_j1.json"))
_CALIB = None


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


def load_weights():
    """加载 J1 全部层权重 (int8) + per-channel scale + 输入量化参数。"""
    layers = ["stem", "s1a", "s2a", "s2b", "s3a", "s3b", "h1", "h2"]
    meta = json.load(open(os.path.join(WDIR, "meta.json")))
    ws = {}
    for name in layers:
        w = np.load(os.path.join(WDIR, f"{name}_w.npy"))
        ws[name] = w
    # 各层 BN 参数 (stem + 光计算层后): {name}_bn.npy = [w, b, running_mean, running_var]
    for name in ["stem", "s1a", "s2a", "s2b", "s3a", "s3b"]:
        p = os.path.join(WDIR, f"{name}_bn.npy")
        if os.path.exists(p):
            ws[f"{name}_bn"] = np.load(p)
    # head bias (h1/h2 Linear bias=True) 与 float 权重 (HEAD_ELEC 用)
    for name in ["h1", "h2"]:
        for suffix in ["bias", "wf"]:
            p = os.path.join(WDIR, f"{name}_{suffix}.npy")
            if os.path.exists(p):
                ws[f"{name}_{suffix}"] = np.load(p)
    return ws, meta


def apply_bn(x, bn, eps):
    """BN 推理: y = (x - mean)/sqrt(var+eps) * w + b。x: (B, C, H, W)。
    TTBN 模式: mean/var 取当前 batch 统计量 (self-normalize 漂移引起的激活平移),
    否则用训练时存储的 running_mean/var。"""
    bn_w, bn_b = bn[0], bn[1]
    if TTBN:
        bn_m = x.mean(axis=(0, 2, 3))
        bn_v = x.var(axis=(0, 2, 3))
    else:
        bn_m, bn_v = bn[2], bn[3]
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
    print(f"J1 on HW: offset={OFFSET} n={n_test}, batch={BATCH}, FAKE={FAKE}, "
          f"HEAD_ELEC={HEAD_ELEC}", flush=True)

    correct = 0
    t0 = time.time()
    all_logits = []
    for start in range(0, n_test, BATCH):
        end = min(start + BATCH, n_test)
        x = images[OFFSET + start:OFFSET + end]  # (B,3,64,64) float [0,1] 归一化后
        logits = forward(x, ws, meta)
        all_logits.append(logits)
        pred = np.argmax(logits, axis=1)
        correct += int(np.sum(pred == labels[OFFSET + start:OFFSET + end]))
        print(f"[{end:5d}/{n_test}] acc={correct*100.0/end:.2f}% elapsed={time.time()-t0:.1f}s",
              flush=True)
    print(f"FINAL: {correct*100.0/n_test:.2f}%", flush=True)
    logits_out = os.environ.get("J1_LOGITS_OUT")
    if logits_out:
        np.save(logits_out, np.vstack(all_logits))
        print(f"logits saved: {logits_out}", flush=True)


def stem_forward(x, ws, meta):
    """stem (电计算): conv3x3 s2 + BN + ReLU + pool2。ws['stem_bn'] 为 BN 参数。"""
    w = ws["stem"].astype(np.float64)
    bn = ws["stem_bn"]  # (w, b, mean, var)
    h = conv2d_np(x, w, 1.0, 1.0, stride=2, pad=1)
    bn_w, bn_b, bn_m, bn_v = bn[0], bn[1], bn[2], bn[3]
    eps = float(meta.get("stem_bn_eps", 1e-5))
    h = (h - bn_m.reshape(1, -1, 1, 1)) / np.sqrt(bn_v.reshape(1, -1, 1, 1) + eps) \
        * bn_w.reshape(1, -1, 1, 1) + bn_b.reshape(1, -1, 1, 1)
    h = np.maximum(0, h)
    h = pool2d(h, 2)
    return h


def forward(x, ws, meta):
    """J1 前向 (电 stem + 光 1x1 convs + 光 FC)。ws 为已量化 int8 权重。
    torch 结构: 每 conv 后跟 BN → ReLU (光计算层反量化输出上应用 BN)。"""
    eps = float(meta.get("stem_bn_eps", 1e-5))
    B = x.shape[0]
    h = stem_forward(x, ws, meta)  # (B,16,16,16)
    # stage1: 1x1 conv 16->32 (光) + BN + ReLU + pool
    h = optical_conv1x1(h, ws["s1a"], meta["s1a_scale"], layer="s1a")  # (B,16,16,32)
    h = apply_bn(h, ws["s1a_bn"], eps)
    h = relu(h)
    h = pool2d(h, 2)  # (B,8,8,32)
    # stage2: [1x1 32->64 + BN + ReLU] [1x1 64->64 + BN + ReLU] pool
    h = optical_conv1x1(h, ws["s2a"], meta["s2a_scale"], layer="s2a")
    h = apply_bn(h, ws["s2a_bn"], eps)
    h = relu(h)
    h = optical_conv1x1(h, ws["s2b"], meta["s2b_scale"], layer="s2b")
    h = apply_bn(h, ws["s2b_bn"], eps)
    h = relu(h)
    h = pool2d(h, 2)  # (B,4,4,64)
    # stage3: [1x1 64->128 + BN + ReLU] [1x1 128->128 + BN + ReLU]
    h = optical_conv1x1(h, ws["s3a"], meta["s3a_scale"], layer="s3a")
    h = apply_bn(h, ws["s3a_bn"], eps)
    h = relu(h)
    h = optical_conv1x1(h, ws["s3b"], meta["s3b_scale"], layer="s3b")
    h = apply_bn(h, ws["s3b_bn"], eps)
    h = relu(h)  # (B,4,4,128)
    # GAP → (B,128)
    g = h.mean(axis=(2, 3))
    if HEAD_ELEC and "h1_wf" in ws:
        # head 电计算: float 权重 + bias, 不经光通路 (logits 不吃光噪声)
        z = relu(g @ ws["h1_wf"].T + ws["h1_bias"])
        return z @ ws["h2_wf"].T + ws["h2_bias"]
    # head: FC 128->128 (光) ReLU → 128->10
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


def quantize_w_per_channel(w):
    """per-channel signed int8 + scale (与 QAT 一致)。"""
    amax = np.abs(w).max(axis=tuple(range(1, w.ndim)), keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    w_int = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return w_int, scale


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


def pool2d(x, k):
    B, C, H, W = x.shape
    oh, ow = H // k, W // k
    return x.reshape(B, C, oh, k, ow, k).max(axis=(3, 5))


def relu(x):
    return np.maximum(0.0, x)


def optical_conv1x1(x, w_int, w_scale, layer=None):
    """1x1 conv 走光计算: im2col (m, C) → (m, C) @ (C, C_out)。
    w_int: (C_out, C) 已 int8 量化; w_scale: (C_out,) per-channel scale。
    反量化: x≈scale·(x_int−zp) → y = x_scale·w_scale·y_int − x_scale·zp·w_scale·col_sum"""
    B, C, H, W = x.shape
    m = B * H * W
    x_flat = x.transpose(0, 2, 3, 1).reshape(m, C)
    x_int, x_scale, x_zp = quantize_act(x_flat)
    y = optical_mm(x_int, w_int.T)  # (m, C_out)
    if layer:
        y = calib_correct(y, layer)
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int.T.astype(np.float64).sum(axis=0, keepdims=True)  # (1, C_out)
    y = x_scale * w_scale_arr * y - x_scale * x_zp * w_scale_arr * col_sum
    return y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)


def optical_fc(x, w_int, w_scale, layer=None, bias=None):
    """FC 走光计算: x (B, C) uint8 → (B, C) @ (C, C_out)。
    w_int: (C_out, C) 已 int8 量化; w_scale: (C_out,) per-channel scale。
    反量化公式同 optical_conv1x1。bias (C_out,) 反量化后加上。"""
    x_int, x_scale, x_zp = quantize_act(x)
    y = optical_mm(x_int, w_int.T)  # (B, C_out)
    if layer:
        y = calib_correct(y, layer)
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int.astype(np.float64).sum(axis=1, keepdims=True).T  # (1, C_out)
    y = x_scale * w_scale_arr * y - x_scale * x_zp * w_scale_arr * col_sum
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64).reshape(1, -1)
    return y


if __name__ == "__main__":
    main()
