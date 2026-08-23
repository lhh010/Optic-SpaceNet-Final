# -*- coding: utf-8 -*-
"""M9/M10 (ds3 family) numpy forward — browser demo backend.

数值语义逐行镜像 03_决赛/eurosat_research/x0/scripts/run_ds3_gazelle.py
(路径 B 板端 runner, M10 真机全量 95.33% 的 canonical 链路):
  - 激活 per-tensor uint8 仿射量化 (quantize_act)
  - 权重 per-channel int8 + scale (export_ds3.py 同款, 加载 .pth 时量化)
  - 光算层: im2col -> backend.matmul_2d -> 反量化 (+可选逐列 calib 折叠)
  - stem 电计算 (conv3x3 s2 + BN + ReLU + max/max3 pool); head FC 光算或电算

backend 协议: matmul_2d(x_u8 (m,k) uint8, w_i8 (k,n) int8) -> (m,n) float64。
用 gazelle_engine.HttpBackend(真机, chunk_rows=2 即 FPGA m<=2 tiling) 或
gazelle_engine.NumpyBackend(离线干净参考)。"""
import json
import os

import numpy as np
import torch


def _quantize_w_per_channel(w_np):
    """per-channel signed int8 + scale (与 export_ds3.py/QAT 训练一致)。"""
    amax = np.abs(w_np).max(axis=tuple(range(1, w_np.ndim)), keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    w_int = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return w_int, scale.reshape(-1)


def load_ds3(pth_path, stem_pool_mode):
    """从 v8 QAT ckpt 加载并量化 -> (ws, meta)。

    stem_pool_mode: 'max3' (M10 ds3pool3) | 'max' (M9 w075ds3)。
    层命名/键路径与 export_ds3.py 完全一致。"""
    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    sd = {k: v.numpy() for k, v in state.items()}

    def get(*parts):
        k = ".".join(parts)
        return sd[k] if k in sd else None

    ws, meta = {}, {"stem_pool_mode": stem_pool_mode,
                    "stem_bn_eps": 1e-5, "source": pth_path}

    def bn_pack(layer, *parts):
        w = get(*parts, "weight"); b = get(*parts, "bias")
        rm = get(*parts, "running_mean"); rv = get(*parts, "running_var")
        assert w is not None, "%s BN missing" % (parts,)
        ws[layer + "_bn"] = np.stack([w, b, rm, rv])

    def conv1x1(layer, *parts):
        w = get(*parts, "weight").squeeze()  # (C_out, C_in)
        wq, s = _quantize_w_per_channel(w)
        ws[layer] = wq; meta[layer + "_scale"] = s

    def conv3s2(layer, *parts):
        w = get(*parts, "weight")  # (C_out, C, 3, 3) -> (C_out, 9C)
        w2 = w.reshape(w.shape[0], -1)
        wq, s = _quantize_w_per_channel(w2)
        ws[layer] = wq; meta[layer + "_scale"] = s

    # stem 电计算 float
    ws["stem"] = get("stem", "0", "weight")
    bn_pack("stem", "stem", "1")
    # stage1: 1x1 + conv3s2
    conv1x1("s1a", "stage1", "0"); bn_pack("s1a", "stage1", "1")
    conv3s2("s1ds", "stage1", "3", "0"); bn_pack("s1ds", "stage1", "3", "1")
    # stage2: 1x1 x2 + conv3s2
    conv1x1("s2a", "stage2", "0"); bn_pack("s2a", "stage2", "1")
    conv1x1("s2b", "stage2", "3"); bn_pack("s2b", "stage2", "4")
    conv3s2("s2ds", "stage2", "6", "0"); bn_pack("s2ds", "stage2", "6", "1")
    # stage3: 1x1 x2
    conv1x1("s3a", "stage3", "0"); bn_pack("s3a", "stage3", "1")
    conv1x1("s3b", "stage3", "3"); bn_pack("s3b", "stage3", "4")
    # head FC (光算 int8 + float wf + bias)
    for tag, parts in (("h1", ("head", "2")), ("h2", ("head", "4"))):
        wgt = get(*parts, "weight"); b = get(*parts, "bias")
        assert wgt is not None and b is not None, "%s head missing" % (parts,)
        ws[tag + "_wf"] = wgt; ws[tag + "_bias"] = b
        wq, s = _quantize_w_per_channel(wgt)
        ws[tag] = wq; meta[tag + "_scale"] = s

    meta["channels"] = [int(ws["stem"].shape[0]), int(ws["s1a"].shape[0]),
                        int(ws["s2a"].shape[0]), int(ws["s3a"].shape[0])]
    return ws, meta


def load_calib_col(path):
    """逐列 calib json (calibrate_col.py 产物): layer -> {alpha, beta}."""
    if not path or not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    out = {}
    for layer, c in raw.items():
        a = np.asarray(c["alpha"], dtype=np.float64).reshape(1, -1)
        b = np.asarray(c["beta"], dtype=np.float64).reshape(1, -1)
        out[layer] = (a, b)
    return out


# ---------------- forward (镜像 run_ds3_gazelle.py) ----------------

def quantize_act(x):
    xmin, xmax = float(x.min()), float(x.max())
    span = max(xmax - xmin, 1e-8)
    scale = span / 255.0
    zp = int(round(-xmin / scale))
    zp = max(0, min(255, zp))
    x_int = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.uint8)
    return x_int, scale, zp


def _mm(backend, x_int, w_int):
    return backend.matmul_2d(x_int, w_int)


def _dequant(y, x_scale, x_zp, w_int_2dT, w_scale, cc):
    """反量化 (+逐列 calib 折叠): y_f = x_scale·(w_scale_c/α_c)·(y − β_c − α_c·x_zp·col_sum_c)"""
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int_2dT.astype(np.float64).sum(axis=0, keepdims=True)
    off = x_zp * col_sum
    if cc is not None:
        w_scale_arr = w_scale_arr / cc[0]
        off = cc[0] * off + cc[1]
    return x_scale * w_scale_arr * (y - off)


def optical_conv1x1(backend, x, w_int, w_scale, cc=None):
    B, C, H, W = x.shape
    m = B * H * W
    x_flat = x.transpose(0, 2, 3, 1).reshape(m, C)
    x_int, x_scale, x_zp = quantize_act(x_flat)
    y = _mm(backend, x_int, w_int.T)
    y = _dequant(y, x_scale, x_zp, w_int.T, w_scale, cc)
    return y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)


def _im2col_3x3s2(x):
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
                cols[r] = xp[b, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3].reshape(-1)
                r += 1
    return cols, oh, ow


def optical_conv3s2(backend, x, w_int2d, w_scale, cc=None):
    B, C, H, W = x.shape
    cols, oh, ow = _im2col_3x3s2(x)
    x_int, x_scale, x_zp = quantize_act(cols)
    y = _mm(backend, x_int, w_int2d.T)
    y = _dequant(y, x_scale, x_zp, w_int2d.T, w_scale, cc)
    return y.reshape(B, oh, ow, -1).transpose(0, 3, 1, 2)


def optical_fc(backend, x, w_int, w_scale, cc=None, bias=None):
    x_int, x_scale, x_zp = quantize_act(x)
    y = _mm(backend, x_int, w_int.T)
    y = _dequant(y, x_scale, x_zp, w_int.T, w_scale, cc)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64).reshape(1, -1)
    return y


def _apply_bn(x, bn, eps=1e-5):
    bn_w, bn_b, bn_m, bn_v = bn[0], bn[1], bn[2], bn[3]
    shape = (1, -1, 1, 1)
    return (x - bn_m.reshape(shape)) / np.sqrt(bn_v.reshape(shape) + eps) \
        * bn_w.reshape(shape) + bn_b.reshape(shape)


def _relu(x):
    return np.maximum(0.0, x)


def _conv2d_np(x, w, stride=2, pad=1):
    B, C, H, W = x.shape
    kh = kw = w.shape[2]
    oh = (H + 2 * pad - kh) // stride + 1
    ow = (W + 2 * pad - kw) // stride + 1
    out = np.zeros((B, w.shape[0], oh, ow), dtype=np.float64)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out


def _pool2d(x, k):
    B, C, H, W = x.shape
    oh, ow = H // k, W // k
    return x.reshape(B, C, oh, k, ow, k).max(axis=(3, 5))


def _pool3s2(x):
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant",
                constant_values=-np.inf)
    oh = (H + 2 - 3) // 2 + 1
    ow = (W + 2 - 3) // 2 + 1
    out = np.empty((B, C, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = xp[:, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3].max(axis=(2, 3))
    return out


def _stem_forward(x, ws, meta):
    w = ws["stem"].astype(np.float64)
    h = _conv2d_np(x, w, stride=2, pad=1)
    h = _apply_bn(h, ws["stem_bn"], float(meta.get("stem_bn_eps", 1e-5)))
    h = _relu(h)
    if meta.get("stem_pool_mode", "max") == "max3":
        return _pool3s2(h)
    return _pool2d(h, 2)


def forward(x, ws, meta, backend, calib_col=None, head_elec=False):
    """x: (B,3,64,64) float64 (ImageNet 归一化后) -> logits (B,10)。"""
    cc = (calib_col or {}).get
    x = np.asarray(x, dtype=np.float64)
    h = _stem_forward(x, ws, meta)
    h = _relu(_apply_bn(optical_conv1x1(
        backend, h, ws["s1a"], meta["s1a_scale"], cc("s1a")), ws["s1a_bn"]))
    h = _relu(_apply_bn(optical_conv3s2(
        backend, h, ws["s1ds"], meta["s1ds_scale"], cc("s1ds")), ws["s1ds_bn"]))
    h = _relu(_apply_bn(optical_conv1x1(
        backend, h, ws["s2a"], meta["s2a_scale"], cc("s2a")), ws["s2a_bn"]))
    h = _relu(_apply_bn(optical_conv1x1(
        backend, h, ws["s2b"], meta["s2b_scale"], cc("s2b")), ws["s2b_bn"]))
    h = _relu(_apply_bn(optical_conv3s2(
        backend, h, ws["s2ds"], meta["s2ds_scale"], cc("s2ds")), ws["s2ds_bn"]))
    h = _relu(_apply_bn(optical_conv1x1(
        backend, h, ws["s3a"], meta["s3a_scale"], cc("s3a")), ws["s3a_bn"]))
    h = _relu(_apply_bn(optical_conv1x1(
        backend, h, ws["s3b"], meta["s3b_scale"], cc("s3b")), ws["s3b_bn"]))
    g = h.mean(axis=(2, 3))  # GAP
    if head_elec:
        z = _relu(g @ ws["h1_wf"].T + ws["h1_bias"])
        return z @ ws["h2_wf"].T + ws["h2_bias"]
    z = _relu(optical_fc(backend, g, ws["h1"], meta["h1_scale"],
                         cc("h1"), ws.get("h1_bias")))
    return optical_fc(backend, z, ws["h2"], meta["h2_scale"],
                      cc("h2"), ws.get("h2_bias"))

# ---------------- forward_traced (逐层激活, 供演示前端光|电对比) ----------------

def forward_traced(x, ws, meta, backend, calib_col=None, head_elec=False):
    """逐层执行, 返回 (logits (B,10), layers)。

    layers: [{"name","where","spec","shape","latency_s","act"(np.ndarray), ...}]
    act 为该层输出激活 (BN+ReLU+pool 之后, 与 Model 3 前端展示语义一致);
    stem 为电层, 其余为光层 (backend.matmul_2d)。
    """
    import time
    cc = (calib_col or {}).get
    x = np.asarray(x, dtype=np.float64)
    eps = float(meta.get("stem_bn_eps", 1e-5))
    layers = []

    def _add(name, where, spec, act, t0):
        layers.append({"name": name, "where": where, "spec": spec,
                       "shape": list(act.shape[1:]), "act": act,
                       "latency_s": time.perf_counter() - t0})

    # stem (电, k5 s2 + BN + ReLU + pool)
    t0 = time.perf_counter()
    w = ws["stem"].astype(np.float64)
    h = _conv2d_np(x, w, stride=2, pad=1)
    h = _apply_bn(h, ws["stem_bn"], eps)
    h = _relu(h)
    h = _pool3s2(h) if meta.get("stem_pool_mode", "max") == "max3" else _pool2d(h, 2)
    _add("stem", "electronic", "Conv2d 3→C0 5×5/s2 + BN + ReLU + MaxPool", h, t0)

    # 7 光层
    for name, kind in [("s1a", "1x1"), ("s1ds", "3x3s2"), ("s2a", "1x1"),
                       ("s2b", "1x1"), ("s2ds", "3x3s2"), ("s3a", "1x1"),
                       ("s3b", "1x1")]:
        t0 = time.perf_counter()
        fn = optical_conv1x1 if kind == "1x1" else optical_conv3s2
        h = fn(backend, h, ws[name], meta[name + "_scale"], cc(name))
        h = _apply_bn(h, ws[name + "_bn"], eps)
        h = _relu(h)
        _add(name, "optical", "%s conv" % kind, h, t0)

    g = h.mean(axis=(2, 3))  # GAP
    if head_elec:
        z = _relu(g @ ws["h1_wf"].T + ws["h1_bias"])
        z = z @ ws["h2_wf"].T + ws["h2_bias"]
        layers.append({"name": "h1", "where": "electronic", "spec": "Linear(GAP→128)",
                       "shape": [], "act": None, "latency_s": 0.0})
    else:
        t0 = time.perf_counter()
        z = _relu(optical_fc(backend, g, ws["h1"], meta["h1_scale"],
                             cc("h1"), ws.get("h1_bias")))
        _add("h1", "optical", "Linear GAP→128", z, t0)
        t0 = time.perf_counter()
        z = optical_fc(backend, z, ws["h2"], meta["h2_scale"],
                       cc("h2"), ws.get("h2_bias"))
        _add("h2", "optical", "Linear 128→10 · logits", z, t0)
    return z, layers
