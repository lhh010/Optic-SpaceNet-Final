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
        ws[layer + "_wf"] = w.astype(np.float64)
        wq, s = _quantize_w_per_channel(w)
        ws[layer] = wq; meta[layer + "_scale"] = s

    def conv3s2(layer, *parts):
        w = get(*parts, "weight")  # (C_out, C, 3, 3) -> (C_out, 9C)
        w2 = w.reshape(w.shape[0], -1)
        ws[layer + "_wf"] = w2.astype(np.float64)
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
    """逐层执行并返回可供前端解释的真实激活、结构和设计分析。"""
    import time
    cc = (calib_col or {}).get
    x = np.asarray(x, dtype=np.float64)
    eps = float(meta.get("stem_bn_eps", 1e-5))
    layers = []

    def _add(name, where, spec, analysis, act, t0):
        layers.append({
            "name": name,
            "where": where,
            "spec": spec,
            "analysis": analysis,
            "shape": list(act.shape[1:]),
            "act": act,
            "latency_s": time.perf_counter() - t0,
        })

    # Stem stays electronic. Both checkpoints use 3x3/s2; only M10 uses
    # MaxPool3x3/s2, while M9 uses MaxPool2x2.
    t0 = time.perf_counter()
    w = ws["stem"].astype(np.float64)
    h = _conv2d_np(x, w, stride=2, pad=1)
    h = _apply_bn(h, ws["stem_bn"], eps)
    h = _relu(h)
    pool3 = meta.get("stem_pool_mode", "max") == "max3"
    h = _pool3s2(h) if pool3 else _pool2d(h, 2)
    c0 = int(w.shape[0])
    pool_spec = "MaxPool3x3/s2" if pool3 else "MaxPool2x2/s2"
    stem_analysis = (
        "电子前端先提取局部纹理并把 64x64 压到 16x16；"
        + ("3x3 池化在进入光学主干前扩大局部感受野。"
           if pool3 else
           "0.75x 窄通道优先控制参数量和 MACs。")
    )
    _add(
        "stem", "electronic",
        "3x3/s2 Conv 3→%d + BN + ReLU + %s" % (c0, pool_spec),
        stem_analysis, h, t0)

    analyses = {
        "s1a": (
            "首个光学通道混合层：保持 16x16 空间分辨率并扩展通道；"
            "1x1 展平只沿通道做矩阵乘，更贴合 Gazelle 8x2 tile。"
        ),
        "s1ds": (
            "第一处可学习光学下采样：im2col 后执行 3x3/s2 矩阵乘，"
            "将 16x16 降到 8x8。它替代直接 MaxPool，避免噪声尖峰被 max 有偏放大。"
        ),
        "s2a": (
            "在 8x8 特征图上将通道翻倍，提升中层地物纹理与光谱组合能力；"
            "BN/ReLU 在电子域完成，矩阵乘在光学域完成。"
        ),
        "s2b": (
            "同分辨率的第二次 1x1 通道重组；实验表明 stage 内第二次混合"
            "是宽度收益生效的重要环节，而单纯继续堆深收益有限。"
        ),
        "s2ds": (
            "第二处 3x3/s2 光学下采样，把 8x8 压到 4x4，同时扩大感受野；"
            "该层同样使用 uint8 激活、per-channel int8 权重和逐输出列校准。"
        ),
        "s3a": (
            "在仅 4x4 的低空间成本区将通道再次翻倍，集中构建高层语义；"
            "这让模型把容量投向通道表达而不是大尺寸特征图。"
        ),
        "s3b": (
            "最终 1x1 光学特征整合，保持 4x4 尺寸；"
            "v8 QAT 按层注入实测列偏移、列增益与等效权重扰动以抑制误差累积。"
        ),
    }

    prev_c = int(h.shape[1])
    for name, kind in [
            ("s1a", "1x1"), ("s1ds", "3x3s2"), ("s2a", "1x1"),
            ("s2b", "1x1"), ("s2ds", "3x3s2"), ("s3a", "1x1"),
            ("s3b", "1x1")]:
        t0 = time.perf_counter()
        fn = optical_conv1x1 if kind == "1x1" else optical_conv3s2
        h = fn(backend, h, ws[name], meta[name + "_scale"], cc(name))
        h = _apply_bn(h, ws[name + "_bn"], eps)
        h = _relu(h)
        out_c = int(ws[name].shape[0])
        if kind == "1x1":
            spec = "1x1 Conv %d→%d + BN + ReLU [光学 MVM]" % (
                prev_c, out_c)
        else:
            spec = "3x3/s2 Conv %d→%d + BN + ReLU [im2col→光学 MVM]" % (
                prev_c, out_c)
        _add(name, "optical", spec, analyses[name], h, t0)
        prev_c = out_c

    # GAP is electronic; the current demo deployment maps both FC layers to
    # Gazelle unless HEAD_ELEC is explicitly selected.
    g = h.mean(axis=(2, 3))
    h1_analysis = (
        "GAP 在电子域把每个通道的 4x4 响应汇聚为一个向量；随后 FC1 "
        "在当前部署中映射到光学矩阵乘，并用 ReLU 形成分类嵌入。"
    )
    h2_analysis = (
        "最终光学全连接输出 EuroSAT 十类 logits，不再接 ReLU；"
        "这一层的扰动会直接改变类别间隔，因此逐列增益/偏移校准尤其关键。"
    )
    if head_elec:
        t0 = time.perf_counter()
        z = _relu(g @ ws["h1_wf"].T + ws["h1_bias"])
        _add("h1", "electronic",
             "GAP + Linear %d→%d + ReLU" % (
                 int(g.shape[1]), int(ws["h1_wf"].shape[0])),
             h1_analysis.replace("光学矩阵乘", "电子矩阵乘"), z, t0)
        t0 = time.perf_counter()
        logits = z @ ws["h2_wf"].T + ws["h2_bias"]
        _add("h2", "electronic",
             "Linear %d→10 [logits]" % int(ws["h2_wf"].shape[1]),
             h2_analysis.replace("最终光学全连接", "最终电子全连接"),
             logits, t0)
    else:
        t0 = time.perf_counter()
        z = _relu(optical_fc(
            backend, g, ws["h1"], meta["h1_scale"],
            cc("h1"), ws.get("h1_bias")))
        _add("h1", "optical",
             "GAP + Linear %d→%d + ReLU [光学 MVM]" % (
                 int(g.shape[1]), int(ws["h1"].shape[0])),
             h1_analysis, z, t0)
        t0 = time.perf_counter()
        logits = optical_fc(
            backend, z, ws["h2"], meta["h2_scale"],
            cc("h2"), ws.get("h2_bias"))
        _add("h2", "optical",
             "Linear %d→10 [光学 MVM · logits]" % int(ws["h2"].shape[1]),
             h2_analysis, logits, t0)
    return logits, layers



def forward_fp32_traced(x, ws, meta):
    """Original floating-point M9/M10 forward used as electric reference.

    This path does not quantize activations or weights. Layer boundaries match
    forward_traced so optical/hardware activations pair with electric results.
    """
    import time

    x = np.asarray(x, dtype=np.float64)
    eps = float(meta.get("stem_bn_eps", 1e-5))
    layers = []

    def add(name, where, spec, analysis, act, t0):
        layers.append({
            "name": name, "where": where, "spec": spec,
            "analysis": analysis, "shape": list(act.shape[1:]),
            "act": act, "latency_s": time.perf_counter() - t0,
        })

    t0 = time.perf_counter()
    h = _relu(_apply_bn(
        _conv2d_np(x, ws["stem"].astype(np.float64), stride=2, pad=1),
        ws["stem_bn"], eps))
    pool3 = meta.get("stem_pool_mode", "max") == "max3"
    h = _pool3s2(h) if pool3 else _pool2d(h, 2)
    c0 = int(h.shape[1])
    pool_spec = "MaxPool3x3/s2" if pool3 else "MaxPool2x2/s2"
    add("stem", "electronic",
        "3x3/s2 Conv 3→%d + BN + ReLU + %s [FP32 电计算]" %
        (c0, pool_spec),
        "原始浮点权重的电子前端；作为所有后端逐层对比的固定基准。",
        h, t0)

    analyses = {
        "s1a": "原始浮点 1x1 通道扩展，是同名光学 MVM 的逐元素参考。",
        "s1ds": "原始浮点 3x3/s2 下采样；im2col 展开与光学路径保持相同。",
        "s2a": "原始浮点中层通道扩展，用于量化与硬件误差的逐层归因。",
        "s2b": "原始浮点同分辨率通道重组，用于观察误差是否在 stage 内累积。",
        "s2ds": "原始浮点第二次可学习下采样，是光学 3x3 MVM 的对照。",
        "s3a": "原始浮点高层通道扩展，保留未经 int8 量化的语义激活。",
        "s3b": "原始浮点最终特征整合，作为进入分类头前的电计算基准。",
    }
    prev_c = c0
    for name, kind in [
            ("s1a", "1x1"), ("s1ds", "3x3s2"), ("s2a", "1x1"),
            ("s2b", "1x1"), ("s2ds", "3x3s2"), ("s3a", "1x1"),
            ("s3b", "1x1")]:
        t0 = time.perf_counter()
        wf = ws[name + "_wf"]
        if kind == "1x1":
            batch, channels, height, width = h.shape
            flat = h.transpose(0, 2, 3, 1).reshape(-1, channels)
            out = flat @ wf.T
            h = out.reshape(batch, height, width, -1).transpose(0, 3, 1, 2)
            kind_spec = "1x1"
        else:
            batch = h.shape[0]
            cols, oh, ow = _im2col_3x3s2(h)
            out = cols @ wf.T
            h = out.reshape(batch, oh, ow, -1).transpose(0, 3, 1, 2)
            kind_spec = "3x3/s2"
        h = _relu(_apply_bn(h, ws[name + "_bn"], eps))
        out_c = int(h.shape[1])
        add(name, "optical",
            "%s Conv %d→%d + BN + ReLU [FP32 电计算]" %
            (kind_spec, prev_c, out_c), analyses[name], h, t0)
        prev_c = out_c

    gap = h.mean(axis=(2, 3))
    t0 = time.perf_counter()
    z = _relu(gap @ ws["h1_wf"].astype(np.float64).T + ws["h1_bias"])
    add("h1", "optical",
        "GAP + Linear %d→%d + ReLU [FP32 电计算]" %
        (int(gap.shape[1]), int(z.shape[1])),
        "GAP 后使用原始浮点 FC1，作为光学分类嵌入的直接参照。", z, t0)
    t0 = time.perf_counter()
    logits = z @ ws["h2_wf"].astype(np.float64).T + ws["h2_bias"]
    add("h2", "optical",
        "Linear %d→10 [FP32 电计算 · logits]" % int(z.shape[1]),
        "原始浮点分类 logits；最终预测与类别概率的电计算基准。",
        logits, t0)
    return logits, layers
