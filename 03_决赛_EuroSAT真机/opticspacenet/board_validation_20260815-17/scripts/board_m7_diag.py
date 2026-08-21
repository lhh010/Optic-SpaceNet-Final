# -*- coding: utf-8 -*-
"""M7 真机逐层诊断: 对比每光层 hw vs ideal (应用校准), 定位误差累积层。
板上运行 (Python 3.6 + compass_sdk)。用法:
  sudo env PYTHONIOENCODING=utf-8 DIAG_IMAGES=200 python3 board_m7_diag.py
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/home/uisrc/j1")
os.environ["J1_WEIGHTS_DIR"] = "/home/uisrc/j1/weights_m7_5400"
os.environ["J1_FAKE"] = "0"
import run_j1_gazelle as R

def json_load(p):
    with open(p) as f:
        return json.load(f)

WD = "/home/uisrc/j1/weights_m7_5400"
NIMG = int(os.environ.get("DIAG_IMAGES", "200"))
CALIB_COL = os.environ.get("DIAG_CALIB_COL", "/home/uisrc/j1/calib_col_m7ccic.json")
CALIB_SC = os.environ.get("DIAG_CALIB_SC", "/home/uisrc/j1/calib_scalar_m7ccic.json")

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i + 2], w).astype(np.float64))
    return np.vstack(outs)

ws, meta = R.load_weights()
images = np.load(WD + "/test_images_j1.npy")[:NIMG]
labels = np.load(WD + "/test_labels_j1.npy")[:NIMG]
eps = float(meta.get("stem_bn_eps", 1e-5))

colc = json_load(CALIB_COL) if os.path.exists(CALIB_COL) else {}
scal = json_load(CALIB_SC) if os.path.exists(CALIB_SC) else {}

def dequant_apply(y, layer, x_scale, x_zp, w_int_2dT, w_scale, use_calib=True):
    """与 run_j1_gazelle.py optical_conv1x1/optical_fc 相同的反量化+校准路径。"""
    w_scale_arr = np.asarray(w_scale, dtype=np.float64).reshape(1, -1)
    col_sum = w_int_2dT.astype(np.float64).sum(axis=0, keepdims=True)
    off = x_zp * col_sum
    if use_calib:
        cc = None
        if layer in colc:
            cc = (np.asarray(colc[layer]["alpha"], dtype=np.float64).reshape(1, -1),
                  np.asarray(colc[layer]["beta"], dtype=np.float64).reshape(1, -1))
        if cc is not None:
            w_scale_arr = w_scale_arr / cc[0]
            off = cc[0] * off + cc[1]
        else:
            sc = scal.get(layer)
            if sc:
                y = (y - sc["beta"]) / sc["alpha"]
    return x_scale * w_scale_arr * (y - off)

def conv1x1_layer(x, name):
    B, C, H, W = x.shape
    m = B * H * W
    x_flat = x.transpose(0, 2, 3, 1).reshape(m, C)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    y_hw = engine(x_int, ws[name].T)
    y_id = x_int.astype(np.float64) @ ws[name].T.astype(np.float64)
    yh = dequant_apply(y_hw, name, x_scale, x_zp, ws[name].T, meta[name + "_scale"])
    yi = dequant_apply(y_id, name, x_scale, x_zp, ws[name].T, meta[name + "_scale"], use_calib=False)
    return yh.reshape(B, H, W, -1).transpose(0, 3, 1, 2), \
           yi.reshape(B, H, W, -1).transpose(0, 3, 1, 2), x_int

def fc_layer(x, name):
    x_int, x_scale, x_zp = R.quantize_act(x)
    y_hw = engine(x_int, ws[name].T)
    y_id = x_int.astype(np.float64) @ ws[name].T.astype(np.float64)
    yh = dequant_apply(y_hw, name, x_scale, x_zp, ws[name].T, meta[name + "_scale"])
    yi = dequant_apply(y_id, name, x_scale, x_zp, ws[name].T, meta[name + "_scale"], use_calib=False)
    if name + "_bias" in ws:
        yh = yh + np.asarray(ws[name + "_bias"]).reshape(1, -1)
        yi = yi + np.asarray(ws[name + "_bias"]).reshape(1, -1)
    return yh, yi

h = R.stem_forward(images, ws, meta)
for name in ["s1a", "s2a", "s2b", "s3a", "s3b"]:
    hh, hi, xi = conv1x1_layer(h, name)
    r_hw = (hh - hi)
    print("%s: |hw|mean=%.1f |ideal|mean=%.1f resid_std=%.1f rel=%.1f%% corr=%.4f" % (
        name, np.abs(hh).mean(), np.abs(hi).mean(), r_hw.std(),
        r_hw.std() / (np.abs(hi).mean() + 1e-9) * 100,
        np.corrcoef(hh.ravel(), hi.ravel())[0, 1]), flush=True)
    h = R.apply_bn(hh, ws[name + "_bn"], eps)
    h = R.relu(h)
    if name in ("s1a", "s2a", "s2b"):
        h = R.pool2d(h, 2)

g = h.mean(axis=(2, 3))
h1h, h1i = fc_layer(g, "h1")
print("h1: |hw|mean=%.1f |ideal|mean=%.1f resid_std=%.1f rel=%.1f%% corr=%.4f" % (
    np.abs(h1h).mean(), np.abs(h1i).mean(), (h1h - h1i).std(),
    (h1h - h1i).std() / (np.abs(h1i).mean() + 1e-9) * 100,
    np.corrcoef(h1h.ravel(), h1i.ravel())[0, 1]), flush=True)
z = R.relu(h1h)
h2h, h2i = fc_layer(z, "h2")
print("h2: |hw|mean=%.1f |ideal|mean=%.1f resid_std=%.1f rel=%.1f%% corr=%.4f" % (
    np.abs(h2h).mean(), np.abs(h2i).mean(), (h2h - h2i).std(),
    (h2h - h2i).std() / (np.abs(h2i).mean() + 1e-9) * 100,
    np.corrcoef(h2h.ravel(), h2i.ravel())[0, 1]), flush=True)

pred = np.argmax(h2h, 1)
acc = (pred == labels).mean() * 100
print("DIAG hw acc: %.2f%% (%d/%d)" % (acc, (pred == labels).sum(), NIMG), flush=True)

# 理想链 acc (同量化路径, 无噪声)
pred2 = np.argmax(h2i, 1)
acc2 = (pred2 == labels).mean() * 100
print("DIAG ideal acc: %.2f%%" % acc2, flush=True)
