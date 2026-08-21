# -*- coding: utf-8 -*-
"""calibrate_any_m1.py — M1 (变体 A, 7 光层) per-layer 标量校准 (含全部光层)。
env: M1_WEIGHTS_DIR, M1_CALIB_OUT
"""
import os, json, time, sys
import numpy as np
sys.path.insert(0, '/home/uisrc/j1')
os.environ.setdefault('M1_WEIGHTS_DIR', '/home/uisrc/j1/weights_m1_5400')
os.environ['M1_FAKE'] = '1'
WD = os.environ['M1_WEIGHTS_DIR']
OUT = os.environ.get('M1_CALIB_OUT', WD + '/../calib_m1.json')
import run_m1_gazelle as R

images = np.load(WD + '/test_images_j1.npy')[:64]
ws, meta = R.load_weights()
eps = float(meta.get('conv1_1_bn_eps', 1e-5))
acts = {}
h = R.conv3x3_np(images, ws['conv1_1'].astype(np.float64))
h = R.apply_bn(h, ws['conv1_1_bn'], eps); h = R.relu(h)
acts['conv1_2'] = h
h = R.optical_conv3x3(h, ws['conv1_2'], meta['conv1_2_scale']); h = R.apply_bn(h, ws['conv1_2_bn'], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts['conv2_1'] = h
h = R.optical_conv3x3(h, ws['conv2_1'], meta['conv2_1_scale']); h = R.apply_bn(h, ws['conv2_1_bn'], eps); h = R.relu(h)
acts['conv2_2'] = h
h = R.optical_conv3x3(h, ws['conv2_2'], meta['conv2_2_scale']); h = R.apply_bn(h, ws['conv2_2_bn'], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts['conv3_1'] = h
h = R.optical_conv3x3(h, ws['conv3_1'], meta['conv3_1_scale']); h = R.apply_bn(h, ws['conv3_1_bn'], eps); h = R.relu(h)
acts['conv3_2'] = h
h = R.optical_conv3x3(h, ws['conv3_2'], meta['conv3_2_scale']); h = R.apply_bn(h, ws['conv3_2_bn'], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts['fc1'] = h.reshape(h.shape[0], -1)
z = R.optical_fc_splitk(acts['fc1'], ws['fc1'], meta['fc1_scale']); z = R.relu(z)
acts['fc2'] = z

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i + 2], w).astype(np.float64))
    return np.vstack(outs)

calib = {}
for name in ["conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "fc1", "fc2"]:
    w = np.load(WD + '/%s_w.npy' % name)
    xf = acts[name]
    if name.startswith("fc"):
        x_flat = xf
        w2d = w
    else:
        cols, _, _ = R.im2col_3x3(xf)
        x_flat = cols
        w2d = w.reshape(w.shape[0], -1)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    # 行数采样控制 (conv 层 2048 行; fc 层全量, 最多 512)
    nrow = 2048 if not name.startswith("fc") else 512
    if x_int.shape[0] > nrow:
        idx = np.random.RandomState(0).choice(x_int.shape[0], nrow, replace=False)
        x_int = x_int[idx]
    ideal = x_int.astype(np.float64) @ w2d.T.astype(np.float64)
    if name == "fc1":
        ks = w2d.shape[1] // 2
        hw = np.zeros((x_int.shape[0], w2d.shape[0]), dtype=np.float64)
        for s in range(2):
            hw += engine(x_int[:, s*ks:(s+1)*ks].astype(np.uint8),
                         w2d[:, s*ks:(s+1)*ks].astype(np.int8).T)
    else:
        hw = engine(x_int.astype(np.uint8), w2d.astype(np.int8).T)
    # 标量 least squares: hw = alpha * ideal + beta (全元素)
    xf_ = ideal.ravel(); yf_ = hw.ravel()
    denom = np.dot(xf_, xf_)
    alpha = np.dot(xf_, yf_) / denom if denom > 1e-12 else 1.0
    beta = np.mean(yf_ - alpha * xf_)
    calib[name] = {"alpha": float(alpha), "beta": float(beta)}
    resid = hw - (alpha * ideal + beta)
    print("%s: alpha=%.5f beta=%.1f resid_std=%.1f ideal_rms=%.1f" % (
        name, alpha, beta, resid.std(), ideal.std()), flush=True)

with open(OUT, "w") as f:
    json.dump(calib, f, indent=1)
print("saved", OUT)
