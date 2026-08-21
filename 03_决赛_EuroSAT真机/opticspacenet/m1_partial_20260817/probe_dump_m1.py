# -*- coding: utf-8 -*-
"""probe_dump_m1.py — M1 (变体 A, 7 光层) 分层 probe。
env: M1_WEIGHTS_DIR, PROBE_OUT_PREFIX, PROBE_IMAGES, PROBE_ROWS_<layer>
"""
import os, time, sys
import numpy as np
sys.path.insert(0, "/home/uisrc/j1")
os.environ.setdefault("M1_WEIGHTS_DIR", "/home/uisrc/j1/weights_m1_5400")
os.environ["M1_FAKE"] = "1"
WD = os.environ["M1_WEIGHTS_DIR"]
OUT_PREFIX = os.environ.get("PROBE_OUT_PREFIX", "probe_m1_")
NIMAGES = int(os.environ.get("PROBE_IMAGES", "64"))
ROWS = {
    "conv1_2": int(os.environ.get("PROBE_ROWS_CONV1_2", "4096")),
    "conv2_1": int(os.environ.get("PROBE_ROWS_CONV2_1", "1024")),
    "conv2_2": int(os.environ.get("PROBE_ROWS_CONV2_2", "1024")),
    "conv3_1": int(os.environ.get("PROBE_ROWS_CONV3_1", "512")),
    "conv3_2": int(os.environ.get("PROBE_ROWS_CONV3_2", "512")),
    "fc1": int(os.environ.get("PROBE_ROWS_FC1", "512")),
    "fc2": int(os.environ.get("PROBE_ROWS_FC2", "512")),
}
LAYERS = os.environ.get("PROBE_LAYERS",
                        "conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,fc1,fc2").split(",")
import run_m1_gazelle as R

images = np.load(os.path.join(WD, "test_images_j1.npy"))[:NIMAGES]
ws, meta = R.load_weights()
eps = float(meta.get("conv1_1_bn_eps", 1e-5))
acts = {}
h = R.conv3x3_np(images, ws["conv1_1"].astype(np.float64))
h = R.apply_bn(h, ws["conv1_1_bn"], eps); h = R.relu(h)
acts["conv1_2"] = h
h = R.optical_conv3x3(h, ws["conv1_2"], meta["conv1_2_scale"]); h = R.apply_bn(h, ws["conv1_2_bn"], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts["conv2_1"] = h
h = R.optical_conv3x3(h, ws["conv2_1"], meta["conv2_1_scale"]); h = R.apply_bn(h, ws["conv2_1_bn"], eps); h = R.relu(h)
acts["conv2_2"] = h
h = R.optical_conv3x3(h, ws["conv2_2"], meta["conv2_2_scale"]); h = R.apply_bn(h, ws["conv2_2_bn"], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts["conv3_1"] = h
h = R.optical_conv3x3(h, ws["conv3_1"], meta["conv3_1_scale"]); h = R.apply_bn(h, ws["conv3_1_bn"], eps); h = R.relu(h)
acts["conv3_2"] = h
h = R.optical_conv3x3(h, ws["conv3_2"], meta["conv3_2_scale"]); h = R.apply_bn(h, ws["conv3_2_bn"], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts["fc1"] = h.reshape(h.shape[0], -1)
z = R.optical_fc_splitk(acts["fc1"], ws["fc1"], meta["fc1_scale"]); z = R.relu(z)
acts["fc2"] = z
print("acts ready, NIMAGES=%d" % NIMAGES, flush=True)

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i + 2], w).astype(np.float64))
    return np.vstack(outs)

rng = np.random.RandomState(0)
for name in LAYERS:
    w = np.load(os.path.join(WD, "%s_w.npy" % name))
    xf = acts[name]
    if name.startswith("fc"):
        x_flat = xf
        w2d = w  # (C_out, k)
    else:
        cols, _, _ = R.im2col_3x3(xf)  # (m, 9C)
        x_flat = cols
        w2d = w.reshape(w.shape[0], -1)  # (C_out, 9C)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    if x_int.shape[0] > ROWS[name]:
        idx = rng.choice(x_int.shape[0], ROWS[name], replace=False)
        x_int = x_int[idx]
    ideal = x_int.astype(np.float64) @ w2d.T.astype(np.float64)
    t0 = time.time()
    if name == "fc1":
        # fc1 k=8192 拆块 (与 runner 一致)
        ks = w2d.shape[1] // 2
        hw = np.zeros((x_int.shape[0], w2d.shape[0]), dtype=np.float64)
        for s in range(2):
            hw += engine(x_int[:, s*ks:(s+1)*ks].astype(np.uint8),
                         w2d[:, s*ks:(s+1)*ks].astype(np.int8).T)
    else:
        hw = engine(x_int.astype(np.uint8), w2d.astype(np.int8).T)
    np.save("/home/uisrc/j1/%s%s_ideal.npy" % (OUT_PREFIX, name), ideal.astype(np.float32))
    np.save("/home/uisrc/j1/%s%s_hw.npy" % (OUT_PREFIX, name), hw.astype(np.float32))
    np.save("/home/uisrc/j1/%s%s_xint.npy" % (OUT_PREFIX, name), x_int.astype(np.uint8))
    resid = hw - ideal
    print("%s: rows=%d cols=%d resid_std=%.1f dt=%.0fs" % (
        name, x_int.shape[0], ideal.shape[1], resid.std(), time.time() - t0), flush=True)
print("DONE")
