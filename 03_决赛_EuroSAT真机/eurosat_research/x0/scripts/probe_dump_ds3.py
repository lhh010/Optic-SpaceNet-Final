# probe_dump_ds3.py — ds3 变体逐层 dump (ideal, hw) pairs (C2-board-ds3, 2026-08-09)
# 由 probe_dump_c3d.py 适配: 层清单含 conv3s2 下采样层 (s1ds/s2ds, k=9C im2col)。
# acts 链走 FAKE 理想路径 (不设 calib); 权重目录/输出前缀/逐层行数全部环境变量化。
# 板上 Python 3.6 / numpy 1.13 兼容。
import os, time, sys
import numpy as np
sys.path.insert(0, "/home/uisrc/j1")
os.environ.setdefault("DS3_WEIGHTS_DIR", "/home/uisrc/j1/weights_w075ds3")
os.environ["DS3_FAKE"] = "1"
WD = os.environ["DS3_WEIGHTS_DIR"]
OUT_PREFIX = os.environ.get("PROBE_OUT_PREFIX", "probe_ds3_")
NIMAGES = int(os.environ.get("PROBE_IMAGES", "512"))
ROWS = {
    "s1a":  int(os.environ.get("PROBE_ROWS_S1A",  "16384")),
    "s1ds": int(os.environ.get("PROBE_ROWS_S1DS", "16384")),
    "s2a":  int(os.environ.get("PROBE_ROWS_S2A",  "4096")),
    "s2b":  int(os.environ.get("PROBE_ROWS_S2B",  "4096")),
    "s2ds": int(os.environ.get("PROBE_ROWS_S2DS", "4096")),
    "s3a":  int(os.environ.get("PROBE_ROWS_S3A",  "1024")),
    "s3b":  int(os.environ.get("PROBE_ROWS_S3B",  "8192")),
}
LAYERS = os.environ.get("PROBE_LAYERS",
                        "s1a,s1ds,s2a,s2b,s2ds,s3a,s3b").split(",")
import run_ds3_gazelle as R

images = np.load(os.path.join(WD, "test_images_j1.npy"))[:NIMAGES]
ws, meta = R.load_weights()
eps = float(meta.get("stem_bn_eps", 1e-5))
acts = {}
h = R.stem_forward(images, ws, meta)
acts["s1a"] = h
h = R.optical_conv1x1(h, ws["s1a"], meta["s1a_scale"]); h = R.apply_bn(h, ws["s1a_bn"], eps); h = R.relu(h)
acts["s1ds"] = h
h = R.optical_conv3s2(h, ws["s1ds"], meta["s1ds_scale"]); h = R.apply_bn(h, ws["s1ds_bn"], eps); h = R.relu(h)
acts["s2a"] = h
h = R.optical_conv1x1(h, ws["s2a"], meta["s2a_scale"]); h = R.apply_bn(h, ws["s2a_bn"], eps); h = R.relu(h)
acts["s2b"] = h
h = R.optical_conv1x1(h, ws["s2b"], meta["s2b_scale"]); h = R.apply_bn(h, ws["s2b_bn"], eps); h = R.relu(h)
acts["s2ds"] = h
h = R.optical_conv3s2(h, ws["s2ds"], meta["s2ds_scale"]); h = R.apply_bn(h, ws["s2ds_bn"], eps); h = R.relu(h)
acts["s3a"] = h
h = R.optical_conv1x1(h, ws["s3a"], meta["s3a_scale"]); h = R.apply_bn(h, ws["s3a_bn"], eps); h = R.relu(h)
acts["s3b"] = h
print("acts ready, NIMAGES=%d stem_pool=%s" % (NIMAGES, meta.get("stem_pool_mode")), flush=True)

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i+2], w).astype(np.float64))
    return np.vstack(outs)

rng = np.random.RandomState(0)
for name in LAYERS:
    w = np.load(os.path.join(WD, "%s_w.npy" % name))
    xf = acts[name]
    if name in ("s1ds", "s2ds"):
        # conv3s2 层: im2col (m, 9C), 与 runner 光计算路径完全同口径
        cols, oh, ow = R.im2col_3x3s2(xf)
        x_flat = cols
    else:
        B, C, H, W = xf.shape
        x_flat = xf.transpose(0, 2, 3, 1).reshape(B * H * W, C)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    nrows = ROWS[name]
    if x_int.shape[0] > nrows:
        idx = rng.choice(x_int.shape[0], nrows, replace=False)
        x_int = x_int[idx]
    ideal = x_int.astype(np.float64) @ w.T.astype(np.float64)
    t0 = time.time()
    hw = engine(x_int.astype(np.uint8), w.astype(np.int8).T)
    np.save("/home/uisrc/j1/%s%s_ideal.npy" % (OUT_PREFIX, name), ideal.astype(np.float32))
    np.save("/home/uisrc/j1/%s%s_hw.npy" % (OUT_PREFIX, name), hw.astype(np.float32))
    np.save("/home/uisrc/j1/%s%s_xint.npy" % (OUT_PREFIX, name), x_int.astype(np.uint8))
    resid = hw - ideal
    print("%s: rows=%d cols=%d resid_std=%.1f dt=%.0fs" %
          (name, x_int.shape[0], ideal.shape[1], resid.std(), time.time() - t0), flush=True)
print("DONE")
