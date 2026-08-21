# probe_dump.py — 逐层 dump (ideal, hw) pairs, 供残差结构分析
import os, time, sys
import numpy as np
sys.path.insert(0, "/home/uisrc/j1")
os.environ.setdefault("J1_WEIGHTS_DIR", "/home/uisrc/j1/weights_c2c")
os.environ["J1_FAKE"] = "1"
WD = os.environ["J1_WEIGHTS_DIR"]
NROWS = int(os.environ.get("PROBE_ROWS", "100000"))
import run_j1_gazelle as R

images = np.load(WD + "/test_images_j1.npy")[:64]
ws, meta = R.load_weights()
eps = float(meta.get("stem_bn_eps", 1e-5))
acts = {}
h = R.stem_forward(images, ws, meta)
acts["s1a"] = h
h = R.optical_conv1x1(h, ws["s1a"], meta["s1a_scale"]); h = R.apply_bn(h, ws["s1a_bn"], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts["s2a"] = h
h = R.optical_conv1x1(h, ws["s2a"], meta["s2a_scale"]); h = R.apply_bn(h, ws["s2a_bn"], eps); h = R.relu(h)
acts["s2b"] = h
h = R.optical_conv1x1(h, ws["s2b"], meta["s2b_scale"]); h = R.apply_bn(h, ws["s2b_bn"], eps); h = R.relu(h); h = R.pool2d(h, 2)
acts["s3a"] = h
h = R.optical_conv1x1(h, ws["s3a"], meta["s3a_scale"]); h = R.apply_bn(h, ws["s3a_bn"], eps); h = R.relu(h)
acts["s3b"] = h

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
compass_init(150); time.sleep(3)

def engine(x_u8, w_i8):
    w = w_i8.astype(np.int8)
    outs = []
    for i in range(0, x_u8.shape[0], 2):
        outs.append(compass_matmul(x_u8[i:i+2], w).astype(np.float64))
    return np.vstack(outs)

rng = np.random.RandomState(0)
for name in ["s1a", "s2a", "s2b", "s3a", "s3b"]:
    w = np.load(WD + f"/{name}_w.npy")
    xf = acts[name]
    B, C, H, W = xf.shape
    x_flat = xf.transpose(0, 2, 3, 1).reshape(B * H * W, C)
    x_int, x_scale, x_zp = R.quantize_act(x_flat)
    if x_int.shape[0] > NROWS:
        idx = rng.choice(x_int.shape[0], NROWS, replace=False)
        x_int = x_int[idx]
    ideal = x_int.astype(np.float64) @ w.T.astype(np.float64)
    t0 = time.time()
    hw = engine(x_int.astype(np.uint8), w.astype(np.int8).T)
    np.save(f"/home/uisrc/j1/probe_{name}_ideal.npy", ideal.astype(np.float32))
    np.save(f"/home/uisrc/j1/probe_{name}_hw.npy", hw.astype(np.float32))
    np.save(f"/home/uisrc/j1/probe_{name}_xint.npy", x_int.astype(np.uint8))
    resid = hw - ideal
    print(f"{name}: rows={x_int.shape[0]} cols={ideal.shape[1]} "
          f"resid_std={resid.std():.1f} dt={time.time()-t0:.0f}s", flush=True)
print("DONE")
