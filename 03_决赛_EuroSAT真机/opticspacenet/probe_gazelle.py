# -*- coding: utf-8 -*-
"""Precise calibration probe: find exact scale/offset of compass_matmul.

No positional CLI args (compass_sdk mutates sys.argv).
Prints raw hw vs exact for known small cases, then does linear fit
(y = a*x + b) on random full-range matmuls of the shapes Optic-SpaceNet
will use, and reports a, b, residual std, and correlation.
"""
import time
import numpy as np

from compass_sdk.fast_calibration.compass_lib import compass_init, compass_matmul
from compass_sdk.fast_calibration.utils import global_var

compass_init(150)
tia_gain = global_var.get_value('tia_gain_scale_factor')
print("tia_gain_scale_factor =", tia_gain, flush=True)


def show(name, vec, wgt):
    v = vec.astype(np.uint8)
    w = wgt.astype(np.int8)
    hw = compass_matmul(v, w)
    ref = np.matmul(v.astype(np.int64), w.astype(np.int64))
    print("--- %s ---" % name, flush=True)
    print("hw :", hw.tolist(), flush=True)
    print("ref:", ref.tolist(), flush=True)
    return hw, ref


# A: known small, positive+negative weights
show("A 1x8@8x2 pos/neg",
     np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32),
     np.array([[1, -1], [1, -1], [1, -1], [1, -1],
               [1, -1], [1, -1], [1, -1], [1, -1]], dtype=np.int32))

# B: all-positive weights (no negative arm)
show("B 1x8@8x2 all-pos",
     np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32),
     np.ones((8, 2), dtype=np.int32))

# C: larger random fit: (512,32)@(32,16)
rng = np.random.RandomState(7)
for name, m, k, n in [("fit (512,32)@(32,16)", 512, 32, 16),
                      ("fit (128,64)@(64,32)", 128, 64, 32),
                      ("fit (8,1024)@(1024,256)", 8, 1024, 256),
                      ("fit (8,256)@(256,10)", 8, 256, 10)]:
    v = rng.randint(0, 256, size=(m, k)).astype(np.uint8)
    w = rng.randint(-128, 128, size=(k, n)).astype(np.int8)
    t0 = time.time()
    hw = compass_matmul(v, w)
    dt = time.time() - t0
    ref = np.matmul(v.astype(np.int64), w.astype(np.int64)).astype(np.float64)
    hf = hw.astype(np.float64)
    # least squares: y = a*x + b
    x = ref.ravel()
    y = hf.ravel()
    a = np.dot(x, y) / np.dot(x, x)
    b = np.mean(y) - a * np.mean(x)
    resid = y - (a * x + b)
    corr = np.corrcoef(x, y)[0, 1]
    print("--- %s ---" % name, flush=True)
    print("hw/ref best scale a=%.4f b=%.1f | resid std=%.1f corr=%.6f time=%.3fs"
          % (a, b, resid.std(), corr, dt), flush=True)
    print("ref abs: mean=%.1f max=%.1f | hw abs: mean=%.1f max=%.1f"
          % (np.abs(x).mean(), np.abs(x).max(), np.abs(y).mean(), np.abs(y).max()), flush=True)

print("PROBE DONE", flush=True)
