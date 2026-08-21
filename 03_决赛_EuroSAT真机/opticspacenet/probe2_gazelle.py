# -*- coding: utf-8 -*-
"""Per-channel scale/offset characterization of compass_matmul.

Measures, for the actual layer shapes Optic-SpaceNet will use:
  - per-output-channel least-squares fit  hw ~= a_j * ref_j + b_j
  - spread of a_j and b_j across channels
  - repeatability: same call twice, diff
  - dependence of b on k (tile count)

No positional CLI args (compass_sdk mutates sys.argv).
"""
import time
import numpy as np

from compass_sdk.fast_calibration.compass_lib import compass_init, compass_matmul

compass_init(150)


def fit_per_col(name, m, k, n, reps=2):
    rng = np.random.RandomState(123)
    v = rng.randint(0, 256, size=(m, k)).astype(np.uint8)
    w = rng.randint(-128, 128, size=(k, n)).astype(np.int8)
    ref = np.matmul(v.astype(np.int64), w.astype(np.int64)).astype(np.float64)
    outs = []
    for r in range(reps):
        t0 = time.time()
        outs.append(compass_matmul(v, w).astype(np.float64))
        dt = time.time() - t0
    hw = outs[-1]
    # per-column fit
    a_j, b_j = [], []
    for j in range(n):
        x = ref[:, j]
        y = hw[:, j]
        a = np.dot(x, y) / np.dot(x, x)
        b = np.mean(y) - a * np.mean(x)
        a_j.append(a)
        b_j.append(b)
    a_j = np.array(a_j)
    b_j = np.array(b_j)
    resid = hw - (a_j * ref + b_j)
    repeat_diff = np.abs(outs[0] - outs[1]).max() if reps > 1 else 0.0
    print("--- %s ---" % name, flush=True)
    print("ref abs mean=%.1f max=%.1f | hw abs mean=%.1f" %
          (np.abs(ref).mean(), np.abs(ref).max(), np.abs(hw).mean()), flush=True)
    print("per-col a: mean=%.4f std=%.4f min=%.4f max=%.4f" %
          (a_j.mean(), a_j.std(), a_j.min(), a_j.max()), flush=True)
    print("per-col b: mean=%.1f std=%.1f min=%.1f max=%.1f" %
          (b_j.mean(), b_j.std(), b_j.min(), b_j.max()), flush=True)
    print("resid after per-col fit: mean=%.1f std=%.1f max=%.1f" %
          (np.abs(resid).mean(), resid.std(), np.abs(resid).max()), flush=True)
    print("repeatability max|call1-call2|=%.1f time=%.3fs" % (repeat_diff, dt), flush=True)
    return a_j, b_j, resid


fit_per_col("(512,32)@(32,16)", 512, 32, 16)
fit_per_col("(512,32)@(32,16) rep", 512, 32, 16)
fit_per_col("(128,64)@(64,32)", 128, 64, 32)
fit_per_col("(8,1024)@(1024,256)", 8, 1024, 256)
fit_per_col("(8,256)@(256,10)", 8, 256, 10)
fit_per_col("(64,16)@(16,8)", 64, 16, 8)

print("PROBE2 DONE", flush=True)
