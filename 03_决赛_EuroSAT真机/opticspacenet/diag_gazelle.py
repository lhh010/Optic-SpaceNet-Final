# -*- coding: utf-8 -*-
"""Gazelle hardware diagnostic for Optic-SpaceNet migration.

Runs on the board (Python 3.6, no torch). No positional args (compass_sdk
mutates sys.argv). Verifies:
  1. compass_init works
  2. tia_gain_scale_factor value (255 or 25.5)
  3. small 1x8@8x2 matmul accuracy (vs exact np.matmul)
  4. larger (m,k)@(k,n) matmul accuracy, incl. k=32/64/1024 shapes
  5. multi-row batches

Env: none required.
"""
import time
import numpy as np

from compass_sdk.fast_calibration.compass_lib import compass_init, compass_matmul
from compass_sdk.fast_calibration.utils import global_var

compass_init(150)

tia_gain = global_var.get_value('tia_gain_scale_factor')
print("tia_gain_scale_factor =", tia_gain, flush=True)


def rel_err(a, b):
    b = np.asarray(b, dtype=np.float64)
    denom = np.abs(b).max()
    return np.abs(np.asarray(a, dtype=np.float64) - b).max() / denom if denom > 0 else 0.0


def test(name, vec, wgt):
    # vec: uint8 (m,k), wgt: int8 (k,n)
    t0 = time.time()
    out = compass_matmul(vec.astype(np.uint8), wgt.astype(np.int8))
    dt = time.time() - t0
    exact = np.matmul(vec.astype(np.float64), wgt.astype(np.float64))
    # hardware returns raw*tia_gain; divide to get MAC units
    mac = out.astype(np.float64) / tia_gain
    err = np.abs(mac - exact)
    print("%-24s %s x %s -> %s  max|err|=%.3f  rel=%.4f  time=%.3fs"
          % (name, vec.shape, wgt.shape, out.shape,
             err.max(), rel_err(mac, exact), dt), flush=True)
    return mac, exact


# 1) tiny official-style 1x8 @ 8x2
rng = np.random.RandomState(0)
v = rng.randint(0, 256, size=(1, 8))
w = rng.randint(-128, 128, size=(8, 2))
test("1x8@8x2", v, w)

# 2) stage1-like: (1024,32) @ (32,16)
v = rng.randint(0, 256, size=(1024, 32))
w = rng.randint(-128, 128, size=(32, 16))
test("(1024,32)@(32,16)", v, w)

# 3) stage2-like: (256,64) @ (64,32)
v = rng.randint(0, 256, size=(256, 64))
w = rng.randint(-128, 128, size=(64, 32))
test("(256,64)@(64,32)", v, w)

# 4) fc1-like: (4,1024) @ (1024,256)
v = rng.randint(0, 256, size=(4, 1024))
w = rng.randint(-128, 128, size=(1024, 256))
test("(4,1024)@(1024,256)", v, w)

# 5) fc2-like: (4,256) @ (256,10)
v = rng.randint(0, 256, size=(4, 256))
w = rng.randint(-128, 128, size=(256, 10))
test("(4,256)@(256,10)", v, w)

# 6) large m chunk: (2048,32) @ (32,16) -> check m beyond 1024
v = rng.randint(0, 256, size=(2048, 32))
w = rng.randint(-128, 128, size=(32, 16))
try:
    test("(2048,32)@(32,16)", v, w)
except Exception as e:
    print("(2048,32) failed:", e, flush=True)

print("DIAG DONE", flush=True)
