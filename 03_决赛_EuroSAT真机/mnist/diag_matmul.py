# -*- coding: utf-8 -*-
"""Diagnostic: verify compass_matmul layout & noise on REAL Gazelle hardware.

No positional CLI args (compass_sdk mutates sys.argv; compass_init parses it).

Tests:
  A. (1,8)@(8,2) with known small integers -> exact compare vs np.matmul
  B. (2,8)@(8,2) multi-row layout check
  C. (4,16)@(16,4) layout check
  D. MNIST layer-1: hardware vs numpy y1 error statistics (first batch)
"""
import os
import time

import numpy as np

from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, vec, wgt):
    v = vec.astype(np.uint8)
    w = wgt.astype(np.int8)
    hw = compass_matmul(v, w).astype(np.int64)
    ref = np.matmul(v.astype(np.int64), w.astype(np.int64))
    print("--- %s ---" % name, flush=True)
    print("vec shape=%s wgt shape=%s hw shape=%s" % (v.shape, w.shape, hw.shape), flush=True)
    print("hw :", hw.tolist(), flush=True)
    print("ref:", ref.tolist(), flush=True)
    print("diff:", (hw - ref).tolist(), flush=True)
    return hw, ref


def main():
    compass_init(150)

    # A. smallest unit: 1x8 @ 8x2
    vec = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    wgt = np.array([[1, -1], [1, -1], [1, -1], [1, -1],
                    [1, -1], [1, -1], [1, -1], [1, -1]], dtype=np.int32)
    check("A: 1x8@8x2", vec, wgt)

    # B. multi-row: 2x8 @ 8x2
    vec = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                    [8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.int32)
    wgt = np.array([[1, 1], [1, 1], [1, 1], [1, 1],
                    [1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.int32)
    check("B: 2x8@8x2", vec, wgt)

    # C. 4x16 @ 16x4
    vec = np.arange(4 * 16, dtype=np.int32).reshape(4, 16) % 16
    wgt = (np.arange(16 * 4, dtype=np.int32).reshape(16, 4) % 8) - 3
    check("C: 4x16@16x4", vec, wgt)

    # D. MNIST layer1: hardware vs numpy
    w1 = np.load(os.path.join(HERE, "w1_int4_dsq.npy"))[0].astype(np.int8)
    images = np.load(os.path.join(HERE, "test_images.npy"))
    q = np.load(os.path.join(HERE, "dsq_quant_params.npy"), allow_pickle=True).item()
    s_in = q["input_scale"]

    x = images[:50].reshape(50, 784) / 255.0
    x_int = np.clip(np.round(x / s_in), 0, 15).astype(np.uint8)

    y1_hw = compass_matmul(x_int, w1).astype(np.int64)
    y1_ref = np.matmul(x_int.astype(np.int64), w1.astype(np.int64))

    print("--- D: MNIST layer1 (50x784 @ 784x128) ---", flush=True)
    print("hw shape:", y1_hw.shape, "ref shape:", y1_ref.shape, flush=True)
    # tia factor: hw = raw * tia (255 or 25.5). Find best scalar fit.
    scale = np.sum(y1_hw * y1_ref) / np.sum(y1_ref * y1_ref)
    print("best linear fit hw ~= %.4f * ref" % scale, flush=True)
    err = y1_hw - scale * y1_ref
    print("scaled abs err: mean=%.1f std=%.1f max=%.1f" % (np.abs(err).mean(), err.std(), np.abs(err).max()), flush=True)
    print("ref abs: mean=%.1f max=%.1f" % (np.abs(y1_ref).mean(), np.abs(y1_ref).max()), flush=True)
    print("corr(row0):", np.corrcoef(y1_hw[0].astype(float), y1_ref[0].astype(float))[0, 1], flush=True)
    print("row0 hw :", y1_hw[0][:16].tolist(), flush=True)
    print("row0 ref:", y1_ref[0][:16].tolist(), flush=True)
    # argmax agreement
    agree = np.mean(np.argmax(y1_hw, axis=1) == np.argmax(y1_ref, axis=1))
    print("argmax agreement (layer1, per-image max channel): %.1f%%" % (agree * 100), flush=True)


if __name__ == "__main__":
    main()
