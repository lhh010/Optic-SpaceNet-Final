# -*- coding: utf-8 -*-
"""MNIST (STE / LSQ+ / DSQ) inference on REAL Gazelle optical hardware.

Runs on the Gazelle board via compass_sdk (no torch needed, Python 3.6 OK).
Also supports a local FAKE mode (pure numpy) for offline validation.

IMPORTANT: no positional CLI args — compass_sdk mutates sys.argv at import
(system_setup.py does sys.argv.extend(['--proj', ...])) and compass_init()
parses sys.argv[1:]; positional args would crash it.

Env control (pass through `sudo env ...`):
    MNIST_METHOD=dsq|ste|lsqplus   method (default dsq)
    MNIST_MODE=raw|scale|advance   how to feed the hardware (default scale)
    MNIST_FAKE=1                   offline: replace optical matmul with np.matmul
    MNIST_LIMIT=1000               test samples (default 1000)
    MNIST_BATCH=50                 batch size (default 50)

Methods' quant paths (faithful to testing/test_*_photonic.py):
  STE   : x = round(x*15); y1 = x@w1; h1=relu(y1); h1q=round(h1*scale_h1); ...
  LSQ+  : x = round(x/s_in)+zp_in; xs=x-zp_in; ws=w-zp_w(trunc);
          y = xs@ws*(s_in*s_w); hq = round(relu(y)/s_h)+zp_h; ...
  DSQ   : x = round(x/s_in); y = x@w*(s_in*s_w); hq = round(relu(y)/s_h); ...
"""
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD = os.environ.get("MNIST_METHOD", "dsq")
MODE = os.environ.get("MNIST_MODE", "scale")
FAKE = os.environ.get("MNIST_FAKE", "0") == "1"

if FAKE:
    def _compass_matmul(vec, wgt):
        return np.matmul(vec.astype(np.float64), wgt.astype(np.float64))
else:
    from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init
    _compass_matmul = compass_matmul


def optical_mm(x_int, w_int):
    """One optical matmul; x_int/w_int are small ints (uint4/int4 range).
    Returns value in 'x_int @ w_int' units (scale mode divides the x16 upscale).
    """
    x_u8 = x_int.astype(np.uint8)
    w_i8 = w_int.astype(np.int8)
    if MODE == "raw":
        return _compass_matmul(x_u8, w_i8)
    if MODE == "scale":
        x_up = (x_int.astype(np.int32) * 16).astype(np.uint8)
        w_up = (w_int.astype(np.int32) * 16).astype(np.int8)
        return _compass_matmul(x_up, w_up) / 256.0
    if MODE == "advance":
        from compass_sdk.fast_calibration.compass_lib import compass_matmul_advance
        return compass_matmul_advance(x_u8, w_i8).astype(np.float64)
    raise ValueError("unknown MNIST_MODE: %s" % MODE)


def relu_q(x, scale, zp=0.0, lo=0, hi=15):
    """ReLU then requantize to uint4 (round((relu(x))/scale)+zp clipped)."""
    h = np.maximum(0.0, x)
    return np.clip(np.round(h / scale) + zp, lo, hi).astype(np.int32)


def load_method(weights_dir):
    """Return (w1,w2,w3, quant_params_dict) for the selected method."""
    if METHOD == "dsq":
        w1 = np.load(os.path.join(weights_dir, "w1_int4_dsq.npy"))[0].astype(np.int32)
        w2 = np.load(os.path.join(weights_dir, "w2_int4_dsq.npy"))[0].astype(np.int32)
        w3 = np.load(os.path.join(weights_dir, "w3_int4_dsq.npy"))[0].astype(np.int32)
        q = np.load(os.path.join(weights_dir, "dsq_quant_params.npy"), allow_pickle=True).item()
        return w1, w2, w3, q
    if METHOD == "ste":
        w1 = np.load(os.path.join(weights_dir, "w1_int4.npy"))[0].astype(np.int32)
        w2 = np.load(os.path.join(weights_dir, "w2_int4.npy"))[0].astype(np.int32)
        w3 = np.load(os.path.join(weights_dir, "w3_int4.npy"))[0].astype(np.int32)
        q = np.load(os.path.join(weights_dir, "steq_quant_params.npy"), allow_pickle=True).item()
        return w1, w2, w3, q
    if METHOD == "lsqplus":
        w1 = np.load(os.path.join(weights_dir, "w1_int4_lsq_plus.npy"))[0].astype(np.int32)
        w2 = np.load(os.path.join(weights_dir, "w2_int4_lsq_plus.npy"))[0].astype(np.int32)
        w3 = np.load(os.path.join(weights_dir, "w3_int4_lsq_plus.npy"))[0].astype(np.int32)
        q = np.load(os.path.join(weights_dir, "lsq_plus_quant_params.npy"), allow_pickle=True).item()
        return w1, w2, w3, q
    raise ValueError("unknown MNIST_METHOD: %s" % METHOD)


def forward(x_int, w1, w2, w3, q):
    """3-layer optical forward; returns class logits (B,10)."""
    if METHOD == "dsq":
        s_in, s_w1, s_h1 = q["input_scale"], q["w1_scale"], q["h1_scale"]
        s_w2, s_h2, s_w3 = q["w2_scale"], q["h2_scale"], q["w3_scale"]
        y1 = optical_mm(x_int, w1) * (s_in * s_w1)
        h1 = relu_q(y1, s_h1)
        y2 = optical_mm(h1, w2) * (s_h1 * s_w2)
        h2 = relu_q(y2, s_h2)
        y3 = optical_mm(h2, w3) * (s_h2 * s_w3)
        return y3
    if METHOD == "ste":
        s1, s2 = q["scale_h1"], q["scale_h2"]
        y1 = optical_mm(x_int, w1)
        h1 = relu_q(y1, 1.0 / s1)          # h1*scale_h1 -> / (1/scale_h1)
        y2 = optical_mm(h1, w2)
        h2 = relu_q(y2, 1.0 / s2)
        y3 = optical_mm(h2, w3)
        return y3
    if METHOD == "lsqplus":
        s_in, zp_in = q["input_scale"], q["input_zp"]
        s_w1, zp_w1 = q["w1_scale"], q["w1_zp"]
        s_h1, zp_h1 = q["h1_scale"], q["h1_zp"]
        s_w2, zp_w2 = q["w2_scale"], q["w2_zp"]
        s_h2, zp_h2 = q["h2_scale"], q["h2_zp"]
        s_w3, zp_w3 = q["w3_scale"], q["w3_zp"]
        xs = x_int - zp_in
        w1s = (w1.astype(np.float64) - zp_w1).astype(np.int32)
        w2s = (w2.astype(np.float64) - zp_w2).astype(np.int32)
        w3s = (w3.astype(np.float64) - zp_w3).astype(np.int32)
        y1 = optical_mm(xs, w1s) * (s_in * s_w1)
        h1 = relu_q(y1, s_h1, zp_h1)
        h1s = h1 - zp_h1
        y2 = optical_mm(h1s, w2s) * (s_h1 * s_w2)
        h2 = relu_q(y2, s_h2, zp_h2)
        h2s = h2 - zp_h2
        y3 = optical_mm(h2s, w3s) * (s_h2 * s_w3)
        return y3
    raise ValueError("unknown method")


def forward_np(x_int, w1, w2, w3, q):
    """Pure-numpy reference on the same quant path (no optical, no tia)."""
    if METHOD == "dsq":
        s_in, s_w1, s_h1 = q["input_scale"], q["w1_scale"], q["h1_scale"]
        s_w2, s_h2, s_w3 = q["w2_scale"], q["h2_scale"], q["w3_scale"]
        y1 = np.matmul(x_int.astype(np.float64), w1.astype(np.float64)) * (s_in * s_w1)
        h1 = relu_q(y1, s_h1)
        y2 = np.matmul(h1.astype(np.float64), w2.astype(np.float64)) * (s_h1 * s_w2)
        h2 = relu_q(y2, s_h2)
        y3 = np.matmul(h2.astype(np.float64), w3.astype(np.float64)) * (s_h2 * s_w3)
        return y3
    if METHOD == "ste":
        s1, s2 = q["scale_h1"], q["scale_h2"]
        y1 = np.matmul(x_int.astype(np.float64), w1.astype(np.float64))
        h1 = relu_q(y1, 1.0 / s1)
        y2 = np.matmul(h1.astype(np.float64), w2.astype(np.float64))
        h2 = relu_q(y2, 1.0 / s2)
        y3 = np.matmul(h2.astype(np.float64), w3.astype(np.float64))
        return y3
    if METHOD == "lsqplus":
        s_in, zp_in = q["input_scale"], q["input_zp"]
        s_w1, zp_w1 = q["w1_scale"], q["w1_zp"]
        s_h1, zp_h1 = q["h1_scale"], q["h1_zp"]
        s_w2, zp_w2 = q["w2_scale"], q["w2_zp"]
        s_h2, zp_h2 = q["h2_scale"], q["h2_zp"]
        s_w3, zp_w3 = q["w3_scale"], q["w3_zp"]
        xs = x_int - zp_in
        w1s = (w1.astype(np.float64) - zp_w1).astype(np.int32)
        w2s = (w2.astype(np.float64) - zp_w2).astype(np.int32)
        w3s = (w3.astype(np.float64) - zp_w3).astype(np.int32)
        y1 = np.matmul(xs.astype(np.float64), w1s.astype(np.float64)) * (s_in * s_w1)
        h1 = relu_q(y1, s_h1, zp_h1)
        h1s = h1 - zp_h1
        y2 = np.matmul(h1s.astype(np.float64), w2s.astype(np.float64)) * (s_h1 * s_w2)
        h2 = relu_q(y2, s_h2, zp_h2)
        h2s = h2 - zp_h2
        y3 = np.matmul(h2s.astype(np.float64), w3s.astype(np.float64)) * (s_h2 * s_w3)
        return y3
    raise ValueError("unknown method")


def quantize_input(x_float, q):
    """float [0,1] images (B,784) -> per-method uint4 ints."""
    if METHOD == "dsq":
        return np.clip(np.round(x_float / q["input_scale"]), 0, 15).astype(np.int32)
    if METHOD == "ste":
        return np.clip(np.round(x_float * 15.0), 0, 15).astype(np.int32)
    if METHOD == "lsqplus":
        return np.clip(np.round(x_float / q["input_scale"]) + q["input_zp"], 0, 15).astype(np.int32)
    raise ValueError("unknown method")


def main():
    if not FAKE:
        compass_init(150)

    weights_dir = os.path.join(HERE, "weights")
    w1, w2, w3, q = load_method(weights_dir)

    images = np.load(os.path.join(HERE, "test_images.npy"))
    labels = np.load(os.path.join(HERE, "test_labels.npy"))

    limit = int(os.environ.get("MNIST_LIMIT", "1000"))
    batch_size = int(os.environ.get("MNIST_BATCH", "50"))
    n_test = min(limit, len(labels))
    tag = "FAKE" if FAKE else ("HW " + MODE)
    print("MNIST method=%s mode=%s (%s): %d samples, batch=%d"
          % (METHOD, MODE, tag, n_test, batch_size), flush=True)
    print("w1=%s w2=%s w3=%s" % (w1.shape, w2.shape, w3.shape), flush=True)

    correct = 0
    t0 = time.time()
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        x = images[start:end].reshape(end - start, 784) / 255.0
        x_int = quantize_input(x, q)
        y3 = forward(x_int, w1, w2, w3, q)
        pred = np.argmax(y3, axis=1)
        correct += int(np.sum(pred == labels[start:end]))
        done = end
        print("[%5d/%5d] acc=%.2f%%  elapsed=%.1fs"
              % (done, n_test, correct * 100.0 / done, time.time() - t0), flush=True)

    print("=" * 50)
    print("FINAL %s accuracy (%s, %d samples): %.2f%%"
          % (tag, METHOD, n_test, correct * 100.0 / n_test), flush=True)

    # NumPy reference on the same quant path (pure matmul, no optical/tia)
    correct_np = 0
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        x = images[start:end].reshape(end - start, 784) / 255.0
        x_int = quantize_input(x, q)
        y3 = forward_np(x_int, w1, w2, w3, q)
        correct_np += int(np.sum(np.argmax(y3, axis=1) == labels[start:end]))
    print("NumPy reference accuracy (same quant path): %.2f%%"
          % (correct_np * 100.0 / n_test), flush=True)
    print("%s vs reference gap: %.2f points"
          % (tag, (correct - correct_np) * 100.0 / n_test), flush=True)


if __name__ == "__main__":
    main()
