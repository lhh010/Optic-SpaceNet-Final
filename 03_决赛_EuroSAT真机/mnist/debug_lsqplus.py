# -*- coding: utf-8 -*-
"""Debug: compare forward() (optical path, FAKE) vs forward_np() layer by layer."""
import os
import numpy as np

os.environ.setdefault("MNIST_METHOD", "lsqplus")
os.environ.setdefault("MNIST_FAKE", "1")

import run_mnist_gazelle as R

HERE = os.path.dirname(os.path.abspath(__file__))
w1, w2, w3, q = R.load_method(os.path.join(HERE, "weights"))
images = np.load(os.path.join(HERE, "test_images.npy"))
labels = np.load(os.path.join(HERE, "test_labels.npy"))

x = images[:50].reshape(50, 784) / 255.0
x_int = R.quantize_input(x, q)
print("x_int range:", x_int.min(), x_int.max())

# layer-by-layer comparison
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
print("w1s range:", w1s.min(), w1s.max(), "w2s:", w2s.min(), w2s.max(), "w3s:", w3s.min(), w3s.max())

# optical (FAKE scale)
R.MODE = "scale"
y1_o = R.optical_mm(xs, w1s) * (s_in * s_w1)
h1_o = R.relu_q(y1_o, s_h1, zp_h1)
h1s_o = h1_o - zp_h1
y2_o = R.optical_mm(h1s_o, w2s) * (s_h1 * s_w2)
h2_o = R.relu_q(y2_o, s_h2, zp_h2)
h2s_o = h2_o - zp_h2
y3_o = R.optical_mm(h2s_o, w3s) * (s_h2 * s_w3)

# numpy
y1_n = np.matmul(xs.astype(np.float64), w1s.astype(np.float64)) * (s_in * s_w1)
h1_n = R.relu_q(y1_n, s_h1, zp_h1)
h1s_n = h1_n - zp_h1
y2_n = np.matmul(h1s_n.astype(np.float64), w2s.astype(np.float64)) * (s_h1 * s_w2)
h2_n = R.relu_q(y2_n, s_h2, zp_h2)
h2s_n = h2_n - zp_h2
y3_n = np.matmul(h2s_n.astype(np.float64), w3s.astype(np.float64)) * (s_h2 * s_w3)

for name, a, b in [("y1", y1_o, y1_n), ("h1", h1_o, h1_n), ("h1s", h1s_o, h1s_n),
                   ("y2", y2_o, y2_n), ("h2", h2_o, h2_n), ("h2s", h2s_o, h2s_n),
                   ("y3", y3_o, y3_n)]:
    same = np.allclose(a, b, atol=1e-6)
    print("%-4s shapes %s maxdiff=%.6f same=%s" % (name, a.shape,
          np.abs(np.asarray(a, float) - np.asarray(b, float)).max(), same))
    if not same:
        idx = np.argmax(np.abs(np.asarray(a, float) - np.asarray(b, float)))
        print("   first diff at flat idx", idx, "a=", np.asarray(a).reshape(-1)[idx], "b=", np.asarray(b).reshape(-1)[idx])

pa = np.argmax(y3_o, axis=1)
pn = np.argmax(y3_n, axis=1)
print("argmax agree:", (pa == pn).sum(), "/", len(pa))
