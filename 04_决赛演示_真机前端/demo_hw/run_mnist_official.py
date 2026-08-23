# -*- coding: utf-8 -*-
"""板上跑 CICC 官方 200 张 MNIST (DSQ, 路径B)。用法:
  sudo env MNIST_METHOD=dsq python3 run_mnist_official.py [limit] [offset]
依赖: 同目录 test_images_official200.npy (N,784 uint8) + test_labels_official200.npy
复用 run_mnist_gazelle.py 的 load_method/forward/forward_np/quantize_input。"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ['MNIST_METHOD'] = os.environ.get('MNIST_METHOD', 'dsq')
os.environ['MNIST_MODE'] = os.environ.get('MNIST_MODE', 'scale')

import run_mnist_gazelle as R

from compass_sdk.fast_calibration.compass_lib import compass_init
compass_init(150)

limit = int(os.environ.get("MNIST_LIMIT", "200"))
offset = int(os.environ.get("MNIST_OFFSET", "0"))

w1, w2, w3, q = R.load_method(HERE)
images = np.load(os.path.join(HERE, 'test_images_official200.npy'))
labels = np.load(os.path.join(HERE, 'test_labels_official200.npy'))
if images.ndim == 3:
    images = images.reshape(len(images), -1)
if offset + limit > len(images):
    limit = len(images) - offset
x = (images[offset:offset + limit].astype(np.float32) / 255.0)
y = labels[offset:offset + limit]
x_int = R.quantize_input(x, q)
print('MNIST official: n=%d offset=%d method=%s' % (limit, offset, R.METHOD), flush=True)

t0 = time.time()
y_hw = R.forward(x_int, w1, w2, w3, q)
acc_hw = float(np.mean(np.argmax(y_hw, 1) == y)) * 100
y_np = R.forward_np(x_int, w1, w2, w3, q)
acc_np = float(np.mean(np.argmax(y_np, 1) == y)) * 100
print('FINAL official accuracy (HW, n=%d): %.2f%%' % (limit, acc_hw), flush=True)
print('NumPy reference accuracy (same quant path): %.2f%%' % acc_np, flush=True)
print('gap: %.2f points, elapsed=%.1fs' % (abs(acc_hw - acc_np), time.time() - t0), flush=True)