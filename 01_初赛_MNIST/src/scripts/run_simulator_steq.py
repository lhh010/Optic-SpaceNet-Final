"""STE baseline photonic simulator runner."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.utils.io import load_weights_and_params
from src.inference.simulator import optical_mac_tiling


def main():
    model = None
    weights_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/ste")
    w1, w2, steq_params = load_weights_and_params("steq", data_dir=weights_dir)
    if not steq_params or "w3" not in steq_params:
        raise FileNotFoundError("Missing `w3_int4.npy` for 3-layer STE simulator run.")
    w3 = steq_params["w3"]

    mnist_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    images = np.load(os.path.join(mnist_dir, "test_images.npy"))
    labels = np.load(os.path.join(mnist_dir, "test_labels.npy"))
    images_flat = images.reshape(-1, 784) / 255.0
    x_uint4 = np.clip(np.round(images_flat * 15.0), 0, 15).astype(np.int32)

    scale_h1 = steq_params.get("scale_h1") if steq_params else None
    scale_h2 = steq_params.get("scale_h2") if steq_params else None
    if scale_h1 is None or scale_h2 is None:
        y1_calib = np.matmul(x_uint4, w1)
        h1_calib = np.maximum(0, y1_calib)
        scale_h1 = 15.0 / max(np.max(h1_calib), 1e-8)
        h1_uint4_calib = np.clip(np.round(h1_calib * scale_h1), 0, 15).astype(np.int32)
        y2_calib = np.matmul(h1_uint4_calib, w2)
        h2_calib = np.maximum(0, y2_calib)
        scale_h2 = 15.0 / max(np.max(h2_calib), 1e-8)

    num_samples = len(x_uint4)
    batch_size = 500
    correct = 0

    start = time.time()
    for i in range(0, num_samples, batch_size):
        x_chunk = x_uint4[i : i + batch_size]
        labels_chunk = labels[i : i + batch_size]

        y1_out = optical_mac_tiling(model, x_chunk, w1)
        h1_act = np.maximum(0, y1_out)
        h1_uint4 = np.clip(np.round(h1_act * scale_h1), 0, 15).astype(np.int32)

        y2_out = optical_mac_tiling(model, h1_uint4, w2)
        h2_act = np.maximum(0, y2_out)
        h2_uint4 = np.clip(np.round(h2_act * scale_h2), 0, 15).astype(np.int32)

        logits = optical_mac_tiling(model, h2_uint4, w3)
        predictions = np.argmax(logits, axis=1)
        correct += np.sum(predictions == labels_chunk)

        processed = min(i + batch_size, num_samples)
        print(f"Progress: [{processed:5d} / {num_samples}] | Acc: {correct/processed*100:.2f}%")

    final_acc = correct / num_samples * 100.0
    print(f"\nFinal accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()
