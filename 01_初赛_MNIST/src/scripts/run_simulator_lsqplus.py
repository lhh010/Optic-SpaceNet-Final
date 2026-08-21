"""LSQ+ photonic simulator runner."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.utils.io import load_weights_and_params
from src.inference.simulator import optical_mac_tiling, compute_optical_ratio


def main():
    model = None
    weights_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/lsqplus")
    w1, w2, quant_params = load_weights_and_params("lsqplus", data_dir=weights_dir)
    if "w3" not in quant_params:
        raise FileNotFoundError("Missing `w3_int4_lsq_plus.npy` for 3-layer LSQ+ simulator run.")
    w3 = quant_params["w3"]

    mnist_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    images = np.load(os.path.join(mnist_dir, "test_images.npy"))
    labels = np.load(os.path.join(mnist_dir, "test_labels.npy"))
    images_flat = images.reshape(-1, 784) / 255.0

    s_in, zp_in = quant_params["input_scale"], quant_params["input_zp"]
    s_w1, zp_w1 = quant_params["w1_scale"], quant_params["w1_zp"]
    s_h1, zp_h1 = quant_params["h1_scale"], quant_params["h1_zp"]
    s_w2, zp_w2 = quant_params["w2_scale"], quant_params["w2_zp"]
    s_h2, zp_h2 = quant_params["h2_scale"], quant_params["h2_zp"]
    s_w3, zp_w3 = quant_params["w3_scale"], quant_params["w3_zp"]

    x_uint4 = np.clip(np.round(images_flat / s_in) + zp_in, 0, 15).astype(np.int32)

    macs_layer1 = w1.shape[0] * w1.shape[1]
    macs_layer2 = w2.shape[0] * w2.shape[1]
    macs_layer3 = w3.shape[0] * w3.shape[1]
    total_macs = macs_layer1 + macs_layer2 + macs_layer3
    optical_ratio = compute_optical_ratio(total_macs, total_macs)

    num_samples = len(x_uint4)
    batch_size = 500
    correct = 0
    start = time.time()

    for i in range(0, num_samples, batch_size):
        x_chunk = x_uint4[i : i + batch_size]
        labels_chunk = labels[i : i + batch_size]

        optical_out1 = optical_mac_tiling(model, x_chunk, w1)
        zp_comp1 = (
            -np.sum(x_chunk, axis=1, keepdims=True) * zp_w1
            - np.sum(w1, axis=0, keepdims=True) * zp_in
            + x_chunk.shape[1] * zp_in * zp_w1
        )
        y1_int_mac = optical_out1 + zp_comp1
        y1_real = y1_int_mac * (s_in * s_w1)
        h1_real = np.maximum(0, y1_real)
        h1_uint4 = np.clip(np.round(h1_real / s_h1) + zp_h1, 0, 15).astype(np.int32)

        optical_out2 = optical_mac_tiling(model, h1_uint4, w2)
        zp_comp2 = (
            -np.sum(h1_uint4, axis=1, keepdims=True) * zp_w2
            - np.sum(w2, axis=0, keepdims=True) * zp_h1
            + h1_uint4.shape[1] * zp_h1 * zp_w2
        )
        y2_int_mac = optical_out2 + zp_comp2
        y2_real = y2_int_mac * (s_h1 * s_w2)
        h2_real = np.maximum(0, y2_real)
        h2_uint4 = np.clip(np.round(h2_real / s_h2) + zp_h2, 0, 15).astype(np.int32)

        optical_out3 = optical_mac_tiling(model, h2_uint4, w3)
        zp_comp3 = (
            -np.sum(h2_uint4, axis=1, keepdims=True) * zp_w3
            - np.sum(w3, axis=0, keepdims=True) * zp_h2
            + h2_uint4.shape[1] * zp_h2 * zp_w3
        )
        y3_int_mac = optical_out3 + zp_comp3

        predictions = np.argmax(y3_int_mac, axis=1)
        correct += np.sum(predictions == labels_chunk)

        processed = min(i + batch_size, num_samples)
        print(
            f"Progress: [{processed:5d} / {num_samples}] | "
            f"Time: {time.time()-start:.1f}s | Acc: {correct/processed*100:.2f}%"
        )

    final_acc = correct / num_samples * 100.0
    print(f"\nFinal accuracy: {final_acc:.2f}% | Optical ratio: {optical_ratio:.2f}%")


if __name__ == "__main__":
    main()
