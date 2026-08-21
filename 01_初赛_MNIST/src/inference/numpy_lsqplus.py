"""NumPy quantized inference engine for LSQ+ with zero-point compensation.

References:
    - src_dsqlsq/train_lsq_plus.py (numpy validation block)
    - src_dsqlsq/run_simulator_lsq_plus.py
"""

import numpy as np


def run_inference(images, labels, w1_int4, w2_int4, w3_int4, quant_params):
    """Run numpy inference for LSQ+ quantized model with ZP compensation.

    Args:
        images: numpy array of shape (N, 28, 28) in range [0, 255] or [0, 1].
        labels: numpy array of shape (N,).
        w1_int4: (784, hidden_dim) int32 weights.
        w2_int4: (hidden_dim, hidden_dim) int32 weights.
        w3_int4: (hidden_dim, 10) int32 weights.
        quant_params: dict with scales and zero_points.

    Returns:
        accuracy: float percentage.
        predictions: numpy array of shape (N,).
    """
    if images.max() > 1.0:
        images = images.astype(np.float32) / 255.0

    s_in, zp_in = quant_params["input_scale"], quant_params["input_zp"]
    s_w1, zp_w1 = quant_params["w1_scale"], quant_params["w1_zp"]
    s_h1, zp_h1 = quant_params["h1_scale"], quant_params["h1_zp"]
    s_w2, zp_w2 = quant_params["w2_scale"], quant_params["w2_zp"]
    s_h2, zp_h2 = quant_params["h2_scale"], quant_params["h2_zp"]
    s_w3, zp_w3 = quant_params["w3_scale"], quant_params["w3_zp"]

    images_flat = images.reshape(-1, 784)
    x_int = np.clip(np.round(images_flat / s_in) + zp_in, 0, 15).astype(np.int32)

    x_shifted = x_int - zp_in
    w1_shifted = w1_int4 - zp_w1
    y1_int_mac = np.matmul(x_shifted, w1_shifted)

    y1_real = y1_int_mac * (s_in * s_w1)
    h1_real = np.maximum(0, y1_real)
    h1_int = np.clip(np.round(h1_real / s_h1) + zp_h1, 0, 15).astype(np.int32)

    h1_shifted = h1_int - zp_h1
    w2_shifted = w2_int4 - zp_w2
    y2_int_mac = np.matmul(h1_shifted, w2_shifted)
    y2_real = y2_int_mac * (s_h1 * s_w2)
    h2_real = np.maximum(0, y2_real)
    h2_int = np.clip(np.round(h2_real / s_h2) + zp_h2, 0, 15).astype(np.int32)

    h2_shifted = h2_int - zp_h2
    w3_shifted = w3_int4 - zp_w3
    y3_int_mac = np.matmul(h2_shifted, w3_shifted)

    predictions = np.argmax(y3_int_mac, axis=1)
    accuracy = np.sum(predictions == labels) / len(labels) * 100.0
    return accuracy, predictions
