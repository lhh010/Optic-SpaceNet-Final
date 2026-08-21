"""NumPy quantized inference engine for STE/FP baseline.

References:
    - src_raw/train_and_quantize.py (numpy validation block)
    - src_raw/run_simulator.py
"""

import numpy as np


def quantize_input_uint4(input_numpy):
    """Map [0.0, 1.0] inputs to unsigned 4-bit integers [0, 15]."""
    return np.clip(np.round(input_numpy * 15.0), 0, 15).astype(np.int32)


def run_inference(images, labels, w1_int4, w2_int4, w3_int4, scale_h1=None, scale_h2=None):
    """Run numpy inference for STE/FP quantized model.

    Args:
        images: numpy array of shape (N, 28, 28) in range [0, 255] or [0, 1].
        labels: numpy array of shape (N,).
        w1_int4: (784, hidden_dim) int32 weights.
        w2_int4: (hidden_dim, hidden_dim) int32 weights.
        w3_int4: (hidden_dim, 10) int32 weights.
        scale_h1: Optional float scaling factor from h1 to uint4.
        scale_h2: Optional float scaling factor from h2 to uint4.

    Returns:
        accuracy: float percentage.
        predictions: numpy array of shape (N,).
        scale_h1: float scaling factor used for first hidden layer.
        scale_h2: float scaling factor used for second hidden layer.
    """
    if images.max() > 1.0:
        images = images.astype(np.float32) / 255.0

    images_flat = images.reshape(-1, 784)
    x_uint4 = quantize_input_uint4(images_flat)

    y1_sim = np.matmul(x_uint4, w1_int4)
    h1_sim = np.maximum(0, y1_sim)

    if scale_h1 is None:
        max_h1 = np.max(h1_sim)
        scale_h1 = 15.0 / max(max_h1, 1e-8)

    h1_uint4 = np.clip(np.round(h1_sim * scale_h1), 0, 15).astype(np.int32)
    y2_sim = np.matmul(h1_uint4, w2_int4)
    h2_sim = np.maximum(0, y2_sim)

    if scale_h2 is None:
        max_h2 = np.max(h2_sim)
        scale_h2 = 15.0 / max(max_h2, 1e-8)

    h2_uint4 = np.clip(np.round(h2_sim * scale_h2), 0, 15).astype(np.int32)
    y3_sim = np.matmul(h2_uint4, w3_int4)

    predictions = np.argmax(y3_sim, axis=1)
    accuracy = np.sum(predictions == labels) / len(labels) * 100.0
    return accuracy, predictions, scale_h1, scale_h2
