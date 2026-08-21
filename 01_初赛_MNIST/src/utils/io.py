"""Weight export/import and quantization parameter I/O utilities.

Preserves existing .npy artifact paths for backward compatibility.
"""

import os
import numpy as np
import torch


def _to_numpy(weight):
    """Convert torch tensor or numpy array to numpy."""
    if hasattr(weight, "detach"):
        return weight.detach().cpu().numpy()
    return np.asarray(weight)


def quantize_weight_int4(weight_tensor):
    """Map float weights to signed 4-bit integers [-8, 7]."""
    w_numpy = _to_numpy(weight_tensor)
    max_val = np.max(np.abs(w_numpy))
    scale = 7.0 / max(max_val, 1e-8)
    w_q = np.clip(np.round(w_numpy * scale), -8, 7).astype(np.int32)
    return w_q


def quantize_weight_with_scale(weight_tensor, scale, zero_point, num_bits=4, is_signed=True):
    """Map float weights to integers using provided scale and zero_point."""
    w_numpy = _to_numpy(weight_tensor)
    qmin = -(2 ** (num_bits - 1)) if is_signed else 0
    qmax = (2 ** (num_bits - 1) - 1) if is_signed else 2**num_bits - 1
    w_q = np.round(w_numpy / abs(scale)) + zero_point
    w_q = np.clip(w_q, qmin, qmax)
    return w_q.astype(np.int32)


def export_weights_steq(model, save_dir="."):
    """Export STE QAT weights. Saves w1_int4.npy, w2_int4.npy, w3_int4.npy."""
    os.makedirs(save_dir, exist_ok=True)
    w1_float = model.fc1.weight.T.detach().cpu().numpy()
    w2_float = model.fc2.weight.T.detach().cpu().numpy()
    w3_float = model.fc3.weight.T.detach().cpu().numpy()

    w1_int4 = quantize_weight_int4(w1_float)
    w2_int4 = quantize_weight_int4(w2_float)
    w3_int4 = quantize_weight_int4(w3_float)

    w1_export = np.expand_dims(w1_int4, axis=0)
    w2_export = np.expand_dims(w2_int4, axis=0)
    w3_export = np.expand_dims(w3_int4, axis=0)

    np.save(os.path.join(save_dir, "w1_int4.npy"), w1_export)
    np.save(os.path.join(save_dir, "w2_int4.npy"), w2_export)
    np.save(os.path.join(save_dir, "w3_int4.npy"), w3_export)


def export_weights_lsqplus(model, save_dir="."):
    """Export LSQ+ weights and quant params."""
    os.makedirs(save_dir, exist_ok=True)
    q_info = model.get_quantization_info()
    s_w1, zp_w1 = q_info["w1_scale"], q_info["w1_zp"]
    s_w2, zp_w2 = q_info["w2_scale"], q_info["w2_zp"]
    s_w3, zp_w3 = q_info["w3_scale"], q_info["w3_zp"]

    w1_int4 = quantize_weight_with_scale(model.fc1.weight.T, s_w1, zp_w1, 4, True)
    w2_int4 = quantize_weight_with_scale(model.fc2.weight.T, s_w2, zp_w2, 4, True)
    w3_int4 = quantize_weight_with_scale(model.fc3.weight.T, s_w3, zp_w3, 4, True)

    w1_export = np.expand_dims(w1_int4, axis=0)
    w2_export = np.expand_dims(w2_int4, axis=0)
    w3_export = np.expand_dims(w3_int4, axis=0)

    np.save(os.path.join(save_dir, "w1_int4_lsq_plus.npy"), w1_export)
    np.save(os.path.join(save_dir, "w2_int4_lsq_plus.npy"), w2_export)
    np.save(os.path.join(save_dir, "w3_int4_lsq_plus.npy"), w3_export)
    np.save(os.path.join(save_dir, "lsq_plus_quant_params.npy"), q_info)


def export_weights_dsq(model, save_dir="."):
    """Export DSQ weights and quant params."""
    os.makedirs(save_dir, exist_ok=True)
    q_info = model.get_quantization_info()
    s_w1 = q_info["w1_scale"]
    s_w2 = q_info["w2_scale"]
    s_w3 = q_info["w3_scale"]

    w1_int4 = quantize_weight_with_scale(model.fc1.weight.T, s_w1, 0, 4, True)
    w2_int4 = quantize_weight_with_scale(model.fc2.weight.T, s_w2, 0, 4, True)
    w3_int4 = quantize_weight_with_scale(model.fc3.weight.T, s_w3, 0, 4, True)

    w1_export = np.expand_dims(w1_int4, axis=0)
    w2_export = np.expand_dims(w2_int4, axis=0)
    w3_export = np.expand_dims(w3_int4, axis=0)

    np.save(os.path.join(save_dir, "w1_int4_dsq.npy"), w1_export)
    np.save(os.path.join(save_dir, "w2_int4_dsq.npy"), w2_export)
    np.save(os.path.join(save_dir, "w3_int4_dsq.npy"), w3_export)
    np.save(os.path.join(save_dir, "dsq_quant_params.npy"), q_info)


def load_weights_and_params(method, data_dir="."):
    """Load existing weight artifacts for a given method.

    method: 'steq' | 'lsqplus' | 'dsq'
    Returns: (w1, w2, params_or_None)
    """
    if method == "steq":
        w1 = np.load(os.path.join(data_dir, "w1_int4.npy"))[0]
        w2 = np.load(os.path.join(data_dir, "w2_int4.npy"))[0]
        params = {}

        w3_path = os.path.join(data_dir, "w3_int4.npy")
        if os.path.exists(w3_path):
            params["w3"] = np.load(w3_path)[0]

        params_path = os.path.join(data_dir, "steq_quant_params.npy")
        if os.path.exists(params_path):
            params.update(np.load(params_path, allow_pickle=True).item())

        return w1, w2, params or None
    elif method == "lsqplus":
        w1 = np.load(os.path.join(data_dir, "w1_int4_lsq_plus.npy"))[0]
        w2 = np.load(os.path.join(data_dir, "w2_int4_lsq_plus.npy"))[0]
        params = np.load(os.path.join(data_dir, "lsq_plus_quant_params.npy"), allow_pickle=True).item()
        w3_path = os.path.join(data_dir, "w3_int4_lsq_plus.npy")
        if os.path.exists(w3_path):
            params["w3"] = np.load(w3_path)[0]
        return w1, w2, params
    elif method == "dsq":
        w1 = np.load(os.path.join(data_dir, "w1_int4_dsq.npy"))[0]
        w2 = np.load(os.path.join(data_dir, "w2_int4_dsq.npy"))[0]
        params = np.load(os.path.join(data_dir, "dsq_quant_params.npy"), allow_pickle=True).item()
        w3_path = os.path.join(data_dir, "w3_int4_dsq.npy")
        if os.path.exists(w3_path):
            params["w3"] = np.load(w3_path)[0]
        return w1, w2, params
    else:
        raise ValueError(f"Unknown method: {method}")
