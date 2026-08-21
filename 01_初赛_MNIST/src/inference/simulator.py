"""Shared photonic simulator tiling logic.

References:
    - src_raw/run_simulator.py
    - src_dsqlsq/run_simulator_lsq_plus.py
    - src_dsqlsq/run_simulator_dsq.py
"""

import numpy as np


def optical_mac_tiling(simulator_model, x_int4, w_int4, use_simulator=False):
    """Simulate full x * w using 8x2 photonic array tiles.

    Args:
        simulator_model: Placeholder for actual simulator (usually None).
        x_int4: (Batch, in_dim) int32 activations.
        w_int4: (in_dim, out_dim) int32 weights.
        use_simulator: If True, call simulator_model; else use np.matmul.

    Returns:
        final_result: (Batch, out_dim) int32 accumulated result.
    """
    B = x_int4.shape[0]
    in_dim = x_int4.shape[1]
    out_dim = w_int4.shape[1]

    final_result = np.zeros((B, out_dim), dtype=np.int32)
    H_in = 8
    W_out = 2

    for j in range(0, out_dim, W_out):
        for i in range(0, in_dim, H_in):
            x_tile = x_int4[:, i : i + H_in]
            w_tile = w_int4[i : i + H_in, j : j + W_out]

            if use_simulator and simulator_model is not None:
                x_tile_sim = np.expand_dims(x_tile, axis=0)
                tile_out = simulator_model(x_tile_sim, w_tile, inputType="uint4").numpy()
            else:
                tile_out = np.matmul(x_tile, w_tile)

            final_result[:, j : j + W_out] += tile_out

    return final_result


def compute_optical_ratio(total_macs, optical_macs):
    """Compute optical MAC percentage."""
    return (optical_macs / total_macs) * 100.0 if total_macs > 0 else 0.0
