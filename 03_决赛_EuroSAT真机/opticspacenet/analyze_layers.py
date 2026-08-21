# -*- coding: utf-8 -*-
"""Per-layer hardware calibration for Optic-SpaceNet on the real Gazelle board.

Runs N images through the optical model on the REAL board, records for each
optical layer the (x_int, w_int, y_exact, y_hw) pairs, fits per-output-channel
affine corrections  y_corrected[:,j] = (y_hw[:,j] - b_j)/a_j, and saves them
to a .npz calibration file keyed by md5 of the int8 weight matrix.

Usage (Windows/WSL client):
  CALIB_OUT=calib.npz LIMIT=100 python analyze_layers.py

Env: LIMIT, BATCH, WEIGHT, DATA, OPTC_HOST, OPTC_PORT, CALIB_OUT.
"""
import hashlib
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TS = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "train-test"))
for _p in (os.path.join(_TS, "src", "core"), os.path.join(_TS, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eurosat_loader import EuroSATTestSet  # noqa: E402
import gazelle_engine as GE  # noqa: E402


class RecorderBackend(GE.HttpBackend):
    """Records (x_int, w_int, y_exact, y_hw) per call."""

    def __init__(self, *args, **kwargs):
        super(RecorderBackend, self).__init__(*args, **kwargs)
        self.records = []

    def matmul_2d(self, x_int, w_int):
        y_exact = np.matmul(x_int.astype(np.float64), w_int.astype(np.float64))
        y_hw = super(RecorderBackend, self).matmul_2d(x_int, w_int)
        self.records.append((x_int.copy(), w_int.copy(), y_exact, y_hw))
        return y_hw


def main():
    limit = int(os.environ.get("LIMIT", "100"))
    batch = int(os.environ.get("BATCH", "1"))
    reps = int(os.environ.get("REP", "1"))
    model_name = os.environ.get("MODEL", "model2")
    weight = os.environ.get("WEIGHT", None)  # None -> per-model default
    data_dir = os.environ.get("DATA", os.path.join(_TS, "data", "EuroSAT_RGB"))
    host = os.environ.get("OPTC_HOST", "127.0.0.1")
    port = int(os.environ.get("OPTC_PORT", "8000"))
    out_path = os.environ.get("CALIB_OUT", os.path.join(_HERE, "calib.npz"))

    loader = torch.utils.data.DataLoader(
        EuroSATTestSet(data_dir, limit=limit), batch_size=batch, shuffle=False)

    backend = RecorderBackend(host=host, port=port, reps=reps)
    model, engine = GE.build_model(weight, backend, model_name=model_name)

    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            correct += int((out.argmax(1) == y).sum())
            total += x.size(0)
    print("HW raw accuracy on %d calib images: %.2f%% (%d/%d)  [%.0fs]"
          % (total, 100.0 * correct / total, correct, total,
             time.time() - t0), flush=True)

    # layer count = records per forward pass (5 for model2/3, 7 for model1)
    n_batches = max(1, len(loader))
    n_layers = len(backend.records) // n_batches
    if n_layers <= 0 or len(backend.records) % n_batches != 0:
        raise RuntimeError("unexpected record count %d for %d batches"
                           % (len(backend.records), n_batches))
    if model_name.startswith("model1"):
        names = ["conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
                 "fc1", "fc2"][:n_layers]
    elif model_name == "model4":
        names = ["stage1.0", "stage1.3", "stage2.0", "stage2.3",
                 "stage3.0", "stage3.3", "head"][:n_layers]
    elif model_name.startswith(("model5", "model6", "model7", "model8")):
        # J1 家族: head 电算 -> 5 光计算层 (与 v8 head_fp32 训练语义一致)
        names = ["stage1.0", "stage2.0", "stage2.3", "stage3.0",
                 "stage3.3"][:n_layers]
    else:
        names = ["stage1", "stage2", "stage3", "fc1", "fc2"][:n_layers]
    calib = {}
    print("", flush=True)
    for li in range(n_layers):
        recs = backend.records[li::n_layers]
        x = np.concatenate([r[0] for r in recs], axis=0)
        w = recs[0][1]
        ye = np.concatenate([r[2] for r in recs], axis=0)
        yh = np.concatenate([r[3] for r in recs], axis=0)
        n, k = x.shape
        ncol = w.shape[1]
        a_j = np.ones(ncol)
        b_j = np.zeros(ncol)
        for j in range(ncol):
            xe, xh = ye[:, j], yh[:, j]
            a = np.dot(xe, xh) / np.dot(xe, xe)
            b = np.mean(xh) - a * np.mean(xe)
            a_j[j], b_j[j] = a, b
        key = hashlib.md5(w.astype(np.int8).tobytes()).hexdigest()
        calib[key] = {"layer": names[li], "k": int(k), "n": int(ncol),
                      "a": a_j.tolist(), "b": b_j.tolist()}
        resid_raw = (yh - ye)
        resid_corr = (yh - (a_j.reshape(1, -1) * ye + b_j.reshape(1, -1)))
        print("layer %-7s (%4d,%3d)@(%3d,%3d)  exact|.|mean=%8.0f  "
              "raw_err std=%8.0f  a_mean=%6.4f b_mean=%8.0f  "
              "corr_err std=%8.0f (%.2fx reduction)"
              % (names[li], n, k, k, ncol, np.abs(ye).mean(),
                 resid_raw.std(), a_j.mean(), b_j.mean(), resid_corr.std(),
                 resid_raw.std() / max(resid_corr.std(), 1e-9)), flush=True)

    np.savez(out_path, **{k: np.array([calib[k]["a"], calib[k]["b"]],
                                      dtype=np.float64)
                          for k in calib})
    # also save weight hashes + layer names
    np.savez(out_path + ".meta",
             keys=list(calib.keys()),
             layers=[calib[k]["layer"] for k in calib])
    print("calibration saved -> %s (%d layers)" % (out_path, len(calib)), flush=True)


if __name__ == "__main__":
    main()
