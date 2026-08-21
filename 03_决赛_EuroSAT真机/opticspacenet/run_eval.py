# -*- coding: utf-8 -*-
"""Optic-SpaceNet (Model 2) real-Gazelle inference client.

Runs the FULL baseline pipeline (stem electronic + 5 optical layers) with the
optical matmuls executed either on the real Gazelle board (BACKEND=http, via
SSH tunnel to the board's server_gazelle.py) or exactly in numpy
(BACKEND=numpy, clean reference).

Env control:
  BACKEND   numpy|http         (default http)
  LIMIT     test images        (default 200)
  BATCH     images per batch   (default 1 — matches baseline per-image quant)
  WEIGHT    .pth path          (default train-test/weights/spacenet_v1_phase4_v3_int8.pth)
  DATA      EuroSAT_RGB dir    (default train-test/data/EuroSAT_RGB)
  OPTC_HOST tunnel host        (default 127.0.0.1)
  OPTC_PORT tunnel port        (default 8000)
  REF       also run numpy reference for the same images (default 1)
  CORRECTION .npz calibration file (per-layer affine correction, optional)
  REP       repeat each optical matmul N times and average (noise reduction)
  MODEL     model2|model3|model1a|model1b (default model2)
"""
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
from gazelle_engine import (  # noqa: E402
    NumpyBackend, HttpBackend, build_model, OpticSpaceNetV1_INT8, load_state)


def evaluate(model, loader, device, desc, print_every=50, err_out=None,
             offset=0):
    """Evaluate; if err_out is a dict, fill per-sample error records:
    err_out["idx"]/["true"]/["pred"] lists (global index = offset + position)."""
    model.eval()
    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.unsqueeze(0) if x.dim() == 3 else x
            out = model(x)
            pred = out.argmax(1)
            correct += int((pred == y).sum())
            total += x.size(0)
            if err_out is not None:
                y_np = y.numpy()
                p_np = pred.numpy()
                base = offset + i * x.size(0)
                for j in range(x.size(0)):
                    if p_np[j] != y_np[j]:
                        err_out["idx"].append(base + j)
                        err_out["true"].append(int(y_np[j]))
                        err_out["pred"].append(int(p_np[j]))
            if (i + 1) % print_every == 0 or (i + 1) == len(loader):
                print("[%s] %5d/%d  acc=%.2f%%  elapsed=%.0fs"
                      % (desc, i + 1, len(loader), correct * 100.0 / total,
                         time.time() - t0), flush=True)
    return correct * 100.0 / total, correct, total


def main():
    backend_name = os.environ.get("BACKEND", "http")
    limit = int(os.environ.get("LIMIT", "200"))
    batch = int(os.environ.get("BATCH", "1"))
    weight = os.environ.get("WEIGHT", None)  # None -> per-model default
    data_dir = os.environ.get("DATA", os.path.join(_TS, "data", "EuroSAT_RGB"))
    host = os.environ.get("OPTC_HOST", "127.0.0.1")
    port = int(os.environ.get("OPTC_PORT", "8000"))
    run_ref = os.environ.get("REF", "1") == "1"
    corr_path = os.environ.get("CORRECTION", "")
    model_name = os.environ.get("MODEL", "model2")

    print("=" * 66)
    print("  Optic-SpaceNet %s -> REAL Gazelle (compass_matmul)" % model_name)
    print("  backend=%s  limit=%d  batch=%d" % (backend_name, limit, batch))
    print("  weight=%s" % weight)
    if corr_path:
        print("  correction=%s" % corr_path)
    print("=" * 66)

    loader = torch.utils.data.DataLoader(
        EuroSATTestSet(data_dir, limit=limit,
                       offset=int(os.environ.get("OFFSET", "0"))),
        batch_size=batch, shuffle=False)

    correction = None
    if backend_name == "http":
        backend = HttpBackend(host=host, port=port,
                              reps=int(os.environ.get("REP", "1")))
    else:
        backend = NumpyBackend()

    if corr_path:
        z = np.load(corr_path)
        correction = {}
        meta = None
        meta_path = corr_path + ".meta"
        if os.path.isfile(meta_path):
            m = np.load(meta_path, allow_pickle=True)
            meta = dict(zip(m["keys"], m["layers"]))
        for k in z.files:
            arr = z[k]  # (2, n) -> a_j, b_j
            correction[k] = (arr[0], arr[1])
        print("correction loaded: %d weight keys" % len(correction), flush=True)

    model, engine = build_model(weight, backend, correction=correction,
                                model_name=model_name)
    offset = int(os.environ.get("OFFSET", "0"))
    err_out = None
    err_path = os.environ.get("ERR_OUT", "")
    if err_path:
        err_out = {"idx": [], "true": [], "pred": []}
        print("error records -> %s (offset=%d)" % (err_path, offset), flush=True)
    acc, correct, total = evaluate(model, loader, "cpu", "HW/%s" % backend.name,
                                   err_out=err_out, offset=offset)
    print("FINAL %s accuracy (%d images): %.2f%% (%d/%d)"
          % (backend.name, total, acc, correct, total), flush=True)
    print("engine stats: calls=%d total_time=%.1fs"
          % (engine.stats["calls"], engine.stats["total_time"]), flush=True)
    if err_out is not None:
        np.savez(err_path,
                 idx=np.asarray(err_out["idx"], dtype=np.int64),
                 true=np.asarray(err_out["true"], dtype=np.int64),
                 pred=np.asarray(err_out["pred"], dtype=np.int64),
                 offset=np.asarray(offset, dtype=np.int64),
                 limit=np.asarray(total, dtype=np.int64))
        print("saved %d error records -> %s" % (len(err_out["idx"]), err_path),
              flush=True)

    if run_ref and backend_name != "numpy":
        print("\n--- NumPy clean reference (same quantization path) ---")
        model2, engine2 = build_model(weight, NumpyBackend(),
                                      model_name=model_name)
        acc2, correct2, total2 = evaluate(model2, loader, "cpu", "REF/numpy")
        print("FINAL numpy reference accuracy (%d images): %.2f%% (%d/%d)"
              % (total2, acc2, correct2, total2), flush=True)
        print("HW vs clean-ref gap: %.2f points"
              % (acc - acc2), flush=True)


if __name__ == "__main__":
    main()
