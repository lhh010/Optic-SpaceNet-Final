#!/usr/bin/env python3
"""M9/M10 200 张抽样验证 — 本地容器内运行。

用法:
  python3 run_sample_verify.py --model model10 \
      [--backend http|numpy] [--limit 200] [--offset 0] \
      [--calib-col <calib_col_*.json>] [--images <npy> --labels <npy>]

输出 (终端 + /workspace/out):
  logits_<model>_<backend>_<offset>.npy   (N,10)
  errors_<model>_<backend>_<offset>.csv   (index,true,pred,top5 概率)
  汇总: acc / 错误数 / 每张耗时 / 引擎
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "demo", "server")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo.server import ds3net  # noqa: E402
from demo.server.gazelle_engine import HttpBackend, NumpyBackend  # noqa: E402

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

MODEL_WEIGHTS = {
    "model9": "/workspace/weights/m9_j1w075ds3_v8probe15.pth",
    "model10": "/workspace/weights/m10_ds3pool3_v8probe15.pth",
}
STEM_POOL = {"model9": "max", "model10": "max3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model10", choices=["model9", "model10"])
    ap.add_argument("--backend", default="http", choices=["http", "numpy"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--calib-col", default="")
    ap.add_argument("--images", default="/workspace/out/test200_images.npy")
    ap.add_argument("--labels", default="/workspace/out/test200_labels.npy")
    ap.add_argument("--host", default=os.environ.get("GAZELLE_HOST", "192.168.31.158"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GAZELLE_PORT", "8000")))
    ap.add_argument("--head-elec", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(MODEL_WEIGHTS[a.model]):
        raise SystemExit("权重不存在: %s" % MODEL_WEIGHTS[a.model])
    imgs = np.load(a.images)
    labels = np.load(a.labels)
    n = min(a.limit, len(labels) - a.offset)
    if n <= 0:
        raise SystemExit("offset/limit 超界 (len=%d)" % len(labels))

    ws, meta = ds3net.load_ds3(MODEL_WEIGHTS[a.model], STEM_POOL[a.model])
    cc = ds3net.load_calib_col(a.calib_col)

    if a.backend == "numpy":
        backend = NumpyBackend()
    else:
        backend = HttpBackend(host=a.host, port=a.port, timeout=300)

    t0 = time.perf_counter()
    correct, outs = 0, []
    B = 8
    for s in range(0, n, B):
        x = imgs[a.offset + s:a.offset + s + B]
        if a.backend == "http":
            # 逐张走 HTTP 更稳 (板上单线程服务 + 权重缓存); 也便于逐张计时
            batch = []
            for i in range(x.shape[0]):
                lg, _ = ds3net.forward_traced(x[i:i + 1], ws, meta, backend,
                                              calib_col=cc,
                                              head_elec=a.head_elec)
                batch.append(lg[0])
            logits = np.stack(batch)
        else:
            logits, _ = ds3net.forward_traced(x, ws, meta, backend,
                                              calib_col=cc,
                                              head_elec=a.head_elec)
        preds = np.argmax(logits, 1)
        seg = labels[a.offset + s:a.offset + s + B]
        correct += int(np.sum(preds == seg))
        outs.append(logits)
        done = s + len(seg)
        print("[%4d/%d] acc=%.2f%% elapsed=%.0fs"
              % (done, n, correct * 100.0 / done, time.perf_counter() - t0),
              flush=True)
    logits = np.vstack(outs)
    acc = correct * 100.0 / n
    elapsed = time.perf_counter() - t0

    out_dir = "/workspace/out"
    os.makedirs(out_dir, exist_ok=True)
    tag = "%s_%s_%d" % (a.model, a.backend, a.offset)
    np.save(os.path.join(out_dir, "logits_%s.npy" % tag), logits)
    preds = np.argmax(logits, 1)
    true = labels[a.offset:a.offset + n]
    err = [i for i in range(n) if preds[i] != true[i]]
    with open(os.path.join(out_dir, "errors_%s.csv" % tag), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true", "pred"] + CLASSES)
        for i in err:
            w.writerow([a.offset + i, CLASSES[true[i]], CLASSES[preds[i]]]
                       + ["%.4f" % v for v in logits[i]])
    print("\n========== %s 抽样验证 (n=%d, backend=%s) =========="
          % (a.model, n, a.backend))
    print("acc = %.2f%%  (%d/%d)  错误 %d"
          % (acc, correct, n, len(err)))
    print("耗时 %.0fs (%.2fs/张)" % (elapsed, elapsed / n))
    if err:
        print("错误样本: " + ", ".join("%s→%s" % (CLASSES[true[i]], CLASSES[preds[i]])
                                       for i in err[:20]))
    print("产物: %s/logits_%s.npy, %s/errors_%s.csv"
          % (out_dir, tag, out_dir, tag))


if __name__ == "__main__":
    main()
