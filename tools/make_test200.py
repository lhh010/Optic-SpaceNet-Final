#!/usr/bin/env python3
"""生成 EuroSAT test 抽样 npy (与 canonical 5400 全量同源, 前 N 张 test)。
用法: python3 make_test200.py --data-dir <EuroSAT_RGB> --out <out.npy 前缀> --limit 200
输出: <out>_images.npy (N,3,64,64) float32 + <out>_labels.npy (N,) int64
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "src", "data"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from eurosat_split import split_indices  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="/workspace/out/test200")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()

    samples = []
    for ci, cls in enumerate(sorted(CLASSES)):
        d = os.path.join(a.data_dir, cls)
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(d, name), ci))
    n = len(samples)
    assert n == 27000, "expect 27000 samples, got %d" % n
    _, _, test_idx = split_indices(n, seed=42, val_ratio=0.2, test_ratio=0.2)
    sel = test_idx[a.offset:a.offset + a.limit]
    if not sel:
        raise SystemExit("offset/limit 超出 test 集 (5400)")

    imgs = np.empty((len(sel), 3, 64, 64), dtype=np.float32)
    labels = np.empty(len(sel), dtype=np.int64)
    for i, gi in enumerate(sel):
        path, cls = samples[gi]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        imgs[i] = (arr.transpose(2, 0, 1) - MEAN) / STD
        labels[i] = cls

    np.save(a.out + "_images.npy", imgs)
    np.save(a.out + "_labels.npy", labels)
    print("saved %s_images.npy (%s) + labels, n=%d [%d:%d]"
          % (a.out, imgs.shape, len(sel), a.offset, a.offset + len(sel)))


if __name__ == "__main__":
    main()
