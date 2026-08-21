#!/usr/bin/env python3
"""生成 EuroSAT test 5400 张 images/labels npy（PIL+numpy 复刻 eurosat_loader 管线）。
输出与 export_ds3.py (torchvision ImageFolder + Normalize) 逐位一致。
用法: python3 make_test5400.py <data_dir> <out_dir>
"""
import os
import sys
import numpy as np
from PIL import Image

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def build_sample_list(data_dir):
    samples = []
    for ci, cls in enumerate(sorted(CLASSES)):
        d = os.path.join(data_dir, cls)
        names = sorted(os.listdir(d))
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(d, name), ci))
    return samples


def test_indices(n, seed=42, val_ratio=0.2, test_ratio=0.2):
    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)
    idx = list(range(n))
    np.random.RandomState(seed).shuffle(idx)
    return idx[val_size:val_size + test_size]


def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    samples = build_sample_list(data_dir)
    n = len(samples)
    assert n == 27000, f"expect 27000 samples, got {n}"
    te = test_indices(n)
    assert len(te) == 5400
    images = np.empty((5400, 3, 64, 64), dtype=np.float32)
    labels = np.empty(5400, dtype=np.int64)
    for i, gi in enumerate(te):
        path, cls = samples[gi]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        t = arr.transpose(2, 0, 1)
        t = (t - MEAN) / STD
        images[i] = t
        labels[i] = cls
    np.save(os.path.join(out_dir, "test_images_5400.npy"), images)
    np.save(os.path.join(out_dir, "test_labels_5400.npy"), labels)
    print("saved", images.shape, labels.shape)
    from collections import Counter
    print("label counts:", dict(sorted(Counter(labels.tolist()).items())))


if __name__ == "__main__":
    main()
