# -*- coding: utf-8 -*-
"""EuroSAT test-set loader replicating torchvision ImageFolder ordering +
the eurosat_split test indices (used for the 90.43% baseline).

ImageFolder ordering (torchvision, sort=True): for each class dir in sorted
order, for each filename in sorted order -> global index 0..n-1.  Then the
baseline test split is:  np.random.RandomState(42).shuffle(list(range(n)))
[5400:10800] (eurosat_split.split_indices with val=test=0.2).

Only PIL + numpy + torch are needed (no torchvision).
"""
import os

import numpy as np
import torch
from PIL import Image

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_sample_list(data_dir):
    """Return list of (abs_path, class_idx) in torchvision ImageFolder order."""
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


class EuroSATTestSet(torch.utils.data.Dataset):
    """The exact 5400-image test set used for the 90.43% baseline."""

    def __init__(self, data_dir, indices=None, limit=None, offset=0):
        self.samples = build_sample_list(data_dir)
        self.n = len(self.samples)
        if indices is None:
            indices = test_indices(self.n)
        end = limit + offset if limit else None
        self.test_idx = indices[offset:end]
        print("EuroSAT total=%d test window=[%d:%s] (limit=%s)"
              % (self.n, offset, end if end else "", limit))

    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, i):
        path, cls = self.samples[self.test_idx[i]]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,64,64) in [0,1]
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, cls


if __name__ == "__main__":
    import sys
    ds = EuroSATTestSet(sys.argv[1] if len(sys.argv) > 1
                        else r"E:\LT-Simulator\train-test\data\EuroSAT_RGB")
    print("samples=%d test=%d" % (ds.n, len(ds)))
    # quick distribution check
    from collections import Counter
    c = Counter(cls for _, cls in ds.samples)
    print("class counts:", dict(sorted(c.items())))
    x, y = ds[0]
    print("sample0 shape", tuple(x.shape), "label", y, "min", float(x.min()),
          "max", float(x.max()))
