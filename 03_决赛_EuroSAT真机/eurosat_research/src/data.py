"""
===============================================================================
 data.py — EuroSAT 数据加载 (复用 eurosat_split 单一数据源)
===============================================================================
 增强策略:
   standard: HFlip + Rot10 (现状)
   strong:   HFlip + Rot90 (遥感无方向性) + RandomResizedCrop + ColorJitter
   none:     无增强
===============================================================================
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
# eurosat_split.py 归档于本仓库 src/data/ (原布局 Ltsimulator-test/src/data 已失效)
sys.path.insert(0, os.path.join(_HERE, "data"))


def _make_transforms(aug):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if aug == "none":
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])
    elif aug == "strong":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(), normalize,
        ])
    else:  # standard
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(), normalize,
        ])
    val_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, val_tf


def load_eurosat(data_dir, batch_size=64, aug="standard", val_split=0.2,
                 seed=42, num_workers=4):
    from eurosat_split import split_indices

    train_tf, val_tf = _make_transforms(aug)
    train_full = datasets.ImageFolder(data_dir, transform=train_tf)
    val_full = datasets.ImageFolder(data_dir, transform=val_tf)

    n = len(train_full)
    train_idx, val_idx, test_idx = split_indices(n, seed=seed,
                                                  val_ratio=val_split,
                                                  test_ratio=val_split)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    test_ds = Subset(val_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
