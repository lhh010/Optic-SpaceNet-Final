"""Unified MNIST data loaders.

References:
    - src_raw/train_with_local_data.py
    - src_raw/train_and_quantize.py
    - src_dsqlsq/train_lsq_plus.py
    - src_dsqlsq/train_dsq.py
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms


class MNISTDataset(torch.utils.data.Dataset):
    """Custom MNIST dataset from numpy arrays."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_mnist_loaders_local(batch_size=256, data_dir="./data/processed"):
    """Load MNIST from local .npy files."""
    train_images = np.load(os.path.join(data_dir, "train_images.npy")).astype(np.float32) / 255.0
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    test_images = np.load(os.path.join(data_dir, "test_images.npy")).astype(np.float32) / 255.0
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def get_mnist_loaders_torchvision(batch_size=256, root="./data"):
    """Load MNIST via torchvision.datasets."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader
