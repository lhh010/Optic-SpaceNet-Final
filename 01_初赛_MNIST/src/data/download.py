"""Hugging Face API MNIST download helpers.

References:
    - src_raw/download_mnist_api.py
    - src_raw/load_dataset.py
"""

import os
import numpy as np
import requests
from datasets import load_dataset


def download_mnist_split(split: str, total_length: int, save_dir: str = "./data/processed"):
    """Download MNIST via HF datasets-server API in batches and save as .npy files."""
    os.makedirs(save_dir, exist_ok=True)
    base_url = "https://datasets-server.huggingface.co/rows"
    dataset = "ylecun/mnist"
    config = "mnist"

    images = []
    labels = []
    batch_size = 100

    for offset in range(0, total_length, batch_size):
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_length - offset),
        }
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for row in data.get("rows", []):
            img_dict = row["row"]["image"]
            if "bytes" in img_dict:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(img_dict["bytes"])).convert("L")
                img_array = np.array(img, dtype=np.uint8)
            else:
                raise ValueError("Unsupported image format in API response")
            images.append(img_array)
            labels.append(row["row"]["label"])

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)

    prefix = "train" if split == "train" else "test"
    np.save(os.path.join(save_dir, f"{prefix}_images.npy"), images)
    np.save(os.path.join(save_dir, f"{prefix}_labels.npy"), labels)
    print(f"Saved {prefix} set: {images.shape}, {labels.shape}")
    return images, labels


def load_mnist_to_numpy(save_dir: str = "./data/processed"):
    """Load MNIST via datasets library and save as .npy files."""
    os.makedirs(save_dir, exist_ok=True)
    dataset = load_dataset("ylecun/mnist")

    train_images = np.array([np.array(img) for img in dataset["train"]["image"]], dtype=np.uint8)
    train_labels = np.array(dataset["train"]["label"], dtype=np.int64)
    test_images = np.array([np.array(img) for img in dataset["test"]["image"]], dtype=np.uint8)
    test_labels = np.array(dataset["test"]["label"], dtype=np.int64)

    np.save(os.path.join(save_dir, "train_images.npy"), train_images)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(save_dir, "test_images.npy"), test_images)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)
    print("MNIST saved to .npy files successfully.")
