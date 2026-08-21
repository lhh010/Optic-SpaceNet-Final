"""STE QAT training script."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.qat_steq import PhotonicMLP_STEQ
from src.data.loaders import get_mnist_loaders_torchvision
from src.training.common import train_epoch, evaluate
from src.utils.io import export_weights_steq
from src.inference.numpy_steq import run_inference


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.join(os.path.dirname(__file__), "../../data/raw")
    train_loader, test_loader = get_mnist_loaders_torchvision(batch_size=256, root=data_root)

    model = PhotonicMLP_STEQ(hidden_dim1=128, hidden_dim2=64, noise_std=0.05).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 10
    print(f"Starting STE QAT training ({epochs} epochs) on {device}...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}%")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Acc (PyTorch): {test_acc:.2f}%")

    print("\nValidating with NumPy inference...")
    all_images = test_loader.dataset.data.numpy() / 255.0
    all_labels = test_loader.dataset.targets.numpy()

    w1_float = model.fc1.weight.T.detach().cpu().numpy()
    w2_float = model.fc2.weight.T.detach().cpu().numpy()
    w3_float = model.fc3.weight.T.detach().cpu().numpy()
    from src.utils.io import quantize_weight_int4

    w1_int4 = quantize_weight_int4(w1_float)
    w2_int4 = quantize_weight_int4(w2_float)
    w3_int4 = quantize_weight_int4(w3_float)

    acc, preds, scale_h1, scale_h2 = run_inference(
        all_images, all_labels, w1_int4, w2_int4, w3_int4
    )
    print(
        f"NumPy inference accuracy: {acc:.2f}% | "
        f"scale_h1 = {scale_h1:.6f} | scale_h2 = {scale_h2:.6f}"
    )

    if acc > 90.0:
        save_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/ste")
        print("Exporting weights...")
        export_weights_steq(model, save_dir=save_dir)
        np.save(
            os.path.join(save_dir, "steq_quant_params.npy"),
            {"scale_h1": scale_h1, "scale_h2": scale_h2},
        )
        print("Saved: w1_int4.npy, w2_int4.npy, w3_int4.npy, steq_quant_params.npy")
    else:
        print("Accuracy too low, weights not saved.")


if __name__ == "__main__":
    main()
