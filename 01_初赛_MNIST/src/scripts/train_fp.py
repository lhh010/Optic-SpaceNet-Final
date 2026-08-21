"""Full-precision training script."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base import PhotonicMLP
from src.data.loaders import get_mnist_loaders_local
from src.training.common import train_epoch, evaluate
from src.utils.io import export_weights_steq


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    train_loader, test_loader = get_mnist_loaders_local(batch_size=256, data_dir=data_dir)

    model = PhotonicMLP(hidden_dim1=128, hidden_dim2=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 3
    print(f"Starting full-precision training ({epochs} epochs) on {device}...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    print("\nExporting weights...")
    save_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/ste")
    export_weights_steq(model, save_dir=save_dir)
    print("Saved: w1_int4.npy, w2_int4.npy, w3_int4.npy")


if __name__ == "__main__":
    main()
