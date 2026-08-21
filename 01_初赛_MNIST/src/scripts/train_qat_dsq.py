"""DSQ QAT training script."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.qat_dsq import PhotonicMLP_DSQ
from src.data.loaders import get_mnist_loaders_torchvision
from src.training.common import train_epoch
from src.utils.io import export_weights_dsq
from src.inference.numpy_dsq import run_inference


def main():
    epochs = 15
    batch_size = 256
    init_temp = 1.0
    target_temp = 20.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.path.join(os.path.dirname(__file__), "../../data/raw")
    train_loader, test_loader = get_mnist_loaders_torchvision(batch_size=batch_size, root=data_root)
    model = PhotonicMLP_DSQ(hidden_dim1=128, hidden_dim2=64).to(device)

    base_params = [p for n, p in model.named_parameters() if "scale" not in n]
    quant_params = [p for n, p in model.named_parameters() if "scale" in n]

    optimizer = optim.AdamW([
        {"params": base_params, "lr": 0.002, "weight_decay": 1e-4},
        {"params": quant_params, "lr": 0.0005},
    ])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting DSQ QAT training ({epochs} epochs) on {device}...")
    for epoch in range(epochs):
        progress = epoch / max(1, epochs - 1)
        current_temp = init_temp * (target_temp / init_temp) ** progress
        model.set_temperature(current_temp)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        qi = model.get_quantization_info()
        print(
            f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
            f"Temp: {current_temp:.1f} | W1_Sc: {qi['w1_scale']:.5f}"
        )
        scheduler.step()

    print("\nValidating with NumPy inference...")
    model.eval()
    q_info = model.get_quantization_info()

    from src.utils.io import quantize_weight_with_scale

    w1_int4 = quantize_weight_with_scale(model.fc1.weight.T, q_info["w1_scale"], 0, 4, True)
    w2_int4 = quantize_weight_with_scale(model.fc2.weight.T, q_info["w2_scale"], 0, 4, True)
    w3_int4 = quantize_weight_with_scale(model.fc3.weight.T, q_info["w3_scale"], 0, 4, True)

    all_images = test_loader.dataset.data.numpy() / 255.0
    all_labels = test_loader.dataset.targets.numpy()

    acc, preds = run_inference(all_images, all_labels, w1_int4, w2_int4, w3_int4, q_info)
    print(f"NumPy inference accuracy: {acc:.2f}%")

    if acc > 90.0:
        save_dir = os.path.join(os.path.dirname(__file__), "../../artifacts/dsq")
        print("Exporting weights...")
        export_weights_dsq(model, save_dir=save_dir)
        print("Saved: w1_int4_dsq.npy, w2_int4_dsq.npy, w3_int4_dsq.npy, dsq_quant_params.npy")
    else:
        print("Accuracy too low, weights not saved.")


if __name__ == "__main__":
    main()
