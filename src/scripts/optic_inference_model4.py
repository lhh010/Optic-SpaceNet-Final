"""
optic_inference_model4.py — MiniVGG-GAP osimulator 光计算推理评估

模型: Model 4 MiniVGG-GAP (FP32, 260K params, 96.65% val)
架构: 7×3×3 Conv + GAP head, bias=False (conv) / bias=True (final Linear)
脚本: src/training/model4_minivgg_gap.py
权重: weights/minivgg_gap.pth

用法 (在光计算 Docker 容器内):
  /local/miniconda/envs/moca_llm/bin/python src/scripts/optic_inference_model4.py
  /local/miniconda/envs/moca_llm/bin/python src/scripts/optic_inference_model4.py --quick 100
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

from optic_layers import (
    OpticalEngine, OpticConv2d, OpticLinear,
    build_optical_model,
    compute_alignment_ratio, print_alignment_detail,
    evaluate_model,
)

DATA_DIR = "data/EuroSAT_RGB"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
WEIGHT_PATH = "weights/minivgg_gap.pth"
DEFAULT_BATCH = 1

print(f"Device: {DEVICE}")


class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def load_test_data(batch_size=DEFAULT_BATCH, test_ratio=0.2):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    from eurosat_split import split_indices
    _, _, test_indices = split_indices(len(full_dataset),
                                        val_ratio=test_ratio, test_ratio=test_ratio)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    print(f"Test set: {len(test_dataset)} images, batch={batch_size}")
    return test_loader


def main():
    quick = None
    if "--quick" in sys.argv:
        idx = sys.argv.index("--quick")
        if idx + 1 < len(sys.argv):
            quick = int(sys.argv[idx + 1])

    print(f"\n{'='*60}")
    print(f"  Model 4 MiniVGG-GAP — Osimulator Optical Inference")
    print(f"{'='*60}")
    print(f"  Weights: {WEIGHT_PATH}")
    print(f"  Quick:   {quick if quick else 'full test set'}")

    # Data
    batch_size = DEFAULT_BATCH
    test_loader = load_test_data(batch_size=batch_size)
    if quick:
        from torch.utils.data import Subset
        test_loader = DataLoader(
            Subset(test_loader.dataset, list(range(min(quick, len(test_loader.dataset))))),
            batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"  => Quick mode: {min(quick, len(test_loader.dataset))} images")

    # Native model
    model_native = MiniVGG(num_classes=NUM_CLASSES)
    state_dict = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=True)
    model_native.load_state_dict(state_dict)
    model_native.to(DEVICE)
    n_params = sum(p.numel() for p in model_native.parameters())
    print(f"  Params: {n_params:,}")

    # Native eval
    print(f"\n[Native FP32 eval]")
    t0 = time.time()
    criterion = nn.CrossEntropyLoss()
    result_native = evaluate_model(model_native, test_loader, DEVICE, criterion=criterion)
    native_time = time.time() - t0
    print(f"  Native Accuracy: {result_native['accuracy']:.2%}")
    print(f"  Native Time:     {native_time:.1f}s")

    # Alignment
    print(f"\n[Hardware alignment]")
    print_alignment_detail(model_native, "MiniVGG-GAP (Native)")

    # Optical model
    print(f"\n[Optical model]")
    engine = OpticalEngine(use_real=True)
    model_optic = MiniVGG(num_classes=NUM_CLASSES)
    model_optic.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu', weights_only=True))
    model_optic.to(DEVICE)
    build_optical_model(model_optic, engine, pad_to_8=True)
    print_alignment_detail(model_optic, "MiniVGG-GAP (Optical)")

    # Optical eval
    print(f"\n[Optical osimulator eval]")
    t0 = time.time()
    result_optic = evaluate_model(model_optic, test_loader, DEVICE, criterion=criterion)
    optic_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Model:        MiniVGG-GAP")
    print(f"  Params:       {n_params:,}")
    print(f"  Native Acc:   {result_native['accuracy']:.2%}")
    print(f"  Optical Acc:  {result_optic['accuracy']:.2%}")
    print(f"  Gap:          {(result_native['accuracy'] - result_optic['accuracy'])*100:.2f} pp")
    print(f"  Native Time:  {native_time:.1f}s")
    print(f"  Optical Time: {optic_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
