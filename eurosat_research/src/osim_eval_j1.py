"""
===============================================================================
 osim_eval_j1.py — J1 架构 osimulator 真机验证 (q500)
===============================================================================
 J1 = MiniVGG-GAP 变体: stem s2 + 全 1×1 + fast_downsample, 全层光计算 (无 stem 特判)

 用法 (容器内):
   python eurosat_research/src/osim_eval_j1.py --weight <path> --quick 500
   python eurosat_research/src/osim_eval_j1.py --weight <path> --qat      # QAT 伪量化对照
===============================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "..", "src", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "..", "src", "core"))

from models import MiniVGG
from data import load_eurosat


def build_j1(weight_path, device):
    """构建 J1 架构 + 加载 QAT 训练权重 (全光计算, 无 stem 特判)。"""
    model = MiniVGG(num_classes=10, channels=[16, 32, 64, 128],
                    stem_stride=2, fast_downsample=True,
                    kernels=(1, 1, 1), head_dims=[128])
    state_dict = torch.load(weight_path, map_location="cpu")
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    print(f"[build_j1] loaded {len(filtered)}/{len(state_dict)} params from {weight_path}")
    model.to(device)
    return model


def eval_qat(model, loader, device):
    """QAT 伪量化模式 (与训练一致的量化, 无噪声)。"""
    from qat_v5 import prepare_model_v5, enable_qat, disable_qat
    # 训练时的 QAT 层已存权重; 需要重建 QAT 层才能施加量化
    model_q = MiniVGG(num_classes=10, channels=[16, 32, 64, 128],
                      stem_stride=2, fast_downsample=True,
                      kernels=(1, 1, 1), head_dims=[128])
    model_q.load_state_dict(model.state_dict())
    prepare_model_v5(model_q, output_noise=False, output_quant=True,
                     activation_style="osim")
    model_q.to(device)
    model_q.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model_q(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def eval_optic(model, loader, device, quick):
    """osimulator 真机模式。"""
    from optic_layers import OpticalEngine, build_optical_model, evaluate_model

    engine = OpticalEngine(use_real=True)
    # 全层光计算 (J1 无 stem 特判 — 用户决策: stem 电计算是扯淡优化)
    build_optical_model(model, engine, pad_to_8=True,
                        input_bit=8, weight_bit=8,
                        keep_first_conv_electronic=False)
    from optic_layers import print_alignment_detail
    print_alignment_detail(model, "J1 (Optical)")

    t0 = time.time()
    result = evaluate_model(model, loader, device,
                            criterion=nn.CrossEntropyLoss(),
                            max_batches=quick,
                            desc="J1 optic (osimulator)",
                            print_interval=max(1, (quick or len(loader)) // 10))
    elapsed = time.time() - t0
    print(f"  Optical Accuracy: {result['accuracy']:.2%}")
    print(f"  Optical Time:     {elapsed:.1f}s ({elapsed/(quick or len(loader)):.1f}s/img)")
    return result["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="path to best.pth")
    ap.add_argument("--quick", type=int, default=500)
    ap.add_argument("--qat", action="store_true", help="QAT pseudo-quant mode")
    ap.add_argument("--data", default="data/EuroSAT_RGB")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_eurosat(args.data, batch_size=args.batch,
                                     aug="none", num_workers=4)

    model = build_j1(args.weight, device)

    if args.qat:
        acc = eval_qat(model, test_loader, device)
        print(f"\n[J1 QAT] test acc = {acc:.2%}")
    else:
        acc = eval_optic(model, test_loader, device, args.quick)
        print(f"\n[J1 OPTIC] test acc = {acc:.2%}")


if __name__ == "__main__":
    main()
