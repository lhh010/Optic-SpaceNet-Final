"""
optic_inference_model4.py — MiniVGG-GAP osimulator 光计算推理评估

模型: Model 4 MiniVGG-GAP (260K params)
  - FP32 基线:      src/training/model4_minivgg_gap.py → weights/minivgg_gap.pth (val 96.65%)
  - QAT int8 版本:  src/training/model4_minivgg_gap_phase4_v3.py
                    → weights/minivgg_gap_phase4_v3_int8.pth (int8 test 95.50%)
架构: 7×3×3 Conv + GAP head, bias=False (conv) / bias=True (final Linear)

★ v4.1 修复: 本脚本原先只评估 FP32 基线权重, QAT int8 权重
  (minivgg_gap_phase4_v3_int8.pth) 从未在 osimulator 上验证 — QAT 管线闭环缺失。
  现支持:
    - QAT 模式 (--mode qat): prepare_model_v4 + enable_qat 伪量化, 与训练配置一致
    - Optic 模式 (--mode optic): build_optical_model + osimulator 真硬件仿真
      可加载 QAT int8 权重, 完成"训练→真机"闭环

用法 (在光计算 Docker 容器内):
  # FP32 基线 → osimulator (旧行为)
  /local/miniconda/envs/moca_llm/bin/python src/scripts/optic_inference_model4.py
  /local/miniconda/envs/moca_llm/bin/python src/scripts/optic_inference_model4.py --quick 100

  # QAT int8 权重 → QAT 伪量化 + osimulator 真硬件 (★ 修复后的闭环)
  /local/miniconda/envs/moca_llm/bin/python src/scripts/optic_inference_model4.py \
      --weights weights/minivgg_gap_phase4_v3_int8.pth --mode qat
  /local/miniconda/envs/moca_llm/bin/python src/scripts/optic_inference_model4.py \
      --weights weights/minivgg_gap_phase4_v3_int8.pth --mode optic --quick 100

  # stem 策略与训练保持一致 (MODEL4_FIRST_CONV_FP32=0 训练时推理也必须用 --first-conv-fp32 0)
  ... --first-conv-fp32 0

  TODO (v4.1 修复后, 见 docs/TODO.md §v4.1 重跑清单):
    - [ ] 用重训后的 QAT int8 权重在容器内跑 --mode qat + --mode optic,
          首次补齐 Model 4 的 "训练→真机" 闭环 (此前 QAT 权重从未上 osimulator)
    - [ ] 若训练改用 MODEL4_FIRST_CONV_FP32=0 (stem 光计算), 推理必须同步 --first-conv-fp32 0,
          否则 stem 训练/推理不一致 (重蹈 §16 BN 偏移教训)
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
WEIGHT_PATH = "weights/minivgg_gap.pth"              # FP32 基线
QAT_WEIGHT_PATH = "weights/minivgg_gap_phase4_v3_int8.pth"  # QAT int8
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


def _load_weights_into_plain(model, weight_path):
    """加载权重到原生 MiniVGG (过滤 QAT 特有键, 如 _qat_enabled 等 buffer)."""
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    skipped = len(state_dict) - len(filtered)
    if skipped:
        print(f"  Skipped {skipped} QAT-specific params (expected for QAT weights)")
    return model


# ============================================================
#  QAT 模式: PyTorch 伪量化 (与训练配置一致, 无 osimulator)
# ============================================================
def evaluate_qat_mode(test_loader, weight_path, quick, first_conv_fp32=True):
    from optic_qat_v4 import prepare_model_v4, enable_qat, disable_qat

    print(f"\n{'='*60}")
    print(f"  Model 4 — QAT int8 伪量化评估 (无 osimulator)")
    print(f"{'='*60}")

    model = MiniVGG(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # 与训练完全一致的转换配置 (噪声不注入 — 推理时不加噪声)
    print(f"  [QAT v4] w8/a8 (uint8+zp 激活), stem "
          f"{'FP32' if first_conv_fp32 else 'int8 QAT'}")
    prepare_model_v4(model, weight_bits=8, act_bits=8, noise=False,
                     first_conv_fp32=first_conv_fp32,
                     quantize_linear=True, preserve_bn=True)
    print_alignment_detail(model, "MiniVGG-GAP (QAT v4)")

    _load_weights_into_plain(model, weight_path)
    model.to(DEVICE)

    # float32 (QAT 关闭)
    disable_qat(model)
    r_fp32 = evaluate_model(model, test_loader, DEVICE,
                            criterion=nn.CrossEntropyLoss(),
                            max_batches=quick,
                            desc="QAT float32")
    # int8 (QAT 开启)
    enable_qat(model)
    r_int8 = evaluate_model(model, test_loader, DEVICE,
                            criterion=nn.CrossEntropyLoss(),
                            max_batches=quick,
                            desc="QAT int8")
    print(f"\n  QAT Float32: {r_fp32['accuracy']:.2%}")
    print(f"  QAT Int8:    {r_int8['accuracy']:.2%}")
    print(f"  量化损失:    {r_fp32['accuracy'] - r_int8['accuracy']:+.2%}")
    return {"fp32": r_fp32["accuracy"], "int8": r_int8["accuracy"]}


# ============================================================
#  Optic 模式: build_optical_model + osimulator 真硬件仿真
# ============================================================
def evaluate_optic_mode(test_loader, weight_path, quick, first_conv_fp32=True):
    print(f"\n{'='*60}")
    print(f"  Model 4 — Optic osimulator 真硬件仿真")
    print(f"{'='*60}")

    engine = OpticalEngine(use_real=True)
    model = MiniVGG(num_classes=NUM_CLASSES)
    _load_weights_into_plain(model, weight_path)
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    print_alignment_detail(model, "MiniVGG-GAP (Native)")

    # 转换为光计算模型 — stem 策略必须与训练一致
    #   first_conv_fp32=True  → keep_first_conv_electronic=True (默认, 与 M2/M3 部署原则一致)
    #   first_conv_fp32=False → stem 也走光计算
    print(f"\n  [build_optical_model] int8, stem "
          f"{'electronic FP32' if first_conv_fp32 else 'optical'}")
    build_optical_model(model, engine, pad_to_8=True,
                        input_bit=8, weight_bit=8,
                        keep_first_conv_electronic=first_conv_fp32)
    print_alignment_detail(model, "MiniVGG-GAP (Optical)")

    print(f"\n  [osimulator eval] ...")
    total_batches = quick if quick else len(test_loader)
    print_interval = 1 if quick else max(1, total_batches // 10)
    t0 = time.time()
    result = evaluate_model(model, test_loader, DEVICE,
                            criterion=nn.CrossEntropyLoss(),
                            max_batches=quick,
                            desc="Model 4 optic",
                            print_interval=print_interval)
    optic_time = time.time() - t0
    print(f"\n  Optical Accuracy: {result['accuracy']:.2%}")
    print(f"  Optical Time:     {optic_time:.1f}s")
    return {"optic": result["accuracy"], "time": optic_time}


def main():
    # ---- 参数解析 ----
    args = sys.argv[1:]
    quick = None
    weight_path = WEIGHT_PATH
    mode = "auto"          # auto | qat | optic
    first_conv_fp32 = True # 与 MODEL4_FIRST_CONV_FP32 训练默认一致
    if "--quick" in args:
        idx = args.index("--quick")
        if idx + 1 < len(args):
            quick = int(args[idx + 1])
    if "--weights" in args:
        idx = args.index("--weights")
        if idx + 1 < len(args):
            weight_path = args[idx + 1]
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            mode = args[idx + 1].lower()
    if "--first-conv-fp32" in args:
        idx = args.index("--first-conv-fp32")
        if idx + 1 < len(args):
            first_conv_fp32 = args[idx + 1].strip().lower() not in ("0", "false", "no")

    # auto: QAT int8 权重默认走 qat+optic 闭环; FP32 基线走 optic (旧行为)
    if mode == "auto":
        mode = "qat" if "phase4" in os.path.basename(weight_path) else "optic"

    print(f"\n{'='*60}")
    print(f"  Model 4 MiniVGG-GAP — Osimulator Optical Inference")
    print(f"{'='*60}")
    print(f"  Weights:     {weight_path}")
    print(f"  Mode:        {mode}")
    print(f"  Stem:        {'FP32 electronic' if first_conv_fp32 else 'int8 optical'}")
    print(f"  Quick:       {quick if quick else 'full test set'}")

    # 数据
    batch_size = DEFAULT_BATCH
    test_loader = load_test_data(batch_size=batch_size)
    if quick:
        from torch.utils.data import Subset
        test_loader = DataLoader(
            Subset(test_loader.dataset, list(range(min(quick, len(test_loader.dataset))))),
            batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"  => Quick mode: {min(quick, len(test_loader.dataset))} images")

    # 原生 FP32 参考 (任意权重文件均可)
    model_native = MiniVGG(num_classes=NUM_CLASSES)
    _load_weights_into_plain(model_native, weight_path)
    model_native.to(DEVICE)
    n_params = sum(p.numel() for p in model_native.parameters())
    print(f"  Params: {n_params:,}")

    print(f"\n[Native FP32 eval]")
    criterion = nn.CrossEntropyLoss()
    result_native = evaluate_model(model_native, test_loader, DEVICE, criterion=criterion)
    print(f"  Native Accuracy: {result_native['accuracy']:.2%}")

    # 对齐率
    print(f"\n[Hardware alignment (native)]")
    print_alignment_detail(model_native, "MiniVGG-GAP (Native)")

    # 评估
    result_qat = None
    result_optic = None
    if mode == "qat":
        result_qat = evaluate_qat_mode(test_loader, weight_path, quick, first_conv_fp32)
    result_optic = evaluate_optic_mode(test_loader, weight_path, quick, first_conv_fp32)

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  Results — Model 4 ({os.path.basename(weight_path)})")
    print(f"{'='*60}")
    print(f"  Native FP32 Acc:   {result_native['accuracy']:.2%}")
    if result_qat:
        print(f"  QAT Int8 Acc:      {result_qat['int8']:.2%}  (量化损失 "
              f"{result_qat['fp32'] - result_qat['int8']:+.2%})")
    print(f"  Optical osimulator: {result_optic['optic']:.2%}  "
          f"({result_optic['time']:.1f}s)")
    print(f"  Gap (native→optic): "
          f"{(result_native['accuracy'] - result_optic['optic'])*100:.2f} pp")
    print(f"{'='*60}")
    return result_optic["optic"]


if __name__ == "__main__":
    main()
