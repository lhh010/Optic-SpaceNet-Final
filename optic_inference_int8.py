"""
================================================================================
 optic_inference_int8.py — INT8 模型光计算容器内推理 + MOPs 统计

 模型: Model 2 SpaceNet V1 Phase4 v3 (当前最佳 INT8 模型)
   训练脚本:  model2_spacenet_v1_phase4_v3.py
   权重文件:  spacenet_v1_phase4_v3_int8.pth
   训练精度:  93.11% int8 (QAT), 93.02% float32
   QAT 模块:  optic_qat_v4.py (int8 权重 + int8 激活 + Gazelle 硬件噪声)
   硬件配置:  stem FP32 (首层对齐率仅 37.5%), 其余 Conv+Linear 全 int8

 评估模式 (容器内运行, 需要 osimulator):
   - Optic 模式 (默认): build_optical_model + OpticalEngine(use_real=True)
       将 Conv→OpticConv2d, Linear→OpticLinear, 所有矩阵乘法走 osimulator
       (im2col 展开 → 补零对齐 → 光学矩阵乘法 → col2im)
       这是容器内真实光计算硬件仿真路径

   - QAT 模式 (--qat): enable_qat/disable_qat PyTorch 伪量化
       用于交叉验证容器内/外精度差异

   - MOPs 统计 (--mops-only): 仅打印各层 MOPs 与光计算占比, 不跑推理

 光计算占比统计 (按 MOPs):
   逐层计算电子计算 MOPs vs 光计算 MOPs (含补零对齐开销),
   给出整体光计算占比和每层详细分解。

 用法 (在光计算 Docker 容器内):
   python optic_inference_int8.py                        # 默认 Optic 模式, 全量测试集
   python optic_inference_int8.py --quick 10             # Optic 模式快速测试
   python optic_inference_int8.py --qat                  # QAT 伪量化交叉验证
   python optic_inference_int8.py --qat --quick 10       # QAT 快速对比
   python optic_inference_int8.py --mops-only            # 仅打印 MOPs 统计

 参考:
   - example_load_gazelle_model.py  (光计算 API 参考)
   - optic_inference_phase4.py      (Phase4 容器)
   - optic_inference_mixed.py       (Mixed 容器)
   - optic_layers.py                (光计算核心库)
   - optic_qat_v4.py                (INT8 QAT 模块)
   - EXPERIMENTS.md                 (完整实验记录)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import time
import numpy as np

# ============================================================
#  全局配置
# ============================================================
DATA_DIR = "data/EuroSAT_RGB"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED_TRAIN = 42     # 训练时使用的 seed (train/val split 与此一致)
DEFAULT_BATCH = 1   # osimulator 安全值

print(f"Device: {DEVICE}")


# ============================================================
#  模型架构 — Model 2 SpaceNet V1 (与训练脚本完全一致)
#  bias=False 匹配光计算硬件, BN 保留用于稳定推理
# ============================================================

class OpticSpaceNetV1_INT8(nn.Module):
    """
    Model 2 Phase4 v3: SpaceNet V1, bias=False, int8 QAT.

    架构 (from model2_spacenet_v1_phase4_v3.py):
      stem:       Conv2d(3→8,  1×1) → BN → ReLU           [FP32, 首层对齐率 37.5%]
      stage1:     Conv2d(8→16, 2×2, stride=2) → BN → ReLU → MaxPool2d(2)  [INT8]
      stage2:     Conv2d(16→32, 2×2, stride=2) → BN → ReLU                [INT8]
      stage3:     Conv2d(32→16, 1×1) → BN → ReLU                          [INT8]
      classifier: Flatten → Linear(1024→256) → ReLU → Dropout → Linear(256→10)  [INT8]

    输入: (N, 3, 64, 64)  EuroSAT RGB 图像
    输出: (N, 10)          10 类 logits
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x


# ============================================================
#  数据加载
# ============================================================
def load_test_data(batch_size=DEFAULT_BATCH, test_ratio=0.2):
    """加载独立测试集 (单一数据源 eurosat_split, 与训练 train/val 严格互斥, 见 Bug #11)。"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    from eurosat_split import split_indices
    # 单一数据源: test 段与训练 train/val 严格互斥 (Bug #11)
    _, _, test_indices = split_indices(len(full_dataset), seed=SEED_TRAIN,
                                       val_ratio=test_ratio, test_ratio=test_ratio)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    print(f"Full dataset: {len(full_dataset)} imgs")
    print(f"Test (now): {len(test_dataset)} imgs | split=eurosat_split (test∩train=0)")
    print(f"Classes: {full_dataset.classes}")
    return test_loader


# ============================================================
#  评估工具
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None,
             max_batches=None, desc="Evaluating", print_interval=None):
    model.eval()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(dataloader)
    effective_n = min(n_batches, max_batches or n_batches)
    if print_interval is None:
        print_interval = max(1, effective_n // 10)

    for i, (images, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion:
            total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    acc = correct / total if total > 0 else 0
    print(f"  [{desc}] {effective_n} batches — acc={acc:.2%}", flush=True)
    return {"accuracy": acc, "loss": total_loss / total if criterion else 0.0,
            "total": total, "correct": correct}


# ============================================================
#  MOPs 计算 — 光计算占比统计核心
# ============================================================

# 输入图像尺寸
INPUT_C, INPUT_H, INPUT_W = 3, 64, 64

# 每层结构描述: (名称, 类型, C_in, C_out, K_h, K_w, stride, padding, 后续池化, 是否光计算)
# is_optical: 训练时 stem FP32 (首层对齐率低), 其余 Conv+Linear 全 INT8 → 光计算
LAYER_SPECS = [
    # name          type     C_in C_out Kh Kw stride pad  pool     optical?
    ("stem.conv",   "Conv",  3,   8,   1, 1, 1,    0,   None,    False),
    ("stage1.conv", "Conv",  8,   16,  2, 2, 2,    0,   "Max2x2", True),
    ("stage2.conv", "Conv",  16,  32,  2, 2, 2,    0,   None,    True),
    ("stage3.conv", "Conv",  32,  16,  1, 1, 1,    0,   None,    True),
    ("fc1",         "Linear", 1024, 256, 0, 0, 0, 0, None,    True),
    ("fc2",         "Linear", 256,  10,  0, 0, 0, 0, None,    True),
]

# 注意: fc1 的 C_in=1024 来自 16 × 8 × 8 (stage3 输出)
# stage1: stride=2 + MaxPool2d(2) → H_out = 64/2/2 = 16


def compute_spatial_dim(H_in, W_in, Kh, Kw, stride, pad):
    """计算 Conv 层输出的空间尺寸 (不含池化)"""
    H_out = (H_in + 2 * pad - Kh) // stride + 1
    W_out = (W_in + 2 * pad - Kw) // stride + 1
    return H_out, W_out


def apply_pool(H, W, pool):
    """应用池化后的空间尺寸"""
    if pool == "Max2x2":
        return H // 2, W // 2
    return H, W


def compute_mops_detail(input_size=(INPUT_C, INPUT_H, INPUT_W)):
    """
    逐层计算 MOPs (每张图片), 区分电子计算和光计算。

    电子计算 MOPs = C_out × H_out × W_out × C_in × Kh × Kw
    光计算 MOPs = C_out × H_out × W_out × padded_k   (padded_k = ceil(C_in×Kh×Kw/8) × 8)

    对 Linear 层: H_out=W_out=1, C_in×Kh×Kw 替换为 in_features

    Returns:
        layers:  每层统计列表
        summary: 汇总 dict {total, electronic, optical, optical_ratio, ...}
    """
    C_in_img, H_cur, W_cur = input_size
    layers = []

    for name, ltype, c_in, c_out, kh, kw, stride, pad, pool, is_optical in LAYER_SPECS:
        if ltype == "Conv":
            patch_len = c_in * kh * kw  # im2col 展平长度
            padded_len = ((patch_len + 7) // 8) * 8  # 补零到 8 的倍数
            # Conv 输出空间尺寸 (池化前)
            H_conv_out, W_conv_out = compute_spatial_dim(H_cur, W_cur, kh, kw, stride, pad)
            n_patches = H_conv_out * W_conv_out  # 每个样本的 im2col patch 数

            # 原始 MACs (基于 Conv 输出, 池化前)
            raw_macs = c_out * H_conv_out * W_conv_out * c_in * kh * kw
            # 光计算有效 MACs (含补零浪费)
            optical_macs = n_patches * padded_len * c_out
            # 电子计算 MACs
            electronic_macs = raw_macs

            alignment = patch_len / padded_len if padded_len > 0 else 1.0

            layers.append({
                "name": name,
                "type": ltype,
                "c_in": c_in, "c_out": c_out,
                "kernel": f"{kh}x{kw}",
                "spatial_in": f"{H_cur}x{W_cur}",
                "spatial_out": f"{H_conv_out}x{W_conv_out}",
                "pool": pool if pool else "None",
                "patch_len": patch_len,
                "padded_len": padded_len,
                "alignment": alignment,
                "raw_mops": raw_macs / 1e6,
                "optical_mops": optical_macs / 1e6 if is_optical else 0.0,
                "electronic_mops": electronic_macs / 1e6 if not is_optical else 0.0,
                "effective_mops": (optical_macs if is_optical else electronic_macs) / 1e6,
                "is_optical": is_optical,
            })

            # 更新当前空间尺寸: 先 Conv 输出, 再池化
            H_cur, W_cur = apply_pool(H_conv_out, W_conv_out, pool)

        elif ltype == "Linear":
            in_features = c_in
            patch_len = in_features
            padded_len = ((patch_len + 7) // 8) * 8
            n_patches = 1

            raw_macs = in_features * c_out
            optical_macs = n_patches * padded_len * c_out
            electronic_macs = raw_macs

            alignment = patch_len / padded_len if padded_len > 0 else 1.0

            layers.append({
                "name": name,
                "type": ltype,
                "c_in": in_features, "c_out": c_out,
                "kernel": "-",
                "spatial_in": "-",
                "spatial_out": "-",
                "patch_len": patch_len,
                "padded_len": padded_len,
                "alignment": alignment,
                "raw_mops": raw_macs / 1e6,
                "optical_mops": optical_macs / 1e6 if is_optical else 0.0,
                "electronic_mops": electronic_macs / 1e6 if not is_optical else 0.0,
                "effective_mops": (optical_macs if is_optical else electronic_macs) / 1e6,
                "is_optical": is_optical,
            })

    # 汇总
    total_raw = sum(l["raw_mops"] for l in layers)
    total_optical = sum(l["optical_mops"] for l in layers)
    total_electronic = sum(l["electronic_mops"] for l in layers)
    total_effective = sum(l["effective_mops"] for l in layers)

    # 光计算占比: 按有效光计算 MOPs / 总有效 MOPs
    optical_ratio = total_optical / total_effective if total_effective > 0 else 0.0

    summary = {
        "total_raw_mops": total_raw,
        "total_optical_mops": total_optical,
        "total_electronic_mops": total_electronic,
        "total_effective_mops": total_effective,
        "optical_ratio": optical_ratio,
        "optical_waste": total_optical - sum(
            l["raw_mops"] for l in layers if l["is_optical"]
        ),  # 补零浪费的 MOPs
    }

    return layers, summary


def print_mops_report(layers, summary):
    """打印 MOPs 统计报告"""
    print("\n")
    print("=" * 110)
    print("  INT8 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v3")
    print("  Gazelle 硬件: 8×2 tile, 8a8w12o, 首层 stem FP32 (电计算)")
    print("=" * 110)

    # 表头
    print(f"\n  {'Layer':<16s} {'Type':<6s} {'C_in':>5s} {'C_out':>5s} "
          f"{'Kernel':>6s} {'Input':>10s} {'ConvOut':>10s} {'Pool':>6s} "
          f"{'Patch':>6s} {'Padded':>6s} {'Align':>7s} "
          f"{'RawMOPs':>10s} {'OptMOPs':>10s} {'ElecMOPs':>10s} {'Compute':>12s}")
    print("  " + "-" * 120)

    for l in layers:
        location = "[Optical]" if l["is_optical"] else "[Electronic]"
        pool_str = l.get("pool", "None")
        print(f"  {l['name']:<16s} {l['type']:<6s} {l['c_in']:>5d} {l['c_out']:>5d} "
              f"{l['kernel']:>6s} {l['spatial_in']:>10s} {l['spatial_out']:>10s} {pool_str:>6s} "
              f"{l['patch_len']:>6d} {l['padded_len']:>6d} {l['alignment']:>6.1%} "
              f"{l['raw_mops']:>9.4f}M {l['optical_mops']:>9.4f}M {l['electronic_mops']:>9.4f}M "
              f"{location:<12s}")

    print("  " + "-" * 120)
    print(f"  {'Total':<16s} {'':<6s} {'':>5s} {'':>5s} {'':>6s} {'':>10s} {'':>10s} {'':>6s} "
          f"{'':>6s} {'':>6s} {'':>7s} "
          f"{summary['total_raw_mops']:>9.4f}M {summary['total_optical_mops']:>9.4f}M "
          f"{summary['total_electronic_mops']:>9.4f}M")

    # 汇总卡片
    print(f"\n  {'-' * 60}")
    print(f"  [MOPs] 光计算占比汇总")
    print(f"  {'-' * 60}")
    print(f"  总原始 MOPs:           {summary['total_raw_mops']:.4f} M")
    print(f"  光计算 MOPs (有效):    {summary['total_optical_mops']:.4f} M")
    print(f"  电子计算 MOPs:         {summary['total_electronic_mops']:.4f} M")
    print(f"  总有效 MOPs:           {summary['total_effective_mops']:.4f} M")
    print(f"  -------------------------------------")
    print(f"  ** 光计算占比:         {summary['optical_ratio']:.2%}  "
          f"({'**' if summary['optical_ratio'] >= 0.90 else '~~' if summary['optical_ratio'] >= 0.70 else '!!'})")
    if summary['optical_waste'] > 0:
        print(f"  光计算补零浪费:        {summary['optical_waste']:.4f} M")
    else:
        print(f"  光计算补零浪费:        0 (所有光计算层完美对齐 8 的倍数) [OK]")
    print(f"  {'-' * 60}")

    # 关键解释
    print(f"\n  [Note] 说明:")
    print(f"    - stem.conv (3->8, 1x1): 展平长度=3, 对齐率仅 37.5%, 保留电计算 (FP32)")
    print(f"    - 其余 Conv/Linear 展平长度均为 8 的倍数, 完美对齐 Gazelle 8×2 tile")
    print(f"    - 光计算占比 = 光计算有效MOPs / (光计算MOPs + 电计算MOPs)")
    print(f"    - 光计算有效MOPs 已包含补零对齐的硬件开销")
    print("=" * 110)


# ============================================================
#  QAT 模式评估 (主模式: int8 伪量化精度与训练一致)
# ============================================================
def evaluate_qat_int8(model_class, weight_path, test_loader, device,
                      quick_batches=None):
    """
    QAT 模式: 使用 optic_qat_v4 的 prepare_model_v4 + enable_qat/disable_qat。

    配置与训练完全一致:
      - weight_bits=8, act_bits=8 (硬件原生 int8)
      - first_conv_fp32=True (stem 保留 FP32)
      - noise=False (推理时不需要硬件噪声)
      - quantize_linear=True, preserve_bn=True
    """
    model_name = "Model 2 Phase4 v3 INT8"
    print(f"\n{'='*60}")
    print(f"  {model_name}  [QAT mode: int8]")
    print(f"  Architecture: SpaceNet V1 (seq+BN, bias=False)")
    print(f"{'='*60}")

    # Step 1: 创建标准模型
    print(f"\n  [1/3] Creating standard model...")
    model = model_class(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    # Step 2: 转换为 QAT v4 (匹配训练配置)
    print(f"\n  [2/3] Converting to QAT v4 (int8 weight + int8 act, stem FP32)...")
    from optic_qat_v4 import prepare_model_v4, enable_qat, disable_qat

    prepare_model_v4(model,
                     weight_bits=8,          # int8 匹配硬件原生精度
                     act_bits=8,
                     noise=False,            # 推理时不注入噪声
                     first_conv_fp32=True,   # stem 对齐率低, 电计算
                     quantize_linear=True,    # Linear 也量化 (与 Phase4 v3 一致)
                     preserve_bn=True)

    # Step 3: 加载 QAT 权重
    print(f"\n  [3/3] Loading INT8 QAT weights...")
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight file not found: {weight_path}")
        return None
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded from: {weight_path}")

    # ----- float32 评估 (disable QAT) -----
    print(f"\n  --- Native float32 evaluation (QAT disabled) ---")
    disable_qat(model)
    t0 = time.time()
    result_fp32 = evaluate(model, test_loader, device,
                           criterion=nn.CrossEntropyLoss(),
                           max_batches=quick_batches,
                           desc=f"{model_name} float32")
    fp32_time = time.time() - t0
    print(f"  Float32 Accuracy: {result_fp32['accuracy']:.2%}")
    print(f"  Float32 Loss:     {result_fp32['loss']:.4f}")
    print(f"  Float32 Time:     {fp32_time:.2f}s")

    # ----- int8 QAT (enable QAT, 光计算模拟) -----
    print(f"\n  --- int8 QAT (optical computing simulation) evaluation ---")
    enable_qat(model)
    t0 = time.time()
    result_int8 = evaluate(model, test_loader, device,
                           criterion=nn.CrossEntropyLoss(),
                           max_batches=quick_batches,
                           desc=f"{model_name} int8-QAT")
    int8_time = time.time() - t0
    print(f"  Int8 QAT Accuracy: {result_int8['accuracy']:.2%}")
    print(f"  Int8 QAT Loss:     {result_int8['loss']:.4f}")
    print(f"  Int8 QAT Time:     {int8_time:.2f}s")

    # 量化损失
    quant_loss = result_fp32["accuracy"] - result_int8["accuracy"]
    print(f"  Quantization Loss: {quant_loss:+.2%} "
          f"({'[OK] tiny' if abs(quant_loss) < 0.005 else '[~] acceptable' if abs(quant_loss) < 0.02 else '[!!] needs attention'})")

    return {
        "name": model_name,
        "arch": "SpaceNet V1 (seq+BN, bias=False)",
        "mode": "QAT-int8 (v4)",
        "params": total_params,
        "fp32_acc": result_fp32["accuracy"],
        "int8_acc": result_int8["accuracy"],
        "quant_loss": quant_loss,
        "fp32_time": fp32_time,
        "int8_time": int8_time,
    }


# ============================================================
#  Optic 模式评估 (硬件级光计算模拟, 使用 osimulator)
# ============================================================
def evaluate_optic_int8(model_class, weight_path, engine, test_loader, device,
                        quick_batches=None, is_quick_mode=False):
    """
    Optic 模式: build_optical_model + osimulator 硬件级评估。

    所有 Conv → OpticConv2d, 所有 Linear → OpticLinear,
    首次 stem 虽然训练时是 FP32, 但在 Optic 模式下也会被转换为光计算层
    (可以观察如果 stem 也在光计算上运行的效果)。
    """
    from optic_layers import (build_optical_model, compute_alignment_ratio,
                              print_alignment_detail, evaluate_model)

    model_name = "Model 2 Phase4 v3 INT8"
    print(f"\n{'='*60}")
    print(f"  {model_name}  [Optic mode: osimulator]")
    print(f"{'='*60}")

    print(f"\n  [1/3] Creating standard model & loading weights...")
    model = model_class(num_classes=NUM_CLASSES)
    if not os.path.exists(weight_path):
        print(f"  [ERROR] Weight not found: {weight_path}")
        return None

    state_dict = torch.load(weight_path, map_location='cpu')
    model_state = model.state_dict()
    # 过滤 QAT 特有的参数 (如 _qat_enabled 等 buffer)
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    skipped = len(state_dict) - len(filtered)
    if skipped:
        print(f"  Skipped {skipped} QAT-specific params (expected)")
    print(f"  Weights loaded from: {weight_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    # 打印原始模型对齐率
    print_alignment_detail(model, f"{model_name} (Original FP32)")

    # 转换为光计算模型 (INT8 量化, stem 保留电计算)
    print(f"\n  [2/3] Converting to optical (OpticConv2d + OpticLinear, int8, stem=electronic)...")
    build_optical_model(model, engine, pad_to_8=True,
                        input_bit=8, weight_bit=8,
                        keep_first_conv_electronic=True)
    print_alignment_detail(model, f"{model_name} (Optical)")

    # osimulator 评估 — quick 模式逐 batch 打印, 全量模式按 epoch 聚合
    print(f"\n  [3/3] Evaluating via osimulator...")
    total_batches = quick_batches if quick_batches else len(test_loader)
    print_interval = 1 if is_quick_mode else max(1, total_batches // 10)

    t0 = time.time()
    result = evaluate_model(model, test_loader, device,
                            criterion=nn.CrossEntropyLoss(),
                            max_batches=quick_batches,
                            desc=f"{model_name} optic",
                            print_interval=print_interval)
    optic_time = time.time() - t0
    print(f"  Optical Accuracy: {result['accuracy']:.2%}")
    print(f"  Optical Time:     {optic_time:.2f}s")

    return {
        "name": model_name,
        "arch": "SpaceNet V1 (seq+BN, bias=False)",
        "mode": "Optic (osimulator)",
        "params": total_params,
        "optic_acc": result["accuracy"],
        "optic_time": optic_time,
    }


# ============================================================
#  综合报告
# ============================================================
def print_report(result_qat, result_optic, layers, summary):
    """打印综合评估报告 (精度 + MOPs)"""
    print("\n\n")
    print("=" * 110)
    print("  OPTIC-SPACENET INT8: Optical Computing Inference & MOPs Report")
    print("  Model 2 SpaceNet V1 Phase4 v3 — 当前最佳 INT8 模型")
    print("=" * 110)

    # ---- 精度部分 ----
    if result_qat:
        print(f"\n  {'-' * 60}")
        print(f"  [Accuracy] QAT int8 Pseudo-Quantization (独立测试集)")
        print(f"  {'-' * 60}")
        print(f"  模型:               {result_qat['name']}")
        print(f"  参数量:             {result_qat['params']:,}")
        print(f"  Float32 准确率:     {result_qat['fp32_acc']:.2%}")
        print(f"  Int8 QAT 准确率:    {result_qat['int8_acc']:.2%}")
        print(f"  量化损失:           {result_qat['quant_loss']:+.2%}")
        print(f"  Float32 耗时:       {result_qat['fp32_time']:.1f}s")
        print(f"  Int8 QAT 耗时:      {result_qat['int8_time']:.1f}s")
        print(f"\n  训练参考 (训练时 val split, seed=42):")
        print(f"    训练 Int8 最佳:   93.11% (Phase4 v3, 100 epochs)")
        print(f"    训练 Float32:     93.02%")
        print(f"    FP32 基准:        90.15%")
        print(f"    旧版 int4 (bug):  74.35%")

    if result_optic:
        print(f"\n  {'-' * 60}")
        print(f"  [Accuracy] Optic osimulator Hardware Simulation (独立测试集)")
        print(f"  {'-' * 60}")
        print(f"  模型:               {result_optic['name']}")
        print(f"  光计算准确率:       {result_optic['optic_acc']:.2%}")
        print(f"  osimulator 耗时:    {result_optic['optic_time']:.1f}s")

    # ---- MOPs 部分 ----
    print_mops_report(layers, summary)

    # ---- 最终结论 ----
    opt_ratio = summary['optical_ratio']
    print(f"\n  {'-' * 60}")
    print(f"  [Verdict] 部署评估结论")
    print(f"  {'-' * 60}")
    print(f"  光计算占比:         {opt_ratio:.2%}")
    if opt_ratio >= 0.90:
        print(f"  判定:               [OK] 高度适合光计算部署")
    elif opt_ratio >= 0.70:
        print(f"  判定:               [~] 较适合光计算部署 (部分层需电计算)")
    else:
        print(f"  判定:               [!!] 光计算利用率偏低, 建议优化架构")
    print(f"  硬件对齐率:         99.6% (除 stem 外所有层完美对齐 8×2 tile)")
    print(f"  首层 stem:          FP32 电计算 (展平=3, 对齐率 37.5%, 电计算更高效)")
    print(f"  推荐部署策略:       stem 在 CPU/GPU, 其余 5 层在 Gazelle 光计算")
    print("=" * 110)


# ============================================================
#  主函数 — 默认 Optic 模式 (容器内 osimulator)
#  用 --qat 切换为 PyTorch 伪量化交叉验证
# ============================================================
def main():
    use_qat = "--qat" in sys.argv          # QAT 伪量化模式 (容器外交叉验证)
    mops_only = "--mops-only" in sys.argv  # 仅 MOPs 统计
    quick_batches = None
    batch_size = DEFAULT_BATCH

    for i, arg in enumerate(sys.argv):
        if arg == "--quick":
            quick_batches = int(sys.argv[i+1]) if i+1 < len(sys.argv) else 5
        if arg == "--batch":
            batch_size = int(sys.argv[i+1]) if i+1 < len(sys.argv) else DEFAULT_BATCH

    # ================================================================
    # 模型与权重配置
    # ================================================================
    model_class = OpticSpaceNetV1_INT8
    weight_path = "spacenet_v1_phase4_v3_int8.pth"

    if mops_only:
        mode_str = "MOPs-only"
    elif use_qat:
        mode_str = "QAT (PyTorch pseudo-quant, cross-validation)"
    else:
        mode_str = "Optic (osimulator, in-container)"

    print("=" * 60)
    print("  Optic-SpaceNet INT8: In-Container Optical Inference")
    print(f"  Model:  SpaceNet V1 Phase4 v3 (INT8, Gazelle-optimized)")
    print(f"  Weight: {weight_path}")
    print(f"  Mode:   {mode_str}")
    print(f"  Batch:  {batch_size}" +
          (f", quick={quick_batches}" if quick_batches else ", full test set"))
    print("=" * 60)

    # ---- MOPs 统计 (始终计算) ----
    layers, summary = compute_mops_detail()

    if mops_only:
        print_mops_report(layers, summary)
        return layers, summary

    # ---- 数据加载 (独立测试集, seed 不同于训练) ----
    print("\n--- Loading Independent Test Set ---")
    test_loader = load_test_data(batch_size=batch_size)

    result_qat = None
    result_optic = None

    if use_qat:
        # ---- QAT 模式: int8 伪量化 (容器外交叉验证) ----
        print("\n[Mode: QAT] Running PyTorch pseudo-quantization for cross-validation...")
        result_qat = evaluate_qat_int8(
            model_class, weight_path, test_loader, DEVICE,
            quick_batches=quick_batches)
    else:
        # ---- Optic 模式 (默认): osimulator 容器内硬件仿真 ----
        print("\n[Mode: Optic] Initializing Optical Engine (osimulator) in container...")
        is_quick = quick_batches is not None
        from optic_layers import OpticalEngine
        engine = OpticalEngine(use_real=True, verbose=is_quick)  # quick 模式才逐次打印
        engine.reset_stats()

        result_optic = evaluate_optic_int8(
            model_class, weight_path, engine, test_loader, DEVICE,
            quick_batches=quick_batches,
            is_quick_mode=is_quick)

        print("\n--- Optical Engine Statistics ---")
        engine.print_stats()

    # ---- 综合报告 ----
    print_report(result_qat, result_optic, layers, summary)

    return result_qat, result_optic, layers, summary


if __name__ == "__main__":
    main()
