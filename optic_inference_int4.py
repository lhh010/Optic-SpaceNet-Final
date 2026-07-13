"""
================================================================================
 optic_inference_int4.py — Model 2 v2 INT4 光计算容器内推理 + MOPs 统计

 模型: Model 2 SpaceNet V1 Phase4 v2 (INT4)
   训练脚本:  model2_spacenet_v1_phase4_v2.py
   权重文件:  spacenet_v1_phase4_v2_ste.pth
   训练精度:  91.06% int4 (QAT eval), 94.57% (独立测试集 QAT)
   QAT 模块:  optic_qat_v3.py (int4 权重 + int8 激活 + STE 噪声)
   硬件配置:  全 Conv+Linear int4 QAT, 无首层 FP32

 ★ 已知限制: osimulator 光学推理精度 ~88%, 低于 QAT 精度 (~94%).
   根因: QAT 训练 (int4 权重, per-channel 输入量化) 与 osimulator 推理路径
   (int8 权重, per-tensor 输入量化) 存在三处量化参数不对齐:
     1. 权重 int4→int8 重量化: 量化网格改变 (scale=max/7 → scale=max/127)
     2. 激活 per-channel→per-tensor: im2col 后通道维度被展平
     3. stem 训练时 QAT / 推理时 FP32 电子: BN 统计量不匹配
   详见 EXPERIMENTS.md §16.

 用法 (在光计算 Docker 容器内):
   python optic_inference_int4.py                        # 默认 Optic 模式全量 (~88%)
   python optic_inference_int4.py --quick 50             # 快速测试
   python optic_inference_int4.py --qat                  # QAT 伪量化交叉验证 (~94%)
   python optic_inference_int4.py --mops-only            # 仅打印 MOPs 统计
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys, time, numpy as np

DATA_DIR = "data/EuroSAT_RGB"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED_TRAIN = 42
DEFAULT_BATCH = 1

print(f"Device: {DEVICE}")


# ============================================================
class OpticSpaceNetV1(nn.Module):
    """Model 2/3 共用架构: 4 Conv + 2 Linear, bias=False"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes, bias=False))

    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.classifier(x)
        return x


# ============================================================
def load_test_data(batch_size=DEFAULT_BATCH, test_ratio=0.2):
    """独立测试集: 与训练 val 零重叠"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    from eurosat_split import split_indices
    # 单一数据源: test 段与训练 train/val 严格互斥 (Bug #11)
    _, _, test_indices = split_indices(len(full_dataset), seed=SEED_TRAIN,
                                       val_ratio=test_ratio, test_ratio=test_ratio)
    test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_indices),
                             batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Full: {len(full_dataset)} | Test: {len(test_indices)} imgs | split=eurosat_split (test∩train=0)")
    return test_loader


@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None, max_batches=None,
             desc="Eval", print_interval=None):
    model.eval(); model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    n = min(len(dataloader), max_batches or len(dataloader))
    if print_interval is None: print_interval = max(1, n // 10)
    for i, (im, lb) in enumerate(dataloader):
        if max_batches and i >= max_batches: break
        out = model(im.to(device))
        if criterion: total_loss += criterion(out, lb.to(device)).item() * im.size(0)
        correct += (out.argmax(1) == lb.to(device)).sum().item()
        total += im.size(0)
    acc = correct / total if total > 0 else 0
    print(f"  [{desc}] {n} batches — acc={acc:.2%}", flush=True)
    return {"accuracy": acc, "loss": total_loss / total if criterion else 0, "total": total, "correct": correct}


# ============================================================
# MOPs — 同 INT8 模型
# ============================================================
LAYER_SPECS = [
    ("stem.conv",   "Conv",   3,   8,  1, 1, 1, 0,  None,    False),
    ("stage1.conv", "Conv",   8,  16,  2, 2, 2, 0,  "Max2x2", True),
    ("stage2.conv", "Conv",  16,  32,  2, 2, 2, 0,  None,    True),
    ("stage3.conv", "Conv",  32,  16,  1, 1, 1, 0,  None,    True),
    ("fc1",         "Linear", 1024, 256, 0, 0, 0, 0, None,    True),
    ("fc2",         "Linear", 256,  10,  0, 0, 0, 0, None,    True),
]

def _spatial(H, W, Kh, Kw, s, p):
    return (H + 2*p - Kh)//s + 1, (W + 2*p - Kw)//s + 1

def _pool(H, W, pool):
    return (H//2, W//2) if pool == "Max2x2" else (H, W)

def compute_mops_detail():
    H, W = 64, 64; layers = []
    for name, ltype, ci, co, kh, kw, s, p, pool, is_opt in LAYER_SPECS:
        if ltype == "Conv":
            pl, pdl = ci*kh*kw, ((ci*kh*kw+7)//8)*8
            Ho, Wo = _spatial(H, W, kh, kw, s, p)
            raw = co * Ho * Wo * ci * kh * kw
            opt_m = Ho * Wo * pdl * co if is_opt else 0
            elec_m = raw if not is_opt else 0
            layers.append({"name": name, "type": ltype, "c_in": ci, "c_out": co,
                           "kernel": f"{kh}x{kw}", "spatial_in": f"{H}x{W}",
                           "spatial_out": f"{Ho}x{Wo}", "pool": pool or "None",
                           "patch_len": pl, "padded_len": pdl,
                           "alignment": pl/pdl if pdl>0 else 1,
                           "raw_mops": raw/1e6, "optical_mops": opt_m/1e6,
                           "electronic_mops": elec_m/1e6,
                           "effective_mops": (opt_m if is_opt else raw)/1e6,
                           "is_optical": is_opt})
            H, W = _pool(Ho, Wo, pool)
        else:
            pl, pdl = ci, ((ci+7)//8)*8
            raw = ci * co
            opt_m = pdl * co if is_opt else 0
            elec_m = raw if not is_opt else 0
            layers.append({"name": name, "type": ltype, "c_in": ci, "c_out": co,
                           "kernel": "-", "spatial_in": "-", "spatial_out": "-",
                           "pool": "None", "patch_len": pl, "padded_len": pdl,
                           "alignment": pl/pdl if pdl>0 else 1,
                           "raw_mops": raw/1e6, "optical_mops": opt_m/1e6,
                           "electronic_mops": elec_m/1e6,
                           "effective_mops": (opt_m if is_opt else raw)/1e6,
                           "is_optical": is_opt})
    total_raw = sum(l["raw_mops"] for l in layers)
    total_opt = sum(l["optical_mops"] for l in layers)
    total_elec = sum(l["electronic_mops"] for l in layers)
    total_eff = sum(l["effective_mops"] for l in layers)
    return layers, {"total_raw_mops": total_raw, "total_optical_mops": total_opt,
                    "total_electronic_mops": total_elec,
                    "total_effective_mops": total_eff,
                    "optical_ratio": total_opt/total_eff if total_eff>0 else 0,
                    "optical_waste": total_opt - sum(l["raw_mops"] for l in layers if l["is_optical"])}

def print_mops_report(layers, summary):
    print(f"\n{'='*110}")
    print(f"  INT4 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v2")
    print(f"  Gazelle 硬件: 8x2 tile, act=int8, weight=int4, stem 电计算")
    print(f"{'='*110}")
    print(f"\n  {'Layer':<16s} {'Type':<6s} {'C_in':>5s} {'C_out':>5s} "
          f"{'Kernel':>6s} {'Input':>10s} {'ConvOut':>10s} {'Pool':>6s} "
          f"{'Patch':>6s} {'Padded':>6s} {'Align':>7s} "
          f"{'RawMOPs':>10s} {'OptMOPs':>10s} {'ElecMOPs':>10s} {'Compute':>12s}")
    print("  " + "-" * 120)
    for l in layers:
        loc = "[Optical]" if l["is_optical"] else "[Electronic]"
        print(f"  {l['name']:<16s} {l['type']:<6s} {l['c_in']:>5d} {l['c_out']:>5d} "
              f"{l['kernel']:>6s} {l['spatial_in']:>10s} {l['spatial_out']:>10s} {l['pool']:>6s} "
              f"{l['patch_len']:>6d} {l['padded_len']:>6d} {l['alignment']:>6.1%} "
              f"{l['raw_mops']:>9.4f}M {l['optical_mops']:>9.4f}M {l['electronic_mops']:>9.4f}M {loc:<12s}")
    print("  " + "-" * 120)
    print(f"  {'Total':<16s} {'':<6s} {'':>5s} {'':>5s} {'':>6s} {'':>10s} {'':>10s} {'':>6s} "
          f"{'':>6s} {'':>6s} {'':>7s} "
          f"{summary['total_raw_mops']:>9.4f}M {summary['total_optical_mops']:>9.4f}M "
          f"{summary['total_electronic_mops']:>9.4f}M")
    print(f"\n  {'-'*60}")
    print(f"  [MOPs] 光计算占比汇总")
    print(f"  {'-'*60}")
    print(f"  光计算占比:          {summary['optical_ratio']:.2%}")
    print(f"  总 MOPs:             {summary['total_raw_mops']:.4f} M")
    print(f"  [Note] stem 展平=3 对齐率仅 37.5%, 保留电计算 (FP32)")
    print(f"  [Note] 预期光学精度 ~88%, QAT 参考 ~94% (test) / 91.06% (val)")
    print(f"  [Note] 6% 损失来源: int4→int8 重量化 + per-channel→per-tensor + stem 不一致")
    print(f"  [Note] 详见 EXPERIMENTS.md §16")
    print(f"{'='*110}")


# ============================================================
def evaluate_qat(model_class, weight_path, test_loader, device, quick_batches=None):
    print(f"\n{'='*60}\n  Model 2 Phase4 v2 INT4  [QAT mode: int4]\n{'='*60}")
    print(f"\n  [1/3] Creating model...")
    model = model_class(num_classes=NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n  [2/3] Converting to QAT v3 (int4 weight, int8 act)...")
    from optic_qat_v3 import prepare_model_v3, enable_qat, disable_qat
    prepare_model_v3(model, mode="ste", weight_bits=4, act_bits=8,
                     noise=False, quantize_linear=True, preserve_bn=True)
    print(f"\n  [3/3] Loading weights: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)

    disable_qat(model)
    t0 = time.time()
    r_fp32 = evaluate(model, test_loader, device, nn.CrossEntropyLoss(), quick_batches, "fp32")
    print(f"  Float32: {r_fp32['accuracy']:.2%} ({time.time()-t0:.1f}s)")

    enable_qat(model)
    t0 = time.time()
    r_int4 = evaluate(model, test_loader, device, nn.CrossEntropyLoss(), quick_batches, "int4-QAT")
    print(f"  Int4 QAT: {r_int4['accuracy']:.2%} ({time.time()-t0:.1f}s)")
    print(f"  Quant Loss: {r_fp32['accuracy']-r_int4['accuracy']:+.2%}")

    return {"name": "Model 2 v2 INT4", "fp32_acc": r_fp32["accuracy"],
            "int_acc": r_int4["accuracy"], "quant_loss": r_fp32["accuracy"]-r_int4["accuracy"]}


def evaluate_optic(model_class, weight_path, engine, test_loader, device,
                   quick_batches=None, is_quick_mode=False):
    from optic_layers import build_optical_model, print_alignment_detail, evaluate_model
    print(f"\n{'='*60}\n  Model 2 Phase4 v2 INT4  [Optic mode: osimulator]\n{'='*60}")
    print(f"\n  [1/3] Loading weights...")
    model = model_class(num_classes=NUM_CLASSES)
    sd = torch.load(weight_path, map_location='cpu')
    ms = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in ms and ms[k].shape == v.shape}, strict=False)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print_alignment_detail(model, "Original FP32")
    print(f"\n  [2/3] Converting to optical (int8 act + int8 weight, stem=electronic)...")
    print(f"  [Note] osimulator 原生 8a8w. QAT int4→optical int8 重量化非无损:")
    print(f"         int4 grid (scale=max/7, 16级) → int8 grid (scale=max/127, 256级)")
    print(f"         叠加 per-channel→per-tensor 输入量化差异, 预期光学精度 ~88%")
    print(f"         (QAT 参考: ~94% on test set, 91.06% on val set)")
    build_optical_model(model, engine, pad_to_8=True, input_bit=8, weight_bit=8,
                        keep_first_conv_electronic=True)
    print_alignment_detail(model, "Optical")

    total_batches = quick_batches or len(test_loader)
    print_interval = 1 if is_quick_mode else max(1, total_batches // 10)
    print(f"\n  [3/3] Evaluating via osimulator (预期 ~88%, 见 EXPERIMENTS.md §16)...")
    t0 = time.time()
    result = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss(),
                            quick_batches, "optic", print_interval)
    t = time.time() - t0
    print(f"  Optical Accuracy: {result['accuracy']:.2%}  Time: {t:.1f}s")
    if result['accuracy'] < 0.85:
        print(f"  [WARN] 精度 < 85%! 检查权重文件是否匹配 (应为 spacenet_v1_phase4_v2_ste.pth)")
    return {"name": "Model 2 v2 INT4", "optic_acc": result["accuracy"], "optic_time": t}


def main():
    use_qat = "--qat" in sys.argv
    mops_only = "--mops-only" in sys.argv
    quick_batches = None; batch_size = DEFAULT_BATCH
    for i, a in enumerate(sys.argv):
        if a == "--quick": quick_batches = int(sys.argv[i+1]) if i+1 < len(sys.argv) else 5
        if a == "--batch": batch_size = int(sys.argv[i+1]) if i+1 < len(sys.argv) else DEFAULT_BATCH

    weight_path = "spacenet_v1_phase4_v2_ste.pth"
    print(f"{'='*60}\n  Optic-SpaceNet INT4: In-Container Optical Inference")
    print(f"  Model 2 Phase4 v2 (int4, 91.06%)  |  Weight: {weight_path}")
    print(f"  Mode: {'MOPs-only' if mops_only else 'QAT' if use_qat else 'Optic (default)'}")
    print(f"{'='*60}")

    layers, summary = compute_mops_detail()
    if mops_only: print_mops_report(layers, summary); return

    print("\n--- Loading Test Set ---")
    test_loader = load_test_data(batch_size=batch_size)

    if use_qat:
        r = evaluate_qat(OpticSpaceNetV1, weight_path, test_loader, DEVICE, quick_batches)
        r_optic = None
    else:
        is_quick = quick_batches is not None
        from optic_layers import OpticalEngine
        engine = OpticalEngine(use_real=True, verbose=is_quick)
        engine.reset_stats()
        r_optic = evaluate_optic(OpticSpaceNetV1, weight_path, engine, test_loader, DEVICE,
                                 quick_batches, is_quick)
        print("\n--- Optical Engine Statistics ---"); engine.print_stats()
        r = None

    # Report
    print(f"\n{'='*100}")
    print(f"  Model 2 Phase4 v2 INT4 — Container Verification Report")
    print(f"{'='*100}")
    if r:
        print(f"  QAT float32: {r['fp32_acc']:.2%}  |  QAT int4: {r['int_acc']:.2%}  |  Quant Loss: {r['quant_loss']:+.2%}")
    if r_optic:
        print(f"  Optic osimulator: {r_optic['optic_acc']:.2%}  |  Time: {r_optic['optic_time']:.0f}s")
    print(f"  ---")
    print(f"  QAT 参考: 91.06% (训练 val) / ~94% (test set)")
    print(f"  Optic 预期: ~88% (int4→int8 重量化 + per-channel→per-tensor 导致 ~6% 损失)")
    print(f"  ---")
    print(f"  根因 (详见 EXPERIMENTS.md §16):")
    print(f"    1. 权重 int4→int8: 量化网格 scale=max/7 → scale=max/127")
    print(f"    2. 激活 per-channel→per-tensor: im2col 后通道维度被展平")
    print(f"    3. stem QAT→FP32 电子: BN 统计量不匹配")
    print_mops_report(layers, summary)
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
