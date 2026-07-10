"""
================================================================================
 optic_inference_mixed_model1.py — Model 1 Mixed 光计算容器内推理 + MOPs 统计

 模型: Model 1 Baseline VGG Mixed (Conv=int4 光计算, Linear=fp32 电计算)
   训练脚本:  model1_baseline_mixed.py
   权重文件:  baseline_vgg_mixed_ste.pth
   训练精度:  98.26% int4 (Mixed STE)
   QAT 模块:  optic_qat_v3.py (Conv=int4, Linear=fp32, bias: Conv=False, Linear=True)
   架构:     Flat VGG (6 Conv + 2 Linear), ~2.39M 参数

 用法 (在光计算 Docker 容器内):
   python optic_inference_mixed_model1.py               # 默认 Optic 模式全量
   python optic_inference_mixed_model1.py --quick 50    # 快速测试
   python optic_inference_mixed_model1.py --qat         # QAT 交叉验证
   python optic_inference_mixed_model1.py --mops-only   # 仅打印 MOPs 统计
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
class BaselineVGG(nn.Module):
    """Model 1 Mixed: Flat VGG, Conv=bias=False (int4), Linear=bias=True (fp32)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=True)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x))); x = self.pool1(x)
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x))); x = self.pool2(x)
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x))); x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.bn_fc(self.fc1(x))); x = self.dropout(x)
        return self.fc2(x)


# ============================================================
def load_test_data(batch_size=DEFAULT_BATCH, test_ratio=0.2):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    n = len(full_dataset); test_size = int(n * test_ratio)
    indices = list(range(n))
    np.random.RandomState(SEED_TRAIN).shuffle(indices)
    test_indices = indices[test_size:test_size * 2]
    assert len(set(indices[:test_size]) & set(test_indices)) == 0
    test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_indices),
                             batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Full: {n} | Test: {len(test_indices)} imgs | Test/Val overlap: 0")
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
# MOPs — Model 1 Baseline VGG (Flat, 6 Conv + 2 Linear)
# Conv=int4 optical, Linear=fp32 electronic (Mixed strategy)
# ============================================================
LAYER_SPECS_M1 = [
    # name        type     C_in  C_out Kh Kw stride pad  pool     optical?
    ("conv1_1",    "Conv",  3,    32,  3, 3, 1,    1,   None,    True),
    ("conv1_2",    "Conv",  32,   32,  3, 3, 1,    1,   None,    True),
    # pool1 after conv1_2
    ("conv2_1",    "Conv",  32,   64,  3, 3, 1,    1,   None,    True),
    ("conv2_2",    "Conv",  64,   64,  3, 3, 1,    1,   None,    True),
    # pool2 after conv2_2
    ("conv3_1",    "Conv",  64,  128,  3, 3, 1,    1,   None,    True),
    ("conv3_2",    "Conv", 128,  128,  3, 3, 1,    1,   None,    True),
    # pool3 after conv3_2
    ("fc1",        "Linear", 8192, 256, 0, 0, 0, 0, None,    False),   # fp32 electronic
    ("fc2",        "Linear", 256,  10,  0, 0, 0, 0, None,    False),   # fp32 electronic
]

# Pool positions: after conv1_2, conv2_2, conv3_2
POOL_AFTER = {"conv1_2": "Max2x2", "conv2_2": "Max2x2", "conv3_2": "Max2x2"}

def _spatial(H, W, Kh, Kw, s, p):
    return (H + 2*p - Kh)//s + 1, (W + 2*p - Kw)//s + 1

def compute_mops_detail():
    H, W = 64, 64; layers = []
    for name, ltype, ci, co, kh, kw, s, p, _pool_type, is_opt in LAYER_SPECS_M1:
        pool = POOL_AFTER.get(name, None)
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
            H, W = Ho, Wo
            if pool == "Max2x2": H, W = H//2, W//2
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
    print(f"  Mixed 模型光计算 MOPs 统计 — Model 1 Baseline VGG (Conv=int4 光, Linear=fp32 电)")
    print(f"  Gazelle 硬件: 8x2 tile, 4w8a for Conv, Linear 保留电计算")
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
    print(f"\n  [MOPs] 光计算占比: {summary['optical_ratio']:.2%}  |  总 MOPs: {summary['total_raw_mops']:.2f} M")
    if summary['optical_waste'] > 0:
        print(f"  补零浪费: {summary['optical_waste']:.4f} M (conv1_1 展平=27→32)")
    else:
        print(f"  补零浪费: 0 (所有光计算层完美对齐 8 的倍数)")
    print(f"  [Note] Mixed 策略: 6 Conv 在 Gazelle 光计算 (int4), 2 Linear 在 CPU/GPU 电计算 (fp32)")
    print(f"{'='*110}")


# ============================================================
def evaluate_qat(model_class, weight_path, test_loader, device, quick_batches=None):
    print(f"\n{'='*60}\n  Model 1 Mixed  [QAT mode: Conv=int4, Linear=fp32]\n{'='*60}")
    print(f"\n  [1/3] Creating model...")
    model = model_class(num_classes=NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n  [2/3] Converting to Mixed QAT (Conv=int4, Linear=fp32)...")
    from optic_qat_v3 import prepare_model_v3, enable_qat, disable_qat
    prepare_model_v3(model, mode="ste", weight_bits=4, act_bits=8,
                     noise=False, quantize_linear=False, preserve_bn=True)
    print(f"\n  [3/3] Loading weights: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)

    disable_qat(model)
    t0 = time.time()
    r_fp32 = evaluate(model, test_loader, device, nn.CrossEntropyLoss(), quick_batches, "fp32")
    print(f"  Float32: {r_fp32['accuracy']:.2%} ({time.time()-t0:.1f}s)")

    enable_qat(model)
    t0 = time.time()
    r_int4 = evaluate(model, test_loader, device, nn.CrossEntropyLoss(), quick_batches, "int4-mixed")
    print(f"  Int4 Mixed: {r_int4['accuracy']:.2%} ({time.time()-t0:.1f}s)")
    print(f"  Quant Loss: {r_fp32['accuracy']-r_int4['accuracy']:+.2%}")

    return {"name": "Model 1 Mixed", "fp32_acc": r_fp32["accuracy"],
            "int_acc": r_int4["accuracy"], "quant_loss": r_fp32["accuracy"]-r_int4["accuracy"]}


def evaluate_optic(model_class, weight_path, engine, test_loader, device,
                   quick_batches=None, is_quick_mode=False):
    from optic_layers import build_optical_model, print_alignment_detail, evaluate_model
    print(f"\n{'='*60}\n  Model 1 Mixed  [Optic mode: osimulator]\n{'='*60}")
    print(f"\n  [1/3] Loading weights...")
    model = model_class(num_classes=NUM_CLASSES)
    sd = torch.load(weight_path, map_location='cpu')
    # Mixed model has Linear with bias=True, different key names
    ms = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  Skipped {len(sd)-len(filtered)} QAT params | Params: {sum(p.numel() for p in model.parameters()):,}")

    print_alignment_detail(model, "Original FP32")
    print(f"\n  [2/3] Converting to optical (Conv=int4, Linear=electronic)...")
    print(f"  [Note] osimulator uses native 8a8w — QAT int4 Conv weights quantized to int8 (lossless)")
    build_optical_model(model, engine, pad_to_8=True, input_bit=8, weight_bit=8,
                        keep_first_conv_electronic=False, convert_linear=False)
    print_alignment_detail(model, "Optical (Conv only)")

    total_batches = quick_batches or len(test_loader)
    print_interval = 1 if is_quick_mode else max(1, total_batches // 10)
    print(f"\n  [3/3] Evaluating via osimulator...")
    t0 = time.time()
    result = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss(),
                            quick_batches, "optic", print_interval)
    t = time.time() - t0
    print(f"  Optical Accuracy: {result['accuracy']:.2%}  Time: {t:.1f}s")
    return {"name": "Model 1 Mixed", "optic_acc": result["accuracy"], "optic_time": t}


def main():
    use_qat = "--qat" in sys.argv
    mops_only = "--mops-only" in sys.argv
    quick_batches = None; batch_size = DEFAULT_BATCH
    for i, a in enumerate(sys.argv):
        if a == "--quick": quick_batches = int(sys.argv[i+1]) if i+1 < len(sys.argv) else 5
        if a == "--batch": batch_size = int(sys.argv[i+1]) if i+1 < len(sys.argv) else DEFAULT_BATCH

    weight_path = "baseline_vgg_mixed_ste.pth"
    print(f"{'='*60}\n  Optic-SpaceNet Mixed: In-Container Optical Inference")
    print(f"  Model 1 Baseline VGG Mixed (int4 Conv + fp32 Linear, 98.26%)")
    print(f"  Weight: {weight_path}")
    print(f"  Mode: {'MOPs-only' if mops_only else 'QAT' if use_qat else 'Optic (default)'}")
    if not mops_only and not use_qat:
        if quick_batches:
            est = quick_batches * 150 / 60
            print(f"  WARNING: Model 1 is ~150x heavier than Model 2 (156.6M vs 1.05M MACs/img)")
            print(f"  Estimated time for {quick_batches} images: ~{est:.0f} min")
            print(f"  Full 5400 images would take ~9 days — use --quick for sanity checks only")
        else:
            print(f"  WARNING: Full 5400 images would take ~9 days!")
            print(f"  Strongly suggest: python optic_inference_mixed_model1.py --quick 20")
    print(f"{'='*60}")

    layers, summary = compute_mops_detail()
    if mops_only: print_mops_report(layers, summary); return

    print("\n--- Loading Test Set ---")
    test_loader = load_test_data(batch_size=batch_size)

    if use_qat:
        r = evaluate_qat(BaselineVGG, weight_path, test_loader, DEVICE, quick_batches)
        r_optic = None
    else:
        is_quick = quick_batches is not None
        from optic_layers import OpticalEngine
        engine = OpticalEngine(use_real=True, verbose=is_quick)
        engine.reset_stats()
        r_optic = evaluate_optic(BaselineVGG, weight_path, engine, test_loader, DEVICE,
                                 quick_batches, is_quick)
        print("\n--- Optical Engine Statistics ---"); engine.print_stats()
        r = None

    print(f"\n{'='*100}")
    print(f"  Model 1 Mixed (VGG) — Container Verification Report")
    print(f"{'='*100}")
    if r:
        print(f"  QAT float32: {r['fp32_acc']:.2%}  |  QAT int4: {r['int_acc']:.2%}  |  Quant Loss: {r['quant_loss']:+.2%}")
    if r_optic:
        print(f"  Optic osimulator: {r_optic['optic_acc']:.2%}  |  Time: {r_optic['optic_time']:.0f}s")
    print(f"  Training ref: 98.26% int4 Mixed")
    print_mops_report(layers, summary)
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
