"""
================================================================================
 optic_inference_h3.py — 光电混合模式 H3: 激进 (两个最慢层都搬电计算)

 模式 H3:
   光计算 (int8 + osimulator): stage1, stage3, fc2          (保 >50% 所需的最小集)
   电计算 (fp32):              stem, stage2, fc1            (两个最慢层 stage2 + fc1 都搬走)
   光计算 MOPs 占比:           53.3%   (> 50% ✓ 刚好满足硬约束)

 目的: 最大提速。stage2 (~3-7s) + fc1 (~2s) 是 osim 最慢的两层, 全搬到电计算;
       只留 stage1(大 MOPs, 必须)+ stage3 + fc2 在光计算, 占比 53% 仍 >50%。
       再砍掉 stage3/fc2 就会跌破 50% (stage1 单层 49.9%), 故这是满足约束的最快切分。

 适用模型 (架构相同, 仅权重不同):
   Model 2: weights/spacenet_v1_phase4_v3_int8.pth   (val int8 92.06%)
   Model 3: weights/spacenet_v2_phase4_v3_int8.pth   (val int8 91.83%, osim q500 90.80%)

 用法 (光计算 Docker 容器内):
   python optic_inference_h3.py --quick 100            # M2+M3 各 100 张
   python optic_inference_h3.py --model 2 --quick 100  # 只跑 Model 2
   python optic_inference_h3.py --mops-only            # 只打印 MOPs 占比
================================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _pathsetup  # noqa: E402,F401

import os, sys, time
import torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = "data/EuroSAT_RGB"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED_TRAIN = 42
DEFAULT_BATCH = 1
print(f"Device: {DEVICE}")

# ==================== 本模式配置 ====================
MODE = "H3"
MODE_DESC = "激进 (stage2+fc1 电计算, stage1+stage3+fc2 光计算)"
# 光计算层的 dotted name (classifier.4=fc2)
OPTICAL_DOTTED = {"stage1.0", "stage3.0", "classifier.4"}
WEIGHTS = {
    "2": ("weights/spacenet_v1_phase4_v3_int8.pth", "Model 2 SpaceNet V1 v3 int8"),
    "3": ("weights/spacenet_v2_phase4_v3_int8.pth", "Model 3 SpaceNet V2 KD v3 int8"),
}

# LAYER_SPECS: name, type, C_in, C_out, Kh, Kw, stride, pad, pool, is_optical(本模式)
LAYER_SPECS = [
    ("stem.conv",   "Conv",   3,   8,  1, 1, 1, 0,  None,    False),  # 电 (stem)
    ("stage1.conv", "Conv",   8,  16,  2, 2, 2, 0,  "Max2x2", True),  # 光 (大 MOPs, 必须)
    ("stage2.conv", "Conv",  16,  32,  2, 2, 2, 0,  None,    False),  # 电 (最慢, 搬走)
    ("stage3.conv", "Conv",  32,  16,  1, 1, 1, 0,  None,    True),   # 光
    ("fc1",         "Linear", 1024, 256, 0, 0, 0, 0, None,    False), # 电 (慢, 搬走)
    ("fc2",         "Linear", 256,  10, 0, 0, 0, 0, None,    True),  # 光
]


# ==================== 模型架构 (Model 2/3 共用) ====================
class SpaceNet(nn.Module):
    """Model 2/3 共用架构 (bias=False, BN 保留). 同 optic_inference_int8.py / optic_inference_kd.py。"""
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


# ==================== 混合转换 (不修改 optic_layers) ====================
def build_hybrid_model(model, engine, optical_names, input_bit=8, weight_bit=8, pad_to_8=True):
    """按 dotted name 选择性转光计算; 未列出的 Conv/Linear 保留原生 (电计算 fp32)。
    不调用也不修改 optic_layers.build_optical_model。"""
    from optic_layers import OpticConv2d, OpticLinear
    s = set(optical_names)
    cf = lambda m: OpticConv2d(m, engine, pad_to_8=pad_to_8, input_bit=input_bit, weight_bit=weight_bit)
    lf = lambda m: OpticLinear(m, engine, pad_to_8=pad_to_8, input_bit=input_bit, weight_bit=weight_bit)

    def _walk(module, prefix):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d) and full in s:
                setattr(module, name, cf(child))
            elif isinstance(child, nn.Linear) and full in s:
                setattr(module, name, lf(child))
            else:
                _walk(child, full)
    _walk(model, "")
    return model


# ==================== 数据 (干净 test, eurosat_split) ====================
def load_test_data(batch_size=DEFAULT_BATCH, test_ratio=0.2):
    tf = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full = datasets.ImageFolder(DATA_DIR, transform=tf)
    from eurosat_split import split_indices
    _, _, test_idx = split_indices(len(full), seed=SEED_TRAIN, val_ratio=test_ratio, test_ratio=test_ratio)
    loader = DataLoader(torch.utils.data.Subset(full, test_idx),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Full: {len(full)} | Test: {len(test_idx)} imgs | split=eurosat_split (test∩train=0)")
    return loader


# ==================== MOPs 统计 (按本模式 is_optical) ====================
def _spatial(H, W, kh, kw, s, p):
    return (H + 2 * p - kh) // s + 1, (W + 2 * p - kw) // s + 1

def _pool(H, W, pool):
    return (H // 2, W // 2) if pool == "Max2x2" else (H, W)

def compute_mops_detail():
    H, W = 64, 64
    layers = []
    for name, ltype, ci, co, kh, kw, s, p, pool, is_opt in LAYER_SPECS:
        if ltype == "Conv":
            pl, pdl = ci * kh * kw, ((ci * kh * kw + 7) // 8) * 8
            Ho, Wo = _spatial(H, W, kh, kw, s, p)
            raw = co * Ho * Wo * ci * kh * kw
            opt_m = Ho * Wo * pdl * co if is_opt else 0
            elec_m = raw if not is_opt else 0
            layers.append({"name": name, "type": ltype, "c_in": ci, "c_out": co,
                           "kernel": f"{kh}x{kw}", "spatial_in": f"{H}x{W}",
                           "spatial_out": f"{Ho}x{Wo}", "pool": pool or "None",
                           "patch_len": pl, "padded_len": pdl,
                           "alignment": pl / pdl if pdl > 0 else 1,
                           "raw_mops": raw / 1e6, "optical_mops": opt_m / 1e6,
                           "electronic_mops": elec_m / 1e6,
                           "effective_mops": (opt_m if is_opt else raw) / 1e6,
                           "is_optical": is_opt})
            H, W = _pool(Ho, Wo, pool)
        else:
            pl, pdl = ci, ((ci + 7) // 8) * 8
            raw = ci * co
            opt_m = pdl * co if is_opt else 0
            elec_m = raw if not is_opt else 0
            layers.append({"name": name, "type": ltype, "c_in": ci, "c_out": co, "kernel": "-",
                           "spatial_in": "-", "spatial_out": "-", "pool": "None",
                           "patch_len": pl, "padded_len": pdl,
                           "alignment": pl / pdl if pdl > 0 else 1,
                           "raw_mops": raw / 1e6, "optical_mops": opt_m / 1e6,
                           "electronic_mops": elec_m / 1e6,
                           "effective_mops": (opt_m if is_opt else raw) / 1e6,
                           "is_optical": is_opt})
    tr = sum(l["raw_mops"] for l in layers)
    to = sum(l["optical_mops"] for l in layers)
    te = sum(l["electronic_mops"] for l in layers)
    tef = sum(l["effective_mops"] for l in layers)
    return layers, {"total_raw_mops": tr, "total_optical_mops": to, "total_electronic_mops": te,
                    "total_effective_mops": tef,
                    "optical_ratio": to / tef if tef > 0 else 0,
                    "optical_waste": to - sum(l["raw_mops"] for l in layers if l["is_optical"])}


def print_mops_report(layers, summary):
    print(f"\n{'=' * 100}")
    print(f"  光电混合 MOPs 统计 — 模式 {MODE}: {MODE_DESC}")
    print(f"  Gazelle 硬件: 8x2 tile, 8a8w12o; 光计算层 int8 (osim), 电计算层 fp32")
    print(f"{'=' * 100}")
    print(f"  {'Layer':<13s}{'Type':<7s}{'Cin':>4s}{'Cout':>5s}{'K':>5s}{'In':>8s}{'Out':>8s}"
          f"{'Pool':>7s}{'Patch':>6s}{'Pad':>5s}{'Align':>7s}{'RawMOPs':>9s}{'OptMOPs':>9s}"
          f"{'ElecMOPs':>9s}{'Compute':>13s}")
    print("  " + "-" * 100)
    for l in layers:
        loc = "[Optical]" if l["is_optical"] else "[Electronic]"
        print(f"  {l['name']:<13s}{l['type']:<7s}{l['c_in']:>4d}{l['c_out']:>5d}{l['kernel']:>5s}"
              f"{l['spatial_in']:>8s}{l['spatial_out']:>8s}{l['pool']:>7s}{l['patch_len']:>6d}"
              f"{l['padded_len']:>5d}{l['alignment']:>6.1%}{l['raw_mops']:>8.4f}M{l['optical_mops']:>8.4f}M"
              f"{l['electronic_mops']:>8.4f}M{loc:<13s}")
    print("  " + "-" * 100)
    ok = summary['optical_ratio'] > 0.50
    print(f"  [MOPs] 光计算占比: {summary['optical_ratio']:.2%}  "
          f"(总有效 {summary['total_effective_mops']:.4f}M)  "
          f"→ {'[OK] >50% 满足硬约束' if ok else '[!!] <50% 不满足!'}")
    print(f"{'=' * 100}")


# ==================== 评估 ====================
def run_model(model_key, engine, test_loader, quick_batches=None, is_quick=False):
    from optic_layers import print_alignment_detail, evaluate_model
    weight, label = WEIGHTS[model_key]
    print(f"\n{'=' * 72}\n  {label}  [Hybrid {MODE}]\n  {MODE_DESC}\n{'=' * 72}")
    if not os.path.exists(weight):
        print(f"  [ERROR] weight not found: {weight}")
        return None
    model = SpaceNet(NUM_CLASSES)
    sd = torch.load(weight, map_location='cpu')
    ms = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in ms and ms[k].shape == v.shape}, strict=False)
    print(f"  Weights: {weight}  |  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Optical layers (dotted): {sorted(OPTICAL_DOTTED)}")

    build_hybrid_model(model, engine, OPTICAL_DOTTED, input_bit=8, weight_bit=8, pad_to_8=True)
    print_alignment_detail(model, f"{label} (Hybrid {MODE})")

    total = quick_batches or len(test_loader)
    pi = 1 if is_quick else max(1, total // 10)
    print(f"\n  Evaluating via osimulator...")
    t0 = time.time()
    r = evaluate_model(model, test_loader, DEVICE,
                       criterion=nn.CrossEntropyLoss(), max_batches=quick_batches,
                       desc=f"{label} {MODE}", print_interval=pi)
    t = time.time() - t0
    print(f"  Hybrid {MODE} Accuracy: {r['accuracy']:.2%}  Time: {t:.1f}s")
    return {"label": label, "acc": r["accuracy"], "time": t}


def main():
    mops_only = "--mops-only" in sys.argv
    quick = None
    batch = DEFAULT_BATCH
    model_sel = "both"
    for i, a in enumerate(sys.argv):
        if a == "--quick": quick = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 5
        if a == "--batch": batch = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else DEFAULT_BATCH
        if a == "--model": model_sel = sys.argv[i + 1] if i + 1 < len(sys.argv) else "both"
    is_quick = quick is not None

    print(f"{'=' * 72}\n  光电混合推理 — 模式 {MODE}: {MODE_DESC}\n  Batch: {batch}"
          f"{f'  quick={quick}' if quick else '  full test'}  Model: {model_sel}\n{'=' * 72}")

    layers, summary = compute_mops_detail()
    if mops_only:
        print_mops_report(layers, summary)
        return

    print("\n--- Loading Test Set ---")
    test_loader = load_test_data(batch)
    keys = ["2", "3"] if model_sel == "both" else ([model_sel] if model_sel in WEIGHTS else ["2", "3"])

    from optic_layers import OpticalEngine
    engine = OpticalEngine(use_real=True, verbose=is_quick)
    results = []
    for k in keys:
        engine.reset_stats()
        r = run_model(k, engine, test_loader, quick, is_quick)
        if r:
            print(f"\n--- {r['label']} Engine Stats ---")
            engine.print_stats()
            results.append(r)

    print(f"\n{'=' * 100}\n  混合模式 {MODE} 汇总 — {MODE_DESC}\n{'=' * 100}")
    n = quick or len(test_loader)
    print(f"  {'Model':<34s}{'Acc':>8s}{'Time':>9s}{'per-img':>9s}")
    for r in results:
        print(f"  {r['label']:<34s}{r['acc']:>8.2%}{r['time']:>8.1f}s{r['time'] / n:>8.2f}s")
    print_mops_report(layers, summary)


if __name__ == "__main__":
    main()
