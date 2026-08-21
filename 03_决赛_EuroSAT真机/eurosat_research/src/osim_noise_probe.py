"""逐层 osimulator 噪声 vs 信号诊断 (J1 架构)。用法: python osim_noise_probe.py <weight>"""
import sys, os, glob
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "..", "src", "core"))
from optic_layers import OpticalEngine, quantize_to_int
from osim_eval_j1 import build_j1
from data import load_eurosat

def main():
    weight = sys.argv[1]
    m = build_j1(weight, torch.device("cpu"))
    m.eval()
    engine = OpticalEngine(use_real=True, verbose=False)
    _, _, te = load_eurosat("data/EuroSAT_RGB", batch_size=8, aug="none", num_workers=2)
    xs, _ = next(iter(te))

    x = xs
    # stem: 3x3 s2 → pool
    x0 = F.relu(m.stem[0](x)); x = F.max_pool2d(x0, 2)
    x1 = F.relu(m.stage1[0](x)); x = F.max_pool2d(x1, 2)
    x2a = F.relu(m.stage2[0](x)); x = F.max_pool2d(x2a, 2)
    x2b = F.relu(m.stage2[3](x))
    x3a = F.relu(m.stage3[0](x2b))
    x3b = F.relu(m.stage3[3](x3a))

    layers = [
        ("stage1.0", x1, m.stage1[0].weight.detach(), 16),
        ("stage2.0", x2a, m.stage2[0].weight.detach(), 32),
        ("stage2.3", x2b, m.stage2[3].weight.detach(), 64),
        ("stage3.0", x3a, m.stage3[0].weight.detach(), 64),
        ("stage3.3", x3b, m.stage3[3].weight.detach(), 128),
    ]

    for name, xt, w, C in layers:
        if w.dim() == 4:
            amax = w.abs().amax(dim=(1, 2, 3)).clamp(min=1e-8)
            w_int = (w / amax.view(-1, 1, 1, 1) * 127).round().clamp(-127, 127)
            w_int = w_int.reshape(C, -1).t().to(torch.int32)
        else:
            amax = w.abs().amax(dim=0).clamp(min=1e-8)
            w_int = (w / amax.view(-1, 1) * 127).round().clamp(-127, 127).t().to(torch.int32)
        B, Cc, H, W = xt.shape
        x_im2col = xt.permute(0, 2, 3, 1).reshape(B * H * W, Cc)
        x_int, x_scale, x_zp = quantize_to_int(x_im2col, 8, signed=False)
        x_int = x_int.reshape(1, B * H * W, Cc)
        inp = x_int.numpy().astype(np.int32)
        wt = np.tile(w_int.numpy()[None, :, :], (1, 1, 1))
        r1 = engine._real_model(inp, wt, inputType="uint8")
        r2 = engine._real_model(inp, wt, inputType="uint8")
        r1 = r1.float() if isinstance(r1, torch.Tensor) else torch.from_numpy(r1).float()
        r2 = r2.float() if isinstance(r2, torch.Tensor) else torch.from_numpy(r2).float()
        diff = (r1 - r2).float()
        col_sum = w_int.float().sum(dim=0)
        w_scale = amax / 127.0
        # 反量化 (浮点域)
        deq = lambda r: x_scale * w_scale.view(1, 1, -1) * r + x_zp * w_scale.view(1, 1, -1) * col_sum.view(1, 1, -1)
        sig = deq(r1)
        noise = deq(r1) - deq(r2)
        print(f"{name}: act_absmean={xt.abs().mean():.4f} qspan={x_int.max() - x_int.min():.0f} "
              f"noise_std={noise.std():.4f} sig_std={sig.std():.4f} n/s_ratio={noise.std() / (sig.std() + 1e-8):.4f}")

if __name__ == "__main__":
    main()
