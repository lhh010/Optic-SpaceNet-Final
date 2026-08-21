"""X0 B1-prep 冒烟验证: 4 臂 + 后向兼容回归 (本地 CPU, 不训练)。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import torch
from config import load_config
from models import build_model, compute_macs

CFG = os.path.join(_HERE, "..", "configs")


def check(cfg_file, note=""):
    cfg, _ = load_config(os.path.join(CFG, cfg_file))
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    macs = compute_macs(model)
    x = torch.randn(2, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, cfg["num_classes"]), f"{cfg_file}: bad out {out.shape}"
    print(f"{cfg['name']:>26s}  params={n_params:>8,}  MACs={macs:>10,} "
          f"({macs/1e6:.3f}M)  out={tuple(out.shape)}  {note}")
    return cfg, model, n_params, macs


print("== X0 4 臂 ==")
arms = {}
for f in ["x0_pool3_160.json", "x0_pool4_160.json",
          "x0_blurpool_160.json", "x0_dsconv3_160.json"]:
    cfg, model, p, m = check(f)
    arms[cfg["name"]] = (p, m)

print("\n== 后向兼容回归 (旧 configs 行为不得改变) ==")
base = check("r6_ctrl.json", "(J1 ctrl)")
check("r7_final_head256_160ep.json")
check("r8_rf_s2k3_160.json")
check("r6_pool_avg.json")
check("r6_pool_s1x1.json")
check("r6_pool_patch.json")
check("c3d_J1_v8probe15.json", "(c3d 冠军架构=J1)")

print("\n== QAT v8 prepare 转换检查 (x0_dsconv3, 含新 conv 层) ==")
cfg, _ = load_config(os.path.join(CFG, "x0_dsconv3_160.json"))
model = build_model(cfg)
from qat_v8 import prepare_model_v8
prepare_model_v8(model,
                 layer_sigmas=cfg["layer_noise_sigmas"],
                 layer_col_off=cfg["layer_col_off"],
                 layer_col_gain=cfg["layer_col_gain"],
                 layer_dw_rms=cfg["layer_dw_rms"],
                 stem_fp32=True, head_fp32=True,
                 weight_bits=cfg["weight_bits"],
                 output_noise=True, output_quant=True,
                 weight_noise=False,
                 weight_noise_ratio=cfg["noise_std_ratio"],
                 activation_style="osim")
model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 3, 64, 64))
print(f"QAT v8 转换后前向 OK, out={tuple(out.shape)}")

print("\n== BlurPool buffer 落盘检查 ==")
cfg, _ = load_config(os.path.join(CFG, "x0_blurpool_160.json"))
m = build_model(cfg)
sd = m.state_dict()
blur_keys = [k for k in sd if "kernel" in k]
print(f"BlurPool buffer keys in state_dict: {blur_keys}")
trainable = [n for n, p in m.named_parameters() if p.requires_grad and "kernel" in n]
print(f"BlurPool kernel 可训练参数 (应为空): {trainable}")
