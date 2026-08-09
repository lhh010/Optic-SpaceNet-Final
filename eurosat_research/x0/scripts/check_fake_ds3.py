"""check_fake_ds3.py — 本地 FAKE 对拍: run_ds3_gazelle (numpy 部署链路) vs PyTorch 参考。
用法: python3 check_fake_ds3.py <ckpt> <config> <weights_dir> [n]
  - runner FAKE 前向 (int8 权重 + uint8 激活量化, 模拟 QAT 部署)
  - torch FP32 前向 (同 ckpt)
  - 报告: 两者 acc / 相互 logits corr / 逐层 stem 一致性
"""
import os
import sys
import json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from models import build_model  # noqa: E402

os.environ["DS3_FAKE"] = "1"
import run_ds3_gazelle as R  # noqa: E402


def main(ckpt, cfg_path, wdir, n=1000):
    os.environ["DS3_WEIGHTS_DIR"] = wdir
    R.WDIR = wdir
    cfg = json.load(open(cfg_path))
    cfg.setdefault("num_classes", 10)
    model = build_model(cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    ws, meta = R.load_weights()
    images = np.load(os.path.join(wdir, meta["images"]))[:n]
    labels = np.load(os.path.join(wdir, meta["labels"]))[:n]

    # torch FP32 参考
    with torch.no_grad():
        logits_t = model(torch.from_numpy(images)).numpy()
    acc_t = float((logits_t.argmax(1) == labels).mean()) * 100

    # runner FAKE (部署链路)
    logits_r = []
    for s in range(0, len(images), 8):
        logits_r.append(R.forward(images[s:s + 8], ws, meta))
    logits_r = np.vstack(logits_r)
    acc_r = float((logits_r.argmax(1) == labels).mean()) * 100

    # stem 一致性 (电计算段应接近 FP32 精确)
    with torch.no_grad():
        stem_t = model.stem(torch.from_numpy(images[:16])).numpy()
    stem_r = R.stem_forward(images[:16], ws, meta)
    stem_err = float(np.abs(stem_t - stem_r).max())

    corr = float(np.corrcoef(logits_t.ravel(), logits_r.ravel())[0, 1])
    agree = float((logits_t.argmax(1) == logits_r.argmax(1)).mean()) * 100
    print(f"n={len(images)}")
    print(f"torch FP32 acc : {acc_t:.2f}%")
    print(f"runner FAKE acc: {acc_r:.2f}%")
    print(f"logits corr={corr:.5f} pred agree={agree:.2f}%")
    print(f"stem max abs err={stem_err:.2e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 1000)
