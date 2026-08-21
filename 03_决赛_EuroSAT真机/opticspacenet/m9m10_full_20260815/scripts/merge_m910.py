#!/usr/bin/env python3
"""合并 M9/M10 分段 logits 计算全量精度 (证据复核脚本)。
用法: python3 merge_m910.py <logits_dir> <labels_5400.npy>
"""
import os
import sys
import numpy as np


def report(name, logits, labels_slice):
    pred = np.argmax(logits, axis=1)
    n = len(labels_slice)
    err = int((pred != labels_slice).sum())
    acc = (n - err) / n * 100
    print(f"{name}: {n} 张 acc={acc:.2f}% 错误={err}")
    return acc, err


def main():
    R = sys.argv[1] if len(sys.argv) > 1 else "."
    labels = np.load(sys.argv[2] if len(sys.argv) > 2 else "test_labels_5400.npy")

    # ---- M9 [1200:5400] (本次 11 段, 每段取 400, 末段 200) ----
    parts = [np.load(os.path.join(R, "logits_probe_m9patch__off1200.npy"))[:400]]
    for off in range(1600, 5200, 400):
        parts.append(np.load(os.path.join(R, f"logits_probe_m9ccic__off{off}.npy"))[:400])
    parts.append(np.load(os.path.join(R, "logits_probe_m9ccic__off5200.npy")))
    lg = np.vstack(parts)
    assert lg.shape[0] == 4200
    print("== M9 本次 [1200:5400] ==")
    report("M9 [1200:5400]", lg, labels[1200:5400])

    # ---- M9 [1000:1200] 补段 ----
    lg1 = np.load(os.path.join(R, "logits_probe_m9patch1000__off1000.npy"))[:200]
    report("M9 [1000:1200]", lg1, labels[1000:1200])

    # ---- M10 [0:5400] ----
    offs = list(range(0, 5280, 240)) + [5280]
    parts = [np.load(os.path.join(R, f"logits_probe_m10ccic__off{off}.npy")) for off in offs]
    lg10 = np.vstack(parts)
    assert lg10.shape[0] == 5400
    print("== M10 全量 [0:5400] ==")
    report("M10 [0:5400]", lg10, labels)
    for off in offs:
        lgx = np.load(os.path.join(R, f"logits_probe_m10ccic__off{off}.npy"))
        report(f"M10 off={off}", lgx, labels[off:off + len(lgx)])


if __name__ == "__main__":
    main()
