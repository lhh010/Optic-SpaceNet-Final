# -*- coding: utf-8 -*-
"""vote_analysis.py — 重复跑批 logits 投票分析。
用法: python3 vote_analysis.py <labels.npy> <logits1.npy> <logits2.npy> [logits3.npy ...]
输出: 各 repeat 精度、两两一致率、累积投票精度 (诊断 run-to-run 方差 + 投票增益)。
"""
import sys
import itertools

import numpy as np


def main():
    labels = np.load(sys.argv[1])
    logits = [np.load(p) for p in sys.argv[2:]]
    n = len(labels)
    for i, lg in enumerate(logits):
        assert lg.shape[0] == n, f"logits {i+1} rows {lg.shape[0]} != labels {n}"
    preds = [lg.argmax(1) for lg in logits]
    accs = [(p == labels).mean() * 100 for p in preds]
    print("individual acc:", [f"{a:.2f}" for a in accs])
    # 两两预测一致率 (独立噪声 → 一致率低; 结构化误差 → 一致率高)
    for i, j in itertools.combinations(range(len(preds)), 2):
        agree = (preds[i] == preds[j]).mean() * 100
        print(f"agree rep{i+1}~rep{j+1}: {agree:.2f}%")
    # 累积投票 (平均 logits)
    for k in range(2, len(logits) + 1):
        voted = sum(logits[:k]).argmax(1)
        print(f"vote 1..{k}: {(voted == labels).mean() * 100:.2f}%")
    # oracle 上界 (任一 repeat 答对)
    oracle = np.zeros(n, dtype=bool)
    for p in preds:
        oracle |= p == labels
    print(f"oracle (any correct): {oracle.mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
