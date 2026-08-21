# -*- coding: utf-8 -*-
"""hw_ab_analysis.py — c3d/c3e vs c2c 同窗 hw logits 误差重叠分析。
用法: python3 hw_ab_analysis.py
输入 (需先 scp 回 /tmp):
  /tmp/logits_c3d.npy /tmp/logits_c2c_win.npy [/tmp/logits_c3e.npy]
  /tmp/logits_fake_clean_1000_c3d.npy /tmp/logits_fake_clean_1000_c3e.npy
  /tmp/labels_1000.npy
"""
import os
import numpy as np

y = np.load('/tmp/labels_1000.npy')


def acc(p):
    return 100 * (p == y).mean()


def z(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


runs = {}
for tag in ['c3d', 'c2c_win', 'c3e']:
    p = f'/tmp/logits_{tag}.npy'
    if os.path.exists(p):
        runs[tag] = np.load(p)

fake = {}
for tag in ['c3d', 'c3e']:
    p = f'/tmp/logits_fake_clean_1000_{tag}.npy'
    if os.path.exists(p):
        fake[tag] = np.load(p)

for tag, lg in runs.items():
    pred = lg.argmax(1)
    print(f'{tag}: hw acc = {acc(pred):.2f}%  err={(pred != y).sum()}')

tags = list(runs)
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        a, b = runs[tags[i]].argmax(1), runs[tags[j]].argmax(1)
        print(f'agree {tags[i]}~{tags[j]}: {100 * (a == b).mean():.2f}%')

# c3d/c3e 是否修掉了 c2c 的 hw 特有结构化错误
if 'c2c_win' in runs:
    e_c2c = runs['c2c_win'].argmax(1) != y
    for tag in ['c3d', 'c3e']:
        if tag in runs:
            e = runs[tag].argmax(1) != y
            fixed = (e_c2c & ~e).sum()
            new = (~e_c2c & e).sum()
            print(f'{tag} vs c2c_win: fixed {fixed}, new err {new}, '
                  f'net {fixed - new:+d}')
        if tag in fake:
            fe = fake[tag].argmax(1) != y
            he = runs[tag].argmax(1) != y if tag in runs else None
            print(f'{tag} FAKE err {fe.sum()}', end='')
            if he is not None:
                print(f', hw-only err {(he & ~fe).sum()}', end='')
            print()

# logit 级: c3d vs c2c 的结构化差异
if 'c3d' in runs and 'c2c_win' in runs:
    d = (z(runs['c3d']) - z(runs['c2c_win'])).ravel().std()
    print(f'z-logit diff rms c3d vs c2c_win: {d:.3f}')
