# -*- coding: utf-8 -*-
"""光计算占比甜甜圈 + 逐层MOPs堆叠条 — 存 docs/figures/optical_ratio.{png,pdf}
从仓库根目录运行: python src/scripts/plot_optical_ratio.py"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

layers = ['stem\n(电,1×1)', 'stage1\n(2×2)', 'stage2\n(2×2)', 'stage3\n(1×1)', 'fc1\n(Linear)', 'fc2\n(Linear)']
opt = [0.0, 0.524, 0.131, 0.033, 0.262, 0.003]
ele = [0.098, 0.0, 0.0, 0.0, 0.0, 0.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300,
                                gridspec_kw={'width_ratios': [1, 1.5]})

# 左：甜甜圈
ax1.pie([0.953, 0.098], labels=['光计算 90.65%', '电计算 9.35%'],
        colors=['#E69F00', '#999999'], startangle=90,
        wedgeprops=dict(width=0.42, edgecolor='w'),
        textprops=dict(fontsize=11, fontweight='bold'))
ax1.text(0, 0, '光计算占比\n90.65%', ha='center', va='center', fontsize=13, fontweight='bold')
ax1.set_title('Model 2/3 光计算占比', fontsize=12, fontweight='bold')

# 右：逐层堆叠条
y = np.arange(len(layers))
ax2.barh(y, opt, color='#E69F00', label='光计算 (int8)', edgecolor='w')
ax2.barh(y, ele, left=opt, color='#999999', label='电计算 (FP32)', edgecolor='w')
for i, (o, e) in enumerate(zip(opt, ele)):
    tot = o + e
    if tot > 0.04:
        ax2.text(tot+0.01, i, f'{tot:.3f}M', va='center', fontsize=8.5)
ax2.set_yticks(y); ax2.set_yticklabels(layers, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('MOPs / 张', fontsize=11)
ax2.set_title('逐层运算量分布（补零浪费 = 0）', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.set_xlim(0, 0.66)
plt.tight_layout()
out = 'docs/figures/optical_ratio'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
