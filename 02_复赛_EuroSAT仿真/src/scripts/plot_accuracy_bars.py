# -*- coding: utf-8 -*-
"""三模型精度对比柱状图 — 存 docs/figures/accuracy_bars.{png,pdf}
从仓库根目录运行: python src/scripts/plot_accuracy_bars.py"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

models = ['Model 1 (A)', 'Model 2', 'Model 3']
fp32  = [97.17, 90.15, 91.44]
val8  = [97.87, 92.06, 91.83]
osim  = [98.00, 90.43, 90.28]
osim_note = ['(q50抽样)', '(全量5400)', '(全量5400)']

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
b1 = ax.bar(x - w, fp32, w, label='FP32 基准', color='#999999', edgecolor='#666')
b2 = ax.bar(x,     val8, w, label='int8 val（干净）', color='#56B4E9', edgecolor='#2a7ab8')
b3 = ax.bar(x + w, osim, w, label='osim 真机', color='#E69F00', edgecolor='#b87700')

for bars, vals in [(b1, fp32), (b2, val8), (b3, osim)]:
    for r, v in zip(bars, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.12, f'{v:.2f}', ha='center', fontsize=8.5)
for i, n in enumerate(osim_note):
    ax.text(x[i]+w, osim[i]-1.1, n, ha='center', fontsize=7.5, color='#b87700')

ax.axhline(90.15, xmin=0.55, xmax=0.72, color='#0072B2', ls='--', lw=1)
ax.annotate('M2 真机反超 FP32 +0.28%', xy=(1+w, 90.43), xytext=(1.6, 95.2),
            fontsize=9, color='#0072B2', arrowprops=dict(arrowstyle='->', color='#0072B2'))
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Top-1 准确率 (%)', fontsize=12)
ax.set_ylim(86, 100)
ax.set_title('三模型精度对比：FP32 → int8 → 光计算真机', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
out = 'docs/figures/accuracy_bars'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
