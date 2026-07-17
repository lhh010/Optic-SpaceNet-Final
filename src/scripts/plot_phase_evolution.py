# -*- coding: utf-8 -*-
"""精度演进折线图（Phase 1->6）— 存 docs/figures/phase_evolution.{png,pdf}
从仓库根目录运行: python src/scripts/plot_phase_evolution.py"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

phases = ['P1\n微调', 'P2\n从零QAT', 'P4\nSTE+噪声', 'P5\nMixed', 'P6\nGazelle int8']
x = np.arange(len(phases))
data = {
    'Model 1 (VGG, 2.39M)': {'c': '#0072B2', 'y': [85.91, 91.17, 96.46, 98.26, 97.87], 'fp32': 97.17},
    'Model 2 (SpaceNet V1, 268K)': {'c': '#E69F00', 'y': [73.63, 81.20, 74.35, 91.26, 92.06], 'fp32': 90.15},
    'Model 3 (SpaceNet V2+KD, 268K)': {'c': '#009E73', 'y': [73.22, 83.26, 78.26, 91.13, 91.83], 'fp32': 91.44},
}
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
for name, d in data.items():
    ax.plot(x, d['y'], '-o', color=d['c'], lw=2.2, ms=8, label=name)
    ax.axhline(d['fp32'], color=d['c'], ls=':', lw=1.2, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(phases)
ax.set_ylabel('Top-1 准确率 (%)', fontsize=12)
ax.set_xlabel('量化迭代阶段', fontsize=12)
ax.set_title('量化迁移精度演进：从全员崩盘（P1）到反超 FP32（P5/P6）', fontsize=13, fontweight='bold')
ax.annotate('P1 微调全员失败\n(-11~-18%)', xy=(0, 78), fontsize=9, color='#555',
            ha='center', bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3CD', ec='#EEE', alpha=0.9))
ax.annotate('P4 M2/M3\nQAT-Conv 全关 bug', xy=(2, 74.35), fontsize=8, color='#555', ha='center')
ax.annotate('反超 FP32', xy=(4, 92.06), xytext=(3.1, 95.5), fontsize=9, color='#0072B2',
            arrowprops=dict(arrowstyle='->', color='#0072B2'))
ax.set_ylim(70, 100)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
out = 'docs/figures/phase_evolution'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
