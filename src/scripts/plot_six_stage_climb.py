# -*- coding: utf-8 -*-
"""六阶段量化演进与精度爬升曲线 — 存 docs/figures/six_stage_climb.{png,pdf}

以 Model 2 (SpaceNet V1) 为主角，展示从 FP32 基准到 Gazelle 硬件匹配 int8 的
六阶段量化方法论演进。三模型并排对照，标注关键突破点。

从仓库根目录运行: python src/scripts/plot_six_stage_climb.py

数据来源: docs/EXPERIMENTS.md §13 精度演进总表 (Bug #11 修复后的干净 val 数)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---- 全局样式 ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6

# ---- 六阶段定义 ----
stage_labels = [
    '① FP32\n基准',
    '② QAT 微调\nFP32→int4',
    '③ 从零 QAT\n全 int4',
    '④ STE+噪声\n+非对称 int4',
    '⑤ 混合精度\nConv int4 / Lin fp32',
    '⑥ Gazelle 匹配\nint8 + 硬件噪声',
]

# ---- 三模型数据 ----
# [M1, M2, M3] × 6 stages
data = {
    'Model 1 (VGG, 2.39M)': {
        'y':     [97.17, 85.91, 91.17, 96.46, 98.26, 97.87],
        'fp32':  97.17,
        'color': '#0072B2',   # 蓝
        'ls':    '-',
        'lw':    1.8,
        'ms':    7,
    },
    'Model 2 (SpaceNet V1, 268K)': {
        'y':     [90.15, 73.63, 81.20, 91.06, 91.26, 93.11],
        'fp32':  90.15,
        'color': '#E69F00',   # 橙 (主角)
        'ls':    '-',
        'lw':    3.2,
        'ms':    11,
    },
    'Model 3 (SpaceNet V2+KD, 268K)': {
        'y':     [91.44, 73.22, 83.26, 91.50, 91.13, 92.35],
        'fp32':  91.44,
        'color': '#009E73',   # 绿
        'ls':    '-',
        'lw':    1.8,
        'ms':    7,
    },
}

# ---- 绘制 ----
x = np.arange(len(stage_labels))
fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=300)
fig.patch.set_facecolor('white')

# FP32 基准虚线 + 半透明带
for name, d in data.items():
    ax.axhline(d['fp32'], color=d['color'], ls='--', lw=1.5, alpha=0.55, zorder=2)

# 折线
for name, d in data.items():
    is_hero = 'Model 2' in name
    ax.plot(x, d['y'], marker='o', color=d['color'],
            ls=d['ls'], lw=d['lw'], ms=d['ms'],
            markeredgewidth=0.8, markeredgecolor='white',
            label=name, zorder=5 if is_hero else 4,
            alpha=1.0 if is_hero else 0.75)

# 数据标签 (仅主角 Model 2)
m2_y = data['Model 2 (SpaceNet V1, 268K)']['y']
offsets = [(-0.32, -2.8), (0.35, -3.3), (0.35, -3.3),
           (-0.35, 2.5), (0.35, -3.5), (0.15, 2.6)]
for i, (y_val, (dx, dy)) in enumerate(zip(m2_y, offsets)):
    ax.annotate(f'{y_val:.2f}%', (x[i], y_val),
                textcoords='offset points', xytext=(dx*7, dy*7),
                fontsize=10.5, fontweight='bold',
                color='#E69F00', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#E69F00', alpha=0.85))

# ---- 关键标注 ----
# A: 阶段② 崩盘区
ax.annotate('全员崩盘\n(−11~−18%)',
            xy=(1, 73.63), xytext=(1, 67),
            fontsize=9.5, color='#CC3333', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF0F0', ec='#FFCCCC', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#CC3333', lw=1.5))

# B: 阶段④ 首次反超
ax.annotate('★ 首次反超 FP32\n+0.91%',
            xy=(3, 91.06), xytext=(2.2, 96.5),
            fontsize=9.5, color='#E69F00', ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E69F00', lw=1.8, connectionstyle='arc3,rad=0.2'))

# C: 阶段⑥ 最终突破
ax.annotate('★★ 最终突破\n+2.96% vs FP32',
            xy=(5, 93.11), xytext=(4.5, 97.5),
            fontsize=10, color='#D47500', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='#FFF8E1', ec='#FFB300', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#D47500', lw=2.0, connectionstyle='arc3,rad=-0.2'))

# D: 方法演进箭头 (阶段③→④ 之间)
ax.annotate('方法论跃迁:\n从"适应量化"到\n"量化原生训练"',
            xy=(2.5, 86), fontsize=9, color='#555555', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#DDDDDD', alpha=0.85))

# E: 阶段④/⑤ 并列说明
ax.annotate('④/⑤ 精度持平\n同一量级的\n两种互补方案',
            xy=(3.5, 89.5), fontsize=8, color='#777777', ha='center')

# ---- FP32 基准标签 (右侧) ----
for name, d in data.items():
    if 'Model 2' in name:
        ax.text(5.55, d['fp32'], f'FP32 基准 ({d["fp32"]:.2f}%)',
                fontsize=8.5, color=d['color'], va='center', alpha=0.7,
                fontstyle='italic')

# ---- 横轴 / 纵轴 ----
ax.set_xticks(x)
ax.set_xticklabels(stage_labels, fontsize=9.5)
ax.set_ylabel('Top-1 准确率 (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('量化方法演进阶段', fontsize=13, fontweight='bold')
ax.set_ylim(60, 101)

# ---- 图例 ----
handles, labels = ax.get_legend_handles_labels()
# 添加 FP32 虚线图例
from matplotlib.lines import Line2D
fp32_line = Line2D([0], [0], color='#555555', ls='--', lw=1.5, alpha=0.55)
handles.insert(0, fp32_line)
labels.insert(0, '各模型 FP32 基准')
ax.legend(handles, labels, loc='lower right', fontsize=9.5,
          framealpha=0.9, edgecolor='#DDDDDD')

# ---- 标题 ----
ax.set_title('六阶段量化演进与精度爬升曲线\n'
             '从 FP32 基准 → int4 崩盘 → 从零 QAT → 硬件匹配 int8 (Gazelle 光计算)',
             fontsize=14, fontweight='bold', pad=18)

# ---- 底部注释 ----
fig.text(0.5, 0.01,
         '数据来源: EXPERIMENTS.md §13 | 干净 val 集 (Bug #11 修复后三分 split) | '
         '主角 Model 2: 268K 参数, 六阶段爬升 +19.48% (从最低点), 最终超 FP32 基准 +2.96%',
         ha='center', fontsize=9, color='#888888', fontstyle='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# ---- 输出 ----
out = 'docs/figures/six_stage_climb'
plt.savefig(f'{out}.png', dpi=300, facecolor='white', edgecolor='none')
plt.savefig(f'{out}.pdf', facecolor='white', edgecolor='none')
print(f'[OK] saved: {out}.png')
print(f'[OK] saved: {out}.pdf')
print(f'  Model 2 climb: {data["Model 2 (SpaceNet V1, 268K)"]["y"]}')
