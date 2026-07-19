# Compute vs Accuracy 散点图 (osimulator) — 布局调整版
# 数据口径: EXPERIMENTS.md §11.13 (M2/M3 全量 5400) / §11.14 (M1 A/B q650)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

# 精确真实数据
models = [
    {'name': 'Model 2: SpaceNet V1',    'color': '#E69F00', 'x': 1.051e6,  'y_fp32': 90.15, 'y_qat': 92.06, 'y_osim': 90.43, 'bubble_size': 120},
    {'name': 'Model 3: SpaceNet V2+KD', 'color': '#009E73', 'x': 1.051e6,  'y_fp32': 91.44, 'y_qat': 91.83, 'y_osim': 90.28, 'bubble_size': 120},
    {'name': 'Model 1-A: VGG+BN',       'color': '#0072B2', 'x': 1.5663e8, 'y_fp32': 97.17, 'y_qat': 97.87, 'y_osim': 98.15, 'bubble_size': 400},
    {'name': 'Model 1-B: VGG+BN',       'color': '#56B4E9', 'x': 1.5663e8, 'y_fp32': 97.17, 'y_qat': 98.02, 'y_osim': 97.54, 'bubble_size': 400},
]

fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
ax.set_xscale("log")
ax.set_xlim(5e5, 3.5e8)
ax.set_ylim(86, 101.8)

# 避免完全重合的 x 轴视觉偏移 (Jitter)
jitter = {
    'Model 2: SpaceNet V1': 0.93,
    'Model 3: SpaceNet V2+KD': 1.07,
    'Model 1-A: VGG+BN': 0.93,
    'Model 1-B: VGG+BN': 1.07,
}

# 数值标签: 左侧系列标左, 右侧系列标右; 近点做 ±5pt 垂直错位; M1-B 的 fp32 与 M1-A 重复不标
label_cfg = {
    'Model 2: SpaceNet V1':    {'side': -1, 'dy': {'fp32': -5, 'qat': 0,    'osim': 5}},
    'Model 3: SpaceNet V2+KD': {'side':  1, 'dy': {'fp32': -5, 'qat': 5,    'osim': 0}},
    'Model 1-A: VGG+BN':       {'side': -1, 'dy': {'fp32': 0,  'qat': -5,   'osim': 5}},
    'Model 1-B: VGG+BN':       {'side':  1, 'dy': {'fp32': None, 'qat': 5,  'osim': -5}},
}

for m in models:
    x = m["x"] * jitter[m["name"]]
    ys = [m["y_fp32"], m["y_qat"], m["y_osim"]]
    c = m["color"]; s = m["bubble_size"]
    ax.plot([x, x], [min(ys), max(ys)], '--', color=c, lw=1.5)
    ax.scatter(x, m["y_fp32"], marker='d', s=s, facecolors='none', edgecolors=c, lw=2.5, zorder=4)
    ax.scatter(x, m["y_qat"], marker='o', s=s, color=c, alpha=.5, edgecolors=c, zorder=3)
    ax.scatter(x, m["y_osim"], marker='^', s=s, color=c, edgecolors='black', zorder=5)

    cfg = label_cfg[m["name"]]
    for key, yv in (('fp32', m["y_fp32"]), ('qat', m["y_qat"]), ('osim', m["y_osim"])):
        dy = cfg['dy'][key]
        if dy is None:
            continue
        ax.annotate(f"{yv:.2f}", (x, yv),
                    xytext=(cfg['side'] * 14, dy), textcoords='offset points',
                    ha='right' if cfg['side'] < 0 else 'left', va='center',
                    fontsize=9,
                    fontweight='bold' if key == 'osim' else 'normal',
                    color='black' if key == 'osim' else '#555555', zorder=6)

# ≈149x 计算量对比箭头
ax.annotate("", xy=(1.5e8, 86.8), xytext=(1.1e6, 86.8),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color="#333"))
ax.text(1.28e7, 87.2, "≈149× compute reduction", ha='center', fontsize=11, fontweight='bold')

# SpaceNet 聚类卡片 + 指示箭头
ax.text(6.8e5, 94.1,
        "SpaceNet (native int8 optical)\n0.268M params | 90.65% opt. MOPs",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
ax.annotate("", xy=(1.0e6, 92.0), xytext=(7.6e5, 93.8),
            arrowprops=dict(arrowstyle="->", color="#666"))

# VGG 聚类卡片 + 指示箭头
ax.text(3.3e8, 99.1,
        "VGG baseline\n2.39M params\n97.74% / 73.64% opt. MOPs",
        ha='right',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
ax.annotate("", xy=(1.57e8, 98.0), xytext=(2.15e8, 98.8),
            arrowprops=dict(arrowstyle="->", color="#666"))

# 图例 1: 精度阶段
leg1 = ax.legend(handles=[
    Line2D([0], [0], marker='d', color='w', markerfacecolor='none', markeredgecolor='black', label='FP32'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='QAT'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='black', label='osim'),
], loc='upper left', bbox_to_anchor=(0.53, -0.09))
ax.add_artist(leg1)

# 图例 2: 模型
leg2 = ax.legend(handles=[
    Line2D([0], [0], color='#E69F00', ls='--', label='Model2'),
    Line2D([0], [0], color='#009E73', ls='--', label='Model3'),
    Line2D([0], [0], color='#0072B2', ls='--', label='Model1-A'),
    Line2D([0], [0], color='#56B4E9', ls='--', label='Model1-B'),
], loc='upper left', bbox_to_anchor=(0.01, -0.09))

ax.set_xlabel("Compute per image (MOPs, log scale)")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Compute vs. Accuracy — EuroSAT on Optical Computing (int8, Gazelle osimulator)", pad=26)

plt.tight_layout(rect=[0, 0.08, 1, 0.97])
plt.subplots_adjust(bottom=0.23)

# 页脚小字: 样本量口径 (防"抽样数冒充全量"质疑)
plt.figtext(
    0.05, 0.02,
    "osim = real optical-hardware simulation; Model 2/3 n=5400 (full test set), Model 1 n=650 (sampled). Source: EXPERIMENTS.md",
    fontsize=8.5, color='#555555', ha='left', style='italic')

out = 'docs/figures/compute_vs_accuracy_final'
plt.savefig(f'{out}.png', bbox_inches='tight', dpi=300)   # 栅格 (预览)
plt.savefig(f'{out}.pdf', bbox_inches='tight')            # 矢量 (论文/打印)
plt.savefig(f'{out}.svg', bbox_inches='tight')            # 矢量 (可编辑)
print(f"saved: {out}.png / {out}.pdf / {out}.svg")
plt.show()
