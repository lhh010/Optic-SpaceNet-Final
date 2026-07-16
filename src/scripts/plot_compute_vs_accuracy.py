import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 设置学术标准排版样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = '#999999'

# 精确真实数据
models = [
    {
        'name': 'Model 2: SpaceNet V1',
        'color': '#E69F00', # 暖橙
        'x': 1.051e6,
        'y_fp32': 90.15,
        'y_qat': 92.06,
        'y_osim': 89.00,
        'param_size': 0.268,
        'optical_ratio': 90.7,
        'bubble_size': 120
    },
    {
        'name': 'Model 3: SpaceNet V2+KD',
        'color': '#009E73', # 经典绿
        'x': 1.051e6,
        'y_fp32': 91.44,
        'y_qat': 91.83,
        'y_osim': 90.80,
        'param_size': 0.268,
        'optical_ratio': 90.7,
        'bubble_size': 120
    },
    {
        'name': 'Model 1-A: VGG+BN',
        'color': '#0072B2', # 深蓝
        'x': 1.5663e8,
        'y_fp32': 97.17,
        'y_qat': 97.87,
        'y_osim': 98.00,
        'param_size': 2.39,
        'optical_ratio': 97.7,
        'bubble_size': 400
    },
    {
        'name': 'Model 1-B: VGG+BN',
        'color': '#56B4E9', # 天蓝
        'x': 1.5663e8,
        'y_fp32': 97.17,
        'y_qat': 98.02,
        'y_osim': 100.00,
        'param_size': 2.39,
        'optical_ratio': 73.6,
        'bubble_size': 400
    }
]

# 创建画布
fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)

# 设置轴标尺和范围
ax.set_xscale('log')
ax.set_xlim(5e5, 3.5e8)
ax.set_ylim(70, 101.5)

# 为避免完全重合，引入微小的 x 轴视觉偏移（Jitter）
jitter = {
    'Model 2: SpaceNet V1': 0.93,
    'Model 3: SpaceNet V2+KD': 1.07,
    'Model 1-A: VGG+BN': 0.93,
    'Model 1-B: VGG+BN': 1.07
}

# 绘制数据点与垂直虚线
for m in models:
    scale_factor = jitter[m['name']]
    x_plot = m['x'] * scale_factor
    ys = [m['y_fp32'], m['y_qat'], m['y_osim']]
    color = m['color']
    size = m['bubble_size']

    # 1. 垂直虚线
    ax.plot([x_plot, x_plot], [min(ys), max(ys)], color=color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)

    # 2. FP32 基准 (◇ - 镂空菱形)
    ax.scatter(x_plot, m['y_fp32'], marker='d', s=size, color='none', edgecolor=color, linewidth=2.5, zorder=4)
    # 3. int8 QAT (● - 半透明圆)
    ax.scatter(x_plot, m['y_qat'], marker='o', s=size, color=color, alpha=0.5, edgecolor=color, linewidth=1.5, zorder=3)
    # 4. osim 真实硬件 (▲ - 三角形加黑边)
    ax.scatter(x_plot, m['y_osim'], marker='^', s=size, color=color, edgecolor='black', linewidth=1.5, zorder=5)

# 绘制 ≈149x 计算量对比箭头
ax.annotate(
    '',
    xy=(1.5e8, 85),
    xytext=(1.1e6, 85),
    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5, ls='-')
)
ax.text(1.28e7, 86, "≈149× compute reduction", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#222222')

# 聚类信息卡片式标注
# SpaceNet 聚类
ax.text(
    1.051e6, 94.0,
    "SpaceNet (native int8 optical)\n0.268M params | 90.7% opt. MOPs",
    ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#111111',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDFDFD', edgecolor='#DDDDDD', alpha=0.9, lw=1)
)

# VGG 聚类
ax.text(
    1.5663e8, 93.5,
    "VGG baseline\n2.39M params\n97.7% / 73.6% opt. MOPs",
    ha='center', va='top', fontsize=10.5, fontweight='bold', color='#111111',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDFDFD', edgecolor='#DDDDDD', alpha=0.9, lw=1)
)

# 手动构建图例 1 (推理精度/阶段)
legend_elements_marker = [
    Line2D([0], [0], marker='d', color='w', label='FP32 Baseline (Benchmark)', markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, markersize=10),
    Line2D([0], [0], marker='o', color='w', label='int8 QAT (Quantization-Aware Training)', markerfacecolor='#999999', alpha=0.6, markeredgecolor='#666666', markersize=11),
    Line2D([0], [0], marker='^', color='w', label='osim (Real Optical-Hardware Inference)', markerfacecolor='black', markeredgecolor='black', markersize=11)
]

# 手动构建图例 2 (模型架构与参数大小)
legend_elements_model = [
    Line2D([0], [0], color='#E69F00', lw=2, linestyle='--', marker='o', markersize=6, label='Model 2: SpaceNet V1 (0.268M)'),
    Line2D([0], [0], color='#009E73', lw=2, linestyle='--', marker='o', markersize=6, label='Model 3: SpaceNet V2+KD (0.268M)'),
    Line2D([0], [0], color='#0072B2', lw=2, linestyle='--', marker='o', markersize=9, label='Model 1-A: VGG+BN (2.39M, 97.7% opt)'),
    Line2D([0], [0], color='#56B4E9', lw=2, linestyle='--', marker='o', markersize=9, label='Model 1-B: VGG+BN (2.39M, 73.6% opt)')
]

# 渲染双重图例
leg1 = ax.legend(handles=legend_elements_marker, loc='lower left', bbox_to_anchor=(0.50, 0.02), title="Inference Precision / Stage", frameon=True, framealpha=0.95, facecolor='white', edgecolor='#CCCCCC')
ax.add_artist(leg1)
leg2 = ax.legend(handles=legend_elements_model, loc='lower left', bbox_to_anchor=(0.02, 0.02), title="Model Architecture (Bubble size ∝ params)", frameon=True, framealpha=0.95, facecolor='white', edgecolor='#CCCCCC')

# 标题与轴标签
ax.set_xlabel("Compute per image (MOPs, log scale)", fontsize=12, fontweight='bold', labelpad=12)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight='bold', labelpad=12)
ax.set_title("Compute vs. Accuracy — EuroSAT on Optical Computing (int8, Gazelle osimulator)", fontsize=13, fontweight='bold', pad=18)

ax.tick_params(axis='both', which='major', labelsize=11)

# 添加页脚小字
plt.figtext(
    0.05, 0.01,
    "osim = real optical-hardware simulation; Model 2/3 n=500, Model 1 n=50 (sampled). Source: EXPERIMENTS.md",
    fontsize=8.5, color='#555555', ha='left', style='italic'
)

# 调整边距并保存
plt.subplots_adjust(bottom=0.14)
out = 'docs/figures/compute_vs_accuracy_final'
plt.savefig(f'{out}.png', bbox_inches='tight', dpi=300)   # 栅格 (预览)
plt.savefig(f'{out}.pdf', bbox_inches='tight')            # 矢量 (论文/打印)
plt.savefig(f'{out}.svg', bbox_inches='tight')            # 矢量 (可编辑)
print(f"saved: {out}.png / {out}.pdf / {out}.svg")
plt.show()
