import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# X轴：7个阶段
stages = [
    "① FP32 基准", 
    "② QAT 微调", 
    "③ 从零 QAT", 
    "④ 非对称量化\n+噪声", 
    "⑤ 混合精度", 
    "⑥ Gazelle\nQAT", 
    "⑦ osimulator\n真机实测"
]
x = np.arange(len(stages))

# Y轴：各个模型在各阶段的准确率 (%)
m1_a = [97.17, 85.91, 91.17, 96.46, 98.26, 97.87, 98.15]
m1_b = [97.17, 85.91, 91.17, 96.46, 98.26, 98.02, 97.54]
m2 = [90.15, 73.63, 81.20, 91.06, 91.26, 92.06, 90.43]
m3 = [91.44, 73.22, 83.26, 91.50, 91.13, 91.83, 90.28]

# 创建图表
plt.figure(figsize=(11, 6), dpi=150)

# 绘制折线
# M1-A 和 M1-B 前5个阶段重合，从第5个阶段开始分叉
plt.plot(x, m1_a, marker='o', linewidth=2, color='#1f77b4', label='M1 (VGG) - 变体 A')
plt.plot(x, m1_b, marker='s', linewidth=2, linestyle='--', color='#aec7e8', label='M1 (VGG) - 变体 B')
plt.plot(x, m2, marker='^', linewidth=2, color='#ff7f0e', label='M2 (SN V1)')
plt.plot(x, m3, marker='d', linewidth=2, color='#2ca02c', label='M3 (SN V2+KD)')

# 添加数据标签 (避免重叠，略微调整位置)
for i in range(len(stages)):
    if i < 5:
        plt.text(i, m1_a[i] + 0.8, f"{m1_a[i]}%", ha='center', va='bottom', color='#1f77b4', fontsize=9)
    else:
        plt.text(i, m1_a[i] + 0.8, f"A:{m1_a[i]}%", ha='center', va='bottom', color='#1f77b4', fontsize=9)
        plt.text(i, m1_b[i] - 1.2, f"B:{m1_b[i]}%", ha='center', va='top', color='#6fa8dc', fontsize=9)
        
    plt.text(i, m2[i] - 1.2, f"{m2[i]}%", ha='center', va='top', color='#d62728', fontsize=9)
    plt.text(i, m3[i] + 0.8, f"{m3[i]}%", ha='center', va='bottom', color='#2ca02c', fontsize=9)

# 样式美化
plt.title("不同模型在各技术阶段的准确率走势对比", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("技术演进阶段", fontsize=12, labelpad=10)
plt.ylabel("准确率 (%)", fontsize=12, labelpad=10)
plt.xticks(x, stages, fontsize=10)
plt.ylim(65, 105)  # 留出上下间距展示标签
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=True)

# 调整布局并展示
plt.tight_layout()
plt.show()