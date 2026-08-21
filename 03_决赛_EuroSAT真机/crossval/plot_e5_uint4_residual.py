# -*- coding: utf-8 -*-
"""E5 uint4 残差分布 hw vs sim（据 crossval/stats.json 重绘，2026-08-20 v2）

v2 修复（实测驱动）:
- 原版问题: 面板间距不足导致相邻面板 x 刻度互叠、右面板右缘刻度越界、
  黄色警示框压在密度曲线上
- 本版: 显式定档刻度 + wspace 加大 + ylim 留高 + 文本框渲染后实测高度堆叠
统计口径与 CROSSVAL_REPORT.md §4 E5 残差表一致 (stats.json E5)。
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

stats = json.load(open("stats.json", encoding="utf-8"))["E5"]
hw, sim = stats["uint4_hw"], stats["uint4_sim"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.175, wspace=0.24)

stats_art = {}
warn_art = {}

def draw(ax, d, color, name, note):
    mu, sd = d["mean"], d["std"]
    lo, hi = d["std_ci"]
    x = np.linspace(mu - 4.2 * sd, mu + 4.2 * sd, 800)
    y = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.fill_between(x, y, color=color, alpha=0.30, lw=0)
    ax.plot(x, y, color=color, lw=1.6)
    ax.axvline(mu, color=color, ls=":", lw=1.2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, y.max() * 1.42)  # 顶部留高给文本框
    step = (x[-1] - x[0]) / 6.0
    ticks = [x[0] + step * i for i in range(7)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([("%g" % round(t)) for t in ticks])
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("残差 (MAC)", fontsize=10)
    ax.grid(True, alpha=0.25)
    txt = ("$\\alpha$ = %.4f\n$\\mu$ = %.2f MAC\n$\\sigma$ = %.2f MAC (%.2f counts)"
           % (d["alpha"], mu, sd, sd / 255.0)
           + "\n$\\sigma$ 95%% CI [%.1f, %.1f]" % (lo, hi))
    t1 = ax.text(0.035, 0.97, txt, transform=ax.transAxes, fontsize=9,
                 ha="left", va="top", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, alpha=0.95))
    if note:
        t2 = ax.text(0.035, 0.97, note, transform=ax.transAxes, fontsize=8.5,
                     ha="left", va="top", color="#8a6d00",
                     bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec="#ccb350", alpha=0.95))
        stats_art[ax] = t1
        warn_art[ax] = t2

draw(ax1, hw, "#d62728", "hw（真机，Gazelle）", "uint4 信号低于噪声底，$\\alpha$ 拟合无物理意义")
draw(ax2, sim, "#1f77b4", "sim（osimulator）", None)
ax1.set_ylabel("概率密度", fontsize=10)

fig.suptitle("E5 uint4 残差分布：hw vs sim（MAC 单位）— 拟合 N($\\mu$, $\\sigma^2$)，$\\sigma$ 相差 %.0f×" % (hw["std"] / sim["std"]),
             fontsize=12.5, y=0.965)
fig.text(0.5, 0.022,
         "据 crossval/stats.json E5（10000 GEMM/regime）；原始直方图数据未随档，曲线为拟合分布。"
         "口径与 CROSSVAL_REPORT.md §4 E5 残差表一致。",
         ha="center", fontsize=8.5, color="#666666")

# 渲染后实测统计框高度, 把警示框堆叠到其下方 (留 0.012 轴分数间隙)
fig.canvas.draw()
ren = fig.canvas.get_renderer()
for ax, t1 in stats_art.items():
    t2 = warn_art[ax]
    bb1 = t1.get_window_extent(renderer=ren)
    bb2 = t2.get_window_extent(renderer=ren)
    ax_bb = ax.get_window_extent(renderer=ren)
    h1 = bb1.height / ax_bb.height
    t2.set_position((0.035, 0.97 - h1 - 0.018))

fig.savefig("figures/e5_uint4_residual.png", dpi=160)
print("saved: figures/e5_uint4_residual.png")
