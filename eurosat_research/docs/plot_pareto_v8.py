# -*- coding: utf-8 -*-
"""Pareto 图 (v8 口径) — X0 单阶段 v8 重测 + M5-M8 两阶段 v8 参考。

两组口径:
  - X0 单阶段 v8 (主系列): qat_v8 probe 实测组分噪声 1.5x 余量, 从零 160ep, SGD lr0.05, test 5400 clean
  - M5-M8 两阶段 v8: clean 160ep -> v8 60ep 续训 (协议不同, 仅参考, 不入前沿)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 数据 ----------------
# X0 单阶段 v8 (主系列): (label, MACs M, acc % 或 None=running, annotation offset)
x0_single = [
    ("x0r_w075",        0.86,  94.65, (6, -12)),
    ("x0_ctrl (J1)",    1.378, 95.57, (6, 10)),
    ("x0_pool3",        1.378, 95.43, (-64, 4)),
    ("x0_blurpool",     1.378, 94.31, (6, -12)),
    ("x0_pool4",        1.378, 94.81, (8, -14)),
    ("x0r_rf_stem5",    2.16,  95.89, (6, -14)),
    ("x0r_w150",        2.77,  96.20, (8, 6)),
    ("x0r_rf_s2k3",     4.52,  96.39, (-72, -16)),
    ("x0r_w200",        4.62,  96.74, (8, -12)),
    ("x0r_model4e",    17.04,  96.81, (-80, 8)),
]

# X0 多 seed 臂: (label, MACs M, [acc per seed], annotation offset)
x0_multiseed = [
    ("M9 (w075ds3)", 1.522, [95.87, 95.69], (10, -26)),
    ("x0_dsconv3", 2.557, [96.22, 96.67], (-90, -20)),
    ("M10 (ds3pool3)", 2.557, [96.76, 96.56], (-104, 12)),
]

# M5-M8 两阶段 v8 (clean 160ep -> v8 60ep); 数值为 X0 独立复测值 (x0/results/M_validate.md)
m_series = [
    ("M7",   0.86, 94.98, (8, -18)),
    ("M6",   1.38, 95.28, (8, -18)),
    ("M8",   2.16, 96.20, (-56, 10)),
    ("M5",   5.31, 96.61, (10, 4)),
]

# ---------------- 画图 ----------------
fig, ax = plt.subplots(figsize=(11.5, 7.5))

C_MAIN = "#1f77b4"
C_MULTI = "#2ca02c"
C_M = "#d62728"
C_V5 = "#999999"

# 主系列
xs = [p[1] for p in x0_single if p[2] is not None]
ys = [p[2] for p in x0_single if p[2] is not None]
ax.scatter(xs, ys, color=C_MAIN, marker="o", s=70, edgecolors="white",
           linewidths=0.6, zorder=3, label="X0 单阶段 v8 160ep (主系列)")
for label, x, y, off in x0_single:
    if y is None:
        ax.annotate(f"{label}\n(running)", (x, 91.3), fontsize=7.5,
                    color=C_MAIN, ha="center", alpha=0.7, zorder=4)
        ax.scatter([x], [91.3], color=C_MAIN, marker="o", s=40, alpha=0.3, zorder=2)
    else:
        ax.annotate(f"{label} {y:.2f}", (x, y), textcoords="offset points",
                    xytext=off, fontsize=8, color=C_MAIN, zorder=4)

# 多 seed: mean 点 + 两点连线
mx, my = [], []
for label, x, seeds, off in x0_multiseed:
    mean = sum(seeds) / len(seeds)
    mx.append(x); my.append(mean)
    ax.plot([x, x], [min(seeds), max(seeds)], color=C_MULTI, lw=1.2, zorder=2)
    ax.scatter([x] * len(seeds), seeds, color=C_MULTI, s=18, alpha=0.55, zorder=3)
    ax.annotate(f"{label} {mean:.2f}", (x, mean), textcoords="offset points",
                xytext=off, fontsize=8, color=C_MULTI, zorder=4)
ax.scatter(mx, my, color=C_MULTI, marker="D", s=65, edgecolors="white",
           linewidths=0.6, zorder=3, label="X0 多 seed 臂 (mean + seed 连线)")

# M5-M8
ax.scatter([p[1] for p in m_series], [p[2] for p in m_series], color=C_M,
           marker="X", s=110, edgecolors="white", linewidths=0.6, zorder=3,
           label="M5-M8 两阶段 v8 (clean160→v8 60ep)")
for label, x, y, off in m_series:
    ax.annotate(f"{label} {y:.2f}", (x, y), textcoords="offset points",
                xytext=off, fontsize=8, color=C_M, zorder=4)

# Pareto 前沿 (v8 主系列 + 多 seed mean + M5-M8; 先按 x 取 max y 再取上包络,
# 否则同 MACs 的 pool3/pool4 等被支配点会因升序逐个刷新 running max 而混入前沿)
from collections import defaultdict
by_x = defaultdict(float)
for _, x, y, _ in x0_single:
    if y is not None:
        by_x[x] = max(by_x[x], y) if x in by_x else y
for _, x, s, _ in x0_multiseed:
    m = sum(s) / len(s)
    by_x[x] = max(by_x[x], m) if x in by_x else m
for _, x, y, _ in m_series:
    by_x[x] = max(by_x[x], y) if x in by_x else y
frontier, best = [], -1
for x in sorted(by_x):
    y = by_x[x]
    if y > best:
        frontier.append((x, y)); best = y
if len(frontier) >= 2:
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "k--", lw=1.2,
            alpha=0.5, zorder=2, label="Pareto 前沿 (含 M5-M8)")
print("frontier:", [(f"{x:.3g}M", y) for x, y in frontier])

ax.axvline(2.0, color="gray", ls="--", lw=1, alpha=0.5)
ax.annotate("≤2M 严格预算", (2.03, 90.6), fontsize=8, color="gray")

ax.set_xscale("log")
ax.set_xlabel("激活计算量 MACs / 张 (log scale)")
ax.set_ylabel("test acc (%, 5400 clean)")
ax.set_title("Perf vs MACs — v8 口径 Pareto (X0)\n"
             "主系列 = qat_v8 单阶段 160ep; M5-M8 = 两阶段 v8 (clean160→v8 60ep, 口径不同)")
ax.set_xlim(0.75, 25)
ax.set_ylim(93.8, 97.2)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

fig.tight_layout()
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160)
print("saved:", out)
