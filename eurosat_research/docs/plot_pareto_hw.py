# plot_pareto_hw.py — 真机上板实验 Pareto 图（MACs vs 真机 acc）
#
# 数据来源（全部为本仓库文档记录的真实 Gazelle 硬件实测，osim/proxy 已排除）：
#   - opticspacenet/OpticSpaceNet迁移至Gazelle真机过程文档.md §5.1（Model 1a/2/3/4）
#   - eurosat_research/docs/round4_hw_deploy.md（J1 champion 90.60）
#   - eurosat_research/docs/round5_c2_drift_robust.md（c2c 91.20、c3d 93.80、c3f 93.20、c3h 93.50）
#   - mnist/MNIST迁移至Gazelle真机过程文档.md（DSQ/STE/LSQ+）
# MACs：Ltsimulator-test src（Model1 156.6M / Model2/3 1.05M / Model4 ~17M）、
#       eurosat_research README（J1 1.38M）；MNIST MLP = 784*128+128*64+64*10 ≈ 0.109M
#
# 用法：python docs/plot_pareto_hw.py  →  生成 docs/pareto_hw_acc.png

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# (label, MACs/张, hw acc %, n 样本, family)
EUROSAT = [
    ("Model 2 SpaceNet V1 (REP=4)", 1.05e6, 86.0, 200, "opticspacenet"),
    ("Model 3 SpaceNet V2 KD",      1.05e6, 89.0, 200, "opticspacenet"),
    ("J1 champion (QAT v5)",        1.38e6, 90.60, 1000, "J1"),
    ("J1 c2c (v6 split-noise)",     1.38e6, 91.20, 1000, "J1"),
    ("J1 c3f (v9 RFF)",             1.38e6, 93.20, 1000, "J1"),
    ("J1 c3h (v9 0.5×RFF)",         1.38e6, 93.50, 1000, "J1"),
    ("J1 c3d (v8 1.5×probe) ★",     1.38e6, 93.80, 1000, "J1"),
    ("Model 4 MiniVGG-GAP",         17.0e6, 96.0, 50,  "opticspacenet"),
    ("Model 1a VGG (n=20 流程验证)", 156.6e6, 100.0, 20, "opticspacenet"),
]

# MNIST 是不同任务（10 分类手写数字，远低于 EuroSAT 难度），只作参考，不进前沿
MNIST = [
    ("MNIST MLP DSQ",  0.109e6, 94.79, 10000),
    ("MNIST MLP STE",  0.109e6, 96.43, 10000),
    ("MNIST MLP LSQ+", 0.109e6, 97.35, 10000),
]

# EuroSAT Pareto 前沿：先按 MACs 分组取该算力档最优 acc，再取随 MACs 升序的严格新高
best_per_macs = {}
for p in EUROSAT:
    if p[1] not in best_per_macs or p[2] > best_per_macs[p[1]][2]:
        best_per_macs[p[1]] = p
frontier = []
best = -1.0
for p in sorted(best_per_macs.values(), key=lambda p: p[1]):
    if p[2] > best:
        frontier.append(p)
        best = p[2]

FAMILY_STYLE = {
    "opticspacenet": dict(color="#1f77b4", marker="o", label="OpticSpaceNet 复赛模型"),
    "J1":            dict(color="#d62728", marker="s", label="J1 QAT 系列（50.3K 参数 / 1.38M MACs）"),
}

fig, ax = plt.subplots(figsize=(9.5, 6.2))

for fam, style in FAMILY_STYLE.items():
    pts = [p for p in EUROSAT if p[4] == fam]
    ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=70, zorder=3, **style)

ax.scatter([p[1] for p in MNIST], [p[2] for p in MNIST], s=60, marker="^",
           color="#7f7f7f", alpha=0.8, zorder=3, label="MNIST MLP（不同任务，参考）")

# 前沿线（跳过 n=20 的 Model 1a？——保留但虚线标注其低置信）
fx = [p[1] for p in frontier]
fy = [p[2] for p in frontier]
ax.plot(fx, fy, "--", color="#2ca02c", lw=1.8, zorder=2, label="Pareto 前沿（EuroSAT 真机）")
for p in frontier:
    ax.scatter([p[1]], [p[2]], s=170, facecolors="none", edgecolors="#2ca02c",
               linewidths=1.8, zorder=4)

ARROW = dict(arrowstyle="-", color="#555555", lw=0.7, shrinkA=0, shrinkB=3)

def anno(text, x, y, dx, dy, fs=8.5, ha="left", color="#222222"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=fs, ha=ha, color=color, arrowprops=ARROW)

# --- OpticSpaceNet 两点（直接标注，空间充足） ---
anno("Model 2 SpaceNet V1 (REP=4)\n86.0% (n=200)", 1.05e6, 86.0, 16, -6)
anno("Model 3 SpaceNet V2 KD\n89.0% (n=200)", 1.05e6, 89.0, -16, -34, ha="right")
anno("Model 4 MiniVGG-GAP\n96.0% (n=50)", 17.0e6, 96.0, -8, -36, ha="right")
anno("Model 1a VGG\n100% (n=20，仅流程验证)", 156.6e6, 100.0, -8, -38, ha="right")

# --- J1 家族（同一 MACs，标签引线呈扇形展开） ---
anno("champion (QAT v5)\n90.60%", 1.38e6, 90.60, -100, -24, ha="right", color="#d62728")
anno("c2c (v6 split)\n91.20%",    1.38e6, 91.20, -18, 14,  ha="right", color="#d62728")
anno("c3f (v9)\n93.20%",          1.38e6, 93.20, 20, -42, color="#d62728")
anno("c3h (v9)\n93.50%",          1.38e6, 93.50, 34, -16, color="#d62728")
anno("c3d (v8 1.5×probe)\n93.80% ★ SOTA", 1.38e6, 93.80, 24, 12,
     color="#d62728")

# --- MNIST（不同任务参考，只标最高点） ---
anno("MNIST MLP LSQ+ 97.35%\n(STE 96.43 / DSQ 94.79, n=10000)",
     0.109e6, 97.35, 16, -10, color="#555555")

ax.set_xscale("log")
ax.set_xlabel("MACs / 张（log 尺度）")
ax.set_ylabel("Gazelle 真机 acc（%）")
ax.set_title("真机上板实验 Pareto：MACs vs 实测精度\n（全部 compass_sdk 真机实测；J1 系列为 1000 样本同窗口径）")
ax.set_ylim(80, 103)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
out = __file__.replace(".py", "").replace("plot_pareto_hw", "pareto_hw_acc") + ".png"
fig.savefig(out, dpi=160)
print(f"saved: {out}")
print("EuroSAT Pareto 前沿：")
for p in frontier:
    print(f"  {p[1]/1e6:8.3f}M MACs  {p[2]:6.2f}%  (n={p[3]})  {p[0]}")
