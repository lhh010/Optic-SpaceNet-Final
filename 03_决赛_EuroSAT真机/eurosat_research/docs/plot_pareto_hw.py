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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# (label, MACs/张, hw acc %, n 样本, family)
# 2026-08-17 收官：M4-M10 全量 5400 真机口径（board_validation 归档）；J1 系列为 1000 样本历史点
EUROSAT = [
    ("Model 2 SpaceNet V1 (REP=4)", 1.05e6, 86.0, 200, "opticspacenet"),
    ("Model 3 SpaceNet V2 KD",      1.05e6, 89.0, 200, "opticspacenet"),
    ("J1 champion (QAT v5)",        1.38e6, 90.60, 1000, "J1"),
    ("J1 c2c (v6 split-noise)",     1.38e6, 91.20, 1000, "J1"),
    ("J1 c3f (v9 RFF)",             1.38e6, 93.20, 1000, "J1"),
    ("J1 c3h (v9 0.5×RFF)",         1.38e6, 93.50, 1000, "J1"),
    ("J1 c3d (v8 1.5×probe) ★",     1.38e6, 93.80, 1000, "J1"),
    # 决赛全量 5400 真机（2026-08-17 收官）：M7 81.7 架构不匹配（板端），仍列出作参考
    ("M7 (J1-w075, MaxPool)",       0.86e6, 81.7, 1800, "final"),
    ("M6 (J1 全1×1)",               1.38e6, 92.78, 5400, "final"),
    ("M9 (w075ds3) ★",              1.52e6, 94.43, 5400, "final"),
    ("M8 (rf_stem5)",               2.16e6, 87.78, 5400, "final"),
    ("M10 (ds3pool3) ★",            2.56e6, 95.33, 5400, "final"),
    ("M5 (J1-RF+, 3×3)",            5.31e6, 90.17, 5400, "final"),
    ("Model 4 MiniVGG-GAP",         17.03e6, 94.19, 5400, "final"),
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
    "final":         dict(color="#2ca02c", marker="D", label="决赛全量 5400 真机（2026-08-17 收官）"),
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

ARROW = dict(arrowstyle="-", color="#999999", lw=0.45, shrinkA=0, shrinkB=2)

# ==== 自动避让标注: 候选位置按距离排序, 与已放文字/数据点/图例碰撞则自动退档 ====
_RENDERER = None
_PT_PX = []
_PLACED = []
_FRONT_SEGS = []

def _init_placement():
    global _RENDERER, _PT_PX, _FRONT_SEGS
    fig.canvas.draw()
    _RENDERER = fig.canvas.get_renderer()
    _PT_PX = [ax.transData.transform((p[1], p[2])) for p in EUROSAT] + \
             [ax.transData.transform((p[1], p[2])) for p in MNIST]
    # 前沿虚线线段 (像素坐标) 也作为禁区, 避免标签文字穿越前沿线
    _FRONT_SEGS = [(ax.transData.transform((fx[i], fy[i])),
                    ax.transData.transform((fx[i+1], fy[i+1])))
                   for i in range(len(fx) - 1)]
    _leg = ax.get_legend()
    if _leg is not None:
        _lb = _leg.get_window_extent(_RENDERER)
        _PLACED.append((_lb.x0, _lb.y0, _lb.x1, _lb.y1))

def _seg_hits_box(p, q, x0, y0, x1, y1, pad=2):
    """线段 (p,q) 与盒子 [x0-pad, x1+pad]x[y0-pad, y1+pad] 是否相交 (Liang-Barsky)."""
    x0 -= pad; x1 += pad; y0 -= pad; y1 += pad
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            r = qq / pp
            if pp < 0:
                if r > t1: return False
                if r > t0: t0 = r
            else:
                if r < t0: return False
                if r < t1: t1 = r
    return True

def _try_place(text, x, y, dx, dy, ha, color, fontsize):
    # Annotation.get_window_extent 无条件包含箭头 patch bbox (源码 union, 不看 visible),
    # 会把 bbox 撑成 点->文字 的大三角带, 永远误判压点。
    # 对策: 用无箭头临时对象测量纯文字 bbox, 通过后再创建带箭头正式标注。
    tmp = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                      fontsize=fontsize, ha=ha, color=color)
    bb = tmp.get_window_extent(_RENDERER)
    tmp.remove()
    if bb.x0 < ax.bbox.x0 + 4 or bb.x1 > ax.bbox.x1 - 4 or bb.y0 < ax.bbox.y0 + 4 or bb.y1 > ax.bbox.y1 - 4:
        return None
    for (bx0, by0, bx1, by1) in _PLACED:
        if min(bx1, bb.x1) - max(bx0, bb.x0) > -2 and min(by1, bb.y1) - max(by0, bb.y0) > -2:
            return None
    for (qx, qy) in _PT_PX:
        if bb.x0 - 5 < qx < bb.x1 + 5 and bb.y0 - 5 < qy < bb.y1 + 5:
            return None
    for (pseg, qseg) in _FRONT_SEGS:
        if _seg_hits_box(pseg, qseg, bb.x0, bb.y0, bb.x1, bb.y1):
            return None
    _PLACED.append((bb.x0, bb.y0, bb.x1, bb.y1))
    return ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                       fontsize=fontsize, ha=ha, color=color, zorder=6, arrowprops=ARROW)

def anno_auto(text, x, y, color="#222222", fontsize=8.5):
    if _RENDERER is None:
        _init_placement()
    cand = []
    for dist in (16, 30, 50, 80, 115, 155, 200, 250):
        for dy in (0, 18, -18, 38, -38, 60, -60):
            cand.append((dist, dy, "left")); cand.append((-dist, dy, "right"))
            if dy != 0:
                cand.append((dist, dy, "right")); cand.append((-dist, dy, "left"))
    cand.sort(key=lambda c: abs(c[0]) + abs(c[1]))
    for dx, dy, ha in cand:
        t = _try_place(text, x, y, dx, dy, ha, color, fontsize)
        if t is not None:
            return
    _try_place(text, x, y, 250, 60, "left", color, fontsize)

# 图例区也加入禁区
_leg = ax.get_legend()
if _leg is not None:
    _lb = _leg.get_window_extent(_RENDERER)
    _PLACED.append((_lb.x0, _lb.y0, _lb.x1, _lb.y1))

def anno(text, x, y, dx, dy, fs=8.5, ha="left", color="#222222"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=fs, ha=ha, color=color, arrowprops=ARROW)

# --- 全部标注: 收集到函数里, 在 tight_layout + 图例之后按重要性顺序自适应放置 ---
def _place_all():
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    _init_placement()
    # 重要性优先: SOTA 双星最先挑位置
    anno_auto("M10 (ds3pool3)\n95.33% ★ 全量 SOTA", 2.56e6, 95.33, color="#2ca02c", fontsize=9)
    anno_auto("M9 (w075ds3)\n94.43% ★", 1.52e6, 94.43, color="#2ca02c", fontsize=9)
    anno_auto("c3d (v8 1.5×probe)\n93.80% ★ SOTA", 1.38e6, 93.80, color="#d62728")
    anno_auto("M4 (MiniVGG-GAP)\n94.19%", 17.03e6, 94.19, color="#2ca02c")
    anno_auto("Model 1a VGG\n100% (n=20，仅流程验证)", 156.6e6, 100.0, color="#1f77b4")
    anno_auto("Model 3 SpaceNet V2 KD\n89.0% (n=200)", 1.05e6, 89.0, color="#1f77b4")
    anno_auto("Model 2 SpaceNet V1 (REP=4)\n86.0% (n=200)", 1.05e6, 86.0, color="#1f77b4")
    anno_auto("c3h (v9)\n93.50%", 1.38e6, 93.50, color="#d62728")
    anno_auto("c3f (v9)\n93.20%", 1.38e6, 93.20, color="#d62728")
    anno_auto("c2c (v6 split)\n91.20%", 1.38e6, 91.20, color="#d62728")
    anno_auto("champion (QAT v5)\n90.60%", 1.38e6, 90.60, color="#d62728")
    anno_auto("M6 (J1 1×1)\n92.78%", 1.38e6, 92.78, color="#2ca02c")
    anno_auto("M5 (J1-RF+)\n90.17%", 5.31e6, 90.17, color="#2ca02c")
    anno_auto("M8 (rf_stem5)\n87.78%", 2.16e6, 87.78, color="#2ca02c")
    anno_auto("M7 (MaxPool)\n81.7% 判定关闭", 0.86e6, 81.7, color="#2ca02c")
    anno_auto("MNIST MLP LSQ+ 97.35%\n(STE 96.43 / DSQ 94.79, n=10000)",
              0.109e6, 97.35, color="#555555", fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("MACs / 张（log 尺度）")
ax.set_ylabel("Gazelle 真机 acc（%）")
ax.set_title("真机上板实验 Pareto：MACs vs 实测精度\n（全部 compass_sdk 真机实测；J1 系列为 1000 样本同窗口径）")
ax.set_ylim(80, 103)
ax.grid(True, which="both", alpha=0.25)

fig.tight_layout()
_place_all()
out = __file__.replace(".py", "").replace("plot_pareto_hw", "pareto_hw_acc") + ".png"
fig.savefig(out, dpi=160)
print(f"saved: {out}")
print("EuroSAT Pareto 前沿：")
for p in frontier:
    print(f"  {p[1]/1e6:8.3f}M MACs  {p[2]:6.2f}%  (n={p[3]})  {p[0]}")
