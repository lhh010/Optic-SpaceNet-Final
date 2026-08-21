# -*- coding: utf-8 -*-
"""Perf (test acc) vs MACs — Ltsimulator-test + auto_research R1/R2/R3/R6/R7/R8 + M5-M10 v8 + 真机全量."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# (label, MACs in millions, acc %, group, annotation offset[, fontsize])
# 第 6 项可选 fontsize: 消融臂小字号 6.5, 主标签 8
points = [
    # Ltsimulator-test 主工程
    ("Model 1 (int8 test)",        156.6, 97.89, "Ltsimulator-test", (8, 3)),
    ("Model 2 (int8 test)",          1.05, 92.20, "Ltsimulator-test", (12, -5)),
    ("Model 3 (int8+KD val)",        1.05, 91.83, "Ltsimulator-test", (-28, -11)),
    # auto_research R1: Model 4 E, 17.04M, QAT v5 各变体 (test 5400); 5 个未命名变体归并标注
    ("R1 Model4E v5+SGD",           17.04, 97.43, "R1 (Model 4 E)", (6, 8)),
    ("R1 v5 变体 ×5 (95.35–97.22)", 17.04, 96.74, "R1 (Model 4 E)", (-64, -11), 6.5),
    # auto_research R2: 架构搜索 (test 5400); 臂名自 round2/round3 notes
    ("R2 J1 (80ep)",                 1.38, 95.52, "R2 架构搜索", (12, -10)),
    ("J1_head",                      1.44, 95.94, "R2 架构搜索", (16, -3), 6.5),
    ("H1",                           1.21, 94.89, "R2 架构搜索", (-16, -7), 6.5),
    ("G2",                           1.19, 94.11, "R2 架构搜索", (0, 7), 6.5),
    ("I1",                           1.14, 93.85, "R2 架构搜索", (-8, -7), 6.5),
    ("G1",                           1.41, 93.83, "R2 架构搜索", (10, -6), 6.5),
    ("R2 G3x (更宽反例)",             3.08, 93.70, "R2 架构搜索", (6, -6)),
    # auto_research R3: J1 精调 (test 5400)
    ("R3 J1_long (160ep)",           1.38, 96.30, "R3 J1 精调", (-64, 10)),
    # auto_research R6: 消融臂 (80ep, test 5400; pool_s1x1 85.07 超出坐标未画)
    ("w075 80ep",                    0.86, 95.30, "R6 消融 (80ep)", (8, -3), 6.5),
    ("pool_patch_n",                 1.30, 95.09, "R6 消融 (80ep)", (12, -4), 6.5),
    ("pool_avg",                     1.38, 94.78, "R6 消融 (80ep)", (-28, -11), 6.5),
    ("glb64",                        1.40, 95.63, "R6 消融 (80ep)", (14, 4), 6.5),
    ("glb128",                       1.43, 95.39, "R6 消融 (80ep)", (8, -3), 6.5),
    ("d133 80ep",                    1.90, 95.67, "R6 消融 (80ep)", (8, -3), 6.5),
    ("pool_patch",                   2.16, 95.33, "R6 消融 (80ep)", (-28, -11), 6.5),
    ("R6 rf_stem5",                  2.16, 96.04, "R6 消融 (80ep)", (-12, -16)),
    ("d244",                         2.69, 95.46, "R6 消融 (80ep)", (8, 3), 6.5),
    ("R6 w150",                      2.77, 96.06, "R6 消融 (80ep)", (8, 3)),
    ("R6 rf_s2k3 (80ep)",            4.52, 96.44, "R6 消融 (80ep)", (-64, -16)),
    ("R6 w200 (80ep)",               4.62, 96.54, "R6 消融 (80ep)", (-64, 5)),
    # auto_research R7/R8: 160ep 决赛 (test 5400)
    ("R8 w075 (160ep)",              0.86, 95.56, "R7/R8 (160ep)", (-12, 10)),
    ("R7 head256",                   1.40, 96.04, "R7/R8 (160ep)", (-64, 5)),
    ("R7 J1 s43",                    1.38, 96.24, "R7/R8 (160ep)", (-40, -10)),
    ("w110",                         1.68, 96.02, "R7/R8 (160ep)", (-8, 7), 6.5),
    ("stem5_c12",                    1.82, 95.76, "R7/R8 (160ep)", (-30, -12), 6.5),
    ("d133 160ep",                   1.90, 95.96, "R7/R8 (160ep)", (14, 12), 6.5),
    ("R8 rf_s2k3 (160ep)",           4.52, 96.93, "R7/R8 (160ep)", (-44, 7)),
    ("R8 w200 (160ep)",              4.62, 96.39, "R7/R8 (160ep)", (12, -24)),
    # M5-M10 (决赛模型, v8 漂移鲁棒 QAT, test 5400; M9/M10 画部署 seed42 值, 双 seed 为 95.87/95.69 与 96.76/96.56, 2026-08-17 收官)
    ("M7 v8",                       0.86, 95.00, "M5-M10 (v8)", (8, -14)),
    ("M6 v8",                       1.38, 95.22, "M5-M10 (v8)", (12, -10)),
    ("M9 v8 ★",                     1.52, 95.87, "M5-M10 (v8)", (-40, -5)),
    ("M8 v8",                       2.16, 96.26, "M5-M10 (v8)", (-22, 10)),
    ("M10 v8 ★",                    2.56, 96.76, "M5-M10 (v8)", (8, 10)),
    ("M5 v8",                       5.31, 96.65, "M5-M10 (v8)", (8, 2)),
]

# 真机全量 5400 口径（2026-08-17 收官，绿色星形）
# 第 5 项可选 dict: ha/va 控制文字对齐, 使文字起点紧贴自身星
hw_points = [
    ("M7 hw",  0.86, 81.72, (8, -16)),                       # 板端 [0:1800], 低于 ylim 未画
    ("M6 hw",  1.38, 92.78, (0, -11)),
    ("M9 hw ★", 1.52, 94.43, (28, 3), {"ha": "center", "va": "top"}),
    ("M8 hw",  2.16, 87.78, (8, 8)),                          # 低于 ylim 未画
    ("M10 hw ★", 2.56, 95.33, (8, -11), {"ha": "center", "va": "top"}),
    ("M5 hw",  5.31, 90.17, (8, 8)),                          # 低于 ylim 未画
    ("M4 hw", 17.03, 94.19, (-16, -11)),
]

groups = {
    "Ltsimulator-test":  dict(color="#d62728", marker="s", s=70),
    "R1 (Model 4 E)":    dict(color="#1f77b4", marker="o", s=55),
    "R2 架构搜索":        dict(color="#2ca02c", marker="^", s=60),
    "R3 J1 精调":         dict(color="#9467bd", marker="*", s=140),
    "R6 消融 (80ep)":     dict(color="#8c564b", marker="v", s=45),
    "R7/R8 (160ep)":     dict(color="#ff7f0e", marker="D", s=50),
    "M5-M10 (v8)":      dict(color="#d62728", marker="X", s=110),
    "真机全量 (5400)":   dict(color="#2ca02c", marker="*"),
}

fig, ax = plt.subplots(figsize=(14, 5.6))  # 宽幅版: 适配 PPT 预览页右侧 ~2.5:1 区域

for gname, style in groups.items():
    xs = [p[1] for p in points if p[3] == gname and p[2] is not None]
    ys = [p[2] for p in points if p[3] == gname and p[2] is not None]
    ax.scatter(xs, ys, label=gname, zorder=3, edgecolors="white",
               linewidths=0.6, **style)

for p in points:
    label, x, y, gname, off = p[0], p[1], p[2], p[3], p[4]
    fs = p[5] if len(p) > 5 else 8
    if label is None or y is None:
        continue
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off,
                fontsize=fs, color=groups[gname]["color"], zorder=4)

# 真机全量 5400 点（绿色星形；M7 81.72 / M8 87.78 / M5 hw 90.17 低于 ylim 未画）
hw_show = [p for p in hw_points if p[2] >= 91]
ax.scatter([p[1] for p in hw_show], [p[2] for p in hw_show],
           color=groups["真机全量 (5400)"]["color"], marker="*", s=240,
           edgecolors="white", linewidths=0.6, zorder=5,
           label="真机全量 5400 (2026-08-17 收官)")
for p in hw_points:
    label, x, y, off = p[0], p[1], p[2], p[3]
    extra = p[4] if len(p) > 4 else {}
    if y < 91:
        continue
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off,
                fontsize=8, color=groups["真机全量 (5400)"]["color"], zorder=6,
                ha=extra.get("ha", "left"), va=extra.get("va", "baseline"))

# ---- 仅保留数据点 + 点旁文字标注（无箭头/callout，2026-08-20 用户指示）----

# 2M 预算线 + Pareto 前沿 (160ep 口径; Model 2 被 w075 严格支配, 不在前沿上)
ax.axvspan(0.8, 2.0, color="gray", alpha=0.10, zorder=1)
ax.axvline(2.0, color="gray", ls="--", lw=1, alpha=0.6)
ax.text(2.12, 91.1, "2M预算线", fontsize=8.5, color="gray")
pareto = [(0.86, 95.56), (1.38, 96.30), (4.52, 96.93), (17.04, 97.43), (156.6, 97.89)]
ax.plot([p[0] for p in pareto], [p[1] for p in pareto], "k--", lw=1.2,
        alpha=0.5, zorder=2, label="Pareto 前沿 (160ep/最终口径)")
# 真机全量口径帕累托前沿 (2026-08-17 收官): M6 92.78 -> M9 94.43 -> M10 95.33
# (M4 94.19 被 M10 支配、M5/M7/M8 低于 ylim, 均不在前沿上)
hw_pareto = [(1.38, 92.78), (1.52, 94.43), (2.56, 95.33)]
ax.plot([p[0] for p in hw_pareto], [p[1] for p in hw_pareto],
        color="#2ca02c", ls="-.", lw=1.6, alpha=0.85, zorder=2,
        label="真机 Pareto 前沿 (全量 5400)")

ax.set_xscale("log")
ax.set_xlabel("激活计算量 MACs / 张 (log scale)")
ax.set_ylabel("test 精度 (%) — QAT clean (R1-R8) / v8 (M5-M10) / 真机全量 (★)")
ax.set_title("Perf vs MACs — EuroSAT test 5400（2026-08-17 收官口径；R6 pool_s1x1 85.07 / M7 81.72 / M8 87.78 低于范围未画）")
ax.set_xlim(0.8, 400)
ax.set_ylim(91, 99.3)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

fig.tight_layout()
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160)
print("saved:", out)
