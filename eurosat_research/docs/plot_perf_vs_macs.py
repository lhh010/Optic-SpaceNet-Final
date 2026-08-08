# -*- coding: utf-8 -*-
"""Perf (QAT clean acc) vs MACs — Ltsimulator-test + auto_research R1/R2/R3/R6/R7/R8."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# (label, MACs in millions, QAT clean acc %, group, annotation offset)
# label=None 表示不标注
points = [
    # Ltsimulator-test 主工程
    ("Model 1 (int8 test)",        156.6, 97.89, "Ltsimulator-test", (6, 6)),
    ("Model 2 (int8 test)",          1.05, 92.20, "Ltsimulator-test", (-98, -4)),
    ("Model 3 (int8+KD val)",        1.05, 91.83, "Ltsimulator-test", (6, -6)),
    # auto_research R1: Model 4 E, 17.04M, QAT v5 各变体 (test 5400)
    ("R1 Model4E v5+SGD",           17.04, 97.43, "R1 (Model 4 E)", (6, 8)),
    (None,                          17.04, 97.22, "R1 (Model 4 E)", (0, 0)),
    (None,                          17.04, 96.93, "R1 (Model 4 E)", (0, 0)),
    (None,                          17.04, 96.74, "R1 (Model 4 E)", (0, 0)),
    (None,                          17.04, 96.81, "R1 (Model 4 E)", (0, 0)),
    (None,                          17.04, 95.35, "R1 (Model 4 E)", (0, 0)),
    # auto_research R2: 架构搜索 (test 5400)
    ("R2 J1 (80ep)",                 1.38, 95.52, "R2 架构搜索", (8, 4)),
    (None,                           1.44, 95.94, "R2 架构搜索", (0, 0)),
    (None,                           1.21, 94.89, "R2 架构搜索", (0, 0)),
    (None,                           1.19, 94.11, "R2 架构搜索", (0, 0)),
    (None,                           1.14, 93.85, "R2 架构搜索", (0, 0)),
    (None,                           1.41, 93.83, "R2 架构搜索", (0, 0)),
    ("R2 G3x (更宽反例)",             3.08, 93.70, "R2 架构搜索", (6, -6)),
    # auto_research R3: J1 精调 (test 5400)
    ("R3 J1_long (160ep)",           1.38, 96.30, "R3 J1 精调", (6, 8)),
    # auto_research R6: 消融臂 (80ep, test 5400; pool_s1x1 85.07 超出坐标未画)
    (None,                           0.86, 95.30, "R6 消融 (80ep)", (0, 0)),   # w075
    (None,                           1.30, 95.09, "R6 消融 (80ep)", (0, 0)),   # pool_patch_n
    (None,                           1.38, 94.78, "R6 消融 (80ep)", (0, 0)),   # pool_avg
    (None,                           1.40, 95.63, "R6 消融 (80ep)", (0, 0)),   # glb64
    (None,                           1.43, 95.39, "R6 消融 (80ep)", (0, 0)),   # glb128
    (None,                           1.90, 95.67, "R6 消融 (80ep)", (0, 0)),   # d133
    (None,                           2.16, 95.33, "R6 消融 (80ep)", (0, 0)),   # pool_patch
    ("R6 rf_stem5",                  2.16, 96.04, "R6 消融 (80ep)", (-72, 6)),
    (None,                           2.69, 95.46, "R6 消融 (80ep)", (0, 0)),   # d244
    ("R6 w150",                      2.77, 96.06, "R6 消融 (80ep)", (-48, -12)),
    ("R6 rf_s2k3 (80ep)",            4.52, 96.44, "R6 消融 (80ep)", (-40, -14)),
    ("R6 w200 (80ep)",               4.62, 96.54, "R6 消融 (80ep)", (8, -4)),
    # auto_research R7/R8: 160ep 决赛 (test 5400)
    ("R8 w075 (160ep)",              0.86, 95.56, "R7/R8 (160ep)", (8, -10)),
    ("R7 head256",                   1.40, 96.04, "R7/R8 (160ep)", (-12, -14)),  # 2 seeds mean
    ("R7 J1 s43",                    1.38, 96.24, "R7/R8 (160ep)", (-70, -2)),
    (None,                           1.68, 96.02, "R7/R8 (160ep)", (0, 0)),   # w110
    (None,                           1.82, 95.76, "R7/R8 (160ep)", (0, 0)),   # stem5_c12
    (None,                           1.90, 95.96, "R7/R8 (160ep)", (0, 0)),   # d133
    ("R8 rf_s2k3 (160ep)",           4.52, 96.93, "R7/R8 (160ep)", (-60, 8)),
    ("R8 w200 (160ep)",              4.62, 96.39, "R7/R8 (160ep)", (8, -8)),
]

groups = {
    "Ltsimulator-test":  dict(color="#d62728", marker="s", s=70),
    "R1 (Model 4 E)":    dict(color="#1f77b4", marker="o", s=55),
    "R2 架构搜索":        dict(color="#2ca02c", marker="^", s=60),
    "R3 J1 精调":         dict(color="#9467bd", marker="*", s=140),
    "R6 消融 (80ep)":     dict(color="#8c564b", marker="v", s=45),
    "R7/R8 (160ep)":     dict(color="#ff7f0e", marker="D", s=50),
}

fig, ax = plt.subplots(figsize=(11, 7))

for gname, style in groups.items():
    xs = [p[1] for p in points if p[3] == gname and p[2] is not None]
    ys = [p[2] for p in points if p[3] == gname and p[2] is not None]
    ax.scatter(xs, ys, label=gname, zorder=3, edgecolors="white",
               linewidths=0.6, **style)

for label, x, y, gname, off in points:
    if label is None or y is None:
        continue
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off,
                fontsize=8, color=groups[gname]["color"], zorder=4)

# 2M 严格预算线 + Pareto 前沿 (160ep 口径; Model 2 被 w075 严格支配, 不在前沿上)
ax.axvline(2.0, color="gray", ls="--", lw=1, alpha=0.6)
ax.annotate("≤2M 严格预算", (2.05, 91.4), fontsize=8, color="gray")
pareto = [(0.86, 95.56), (1.38, 96.30), (4.52, 96.93), (17.04, 97.43), (156.6, 97.89)]
ax.plot([p[0] for p in pareto], [p[1] for p in pareto], "k--", lw=1.2,
        alpha=0.5, zorder=2, label="Pareto 前沿 (160ep/最终口径)")

ax.set_xscale("log")
ax.set_xlabel("激活计算量 MACs / 张 (log scale)")
ax.set_ylabel("QAT clean 精度 (%)")
ax.set_title("Perf vs MACs — QAT clean 口径\nLtsimulator-test + auto_research R1/R2/R3/R6/R7/R8 (EuroSAT)\n"
             "(R6 pool_s1x1 = 85.07% 超出坐标范围未画)")
ax.set_xlim(0.8, 400)
ax.set_ylim(91, 99)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

fig.tight_layout()
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160)
print("saved:", out)
