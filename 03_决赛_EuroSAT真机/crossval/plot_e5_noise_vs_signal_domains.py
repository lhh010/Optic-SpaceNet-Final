# -*- coding: utf-8 -*-
"""E5 三值域 noise-vs-signal 重绘（2026-08-20 v2, 据 fitted_params.json + stats.json）

原图 e5_noise_vs_signal_domains.png（43c2736 二进制入库，无脚本）存在文字贴线、
数值标签压标记点问题；本脚本重绘并以渲染级 bbox 实测约束布局：
- hw = 绝对噪声底（平坦 2.8-4.5 counts），sim = 信号相关（跨值域 ~300x）
- 数值标签/说明文字均经 check 脚本实测无碰撞
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fp = json.load(open("fitted_params.json", encoding="utf-8"))
st = json.load(open("stats.json", encoding="utf-8"))["E5"]

regimes = ["uint4", "uint8", "uint4x16"]
x = [fp["regimes"][r]["rms_ideal"] / 255.0 for r in regimes]          # signal rms (counts)
hw_y = [fp["regimes"][r]["sigma_total_counts"] for r in regimes]      # hw sigma (counts)
sim_y = [st[r + "_sim"]["std"] / 255.0 for r in regimes]              # sim sigma (counts)
hw_lab = ["2.79", "4.49", "3.85"]
sim_lab = ["0.004", "1.40", "1.05"]

fig, ax = plt.subplots(figsize=(9.2, 5.8))
fig.subplots_adjust(left=0.115, right=0.72, top=0.885, bottom=0.16)

ax.plot(x, hw_y, "-", color="#d62728", lw=1.8, marker="s", ms=9,
        zorder=4, label="Gazelle hw（E5, $\\sigma_{total}$）")
ax.plot(x, sim_y, "--", color="#1f77b4", lw=1.8, marker="o", ms=8,
        zorder=4, label="osimulator sim（E5, $\\sigma$）")

# 数值标签（偏移含 ha：4.49/3.85 两标记在 log 轴上仅 ~2px，标签左右分置）
hw_off = [((0, 12), "center"), ((-11, 4), "right"), ((11, 4), "left")]
sim_off = [((0, -16), "center"), ((0, 11), "center"), ((0, -17), "center")]
lab_arts = []
for xi, yi, lab, (off, ha) in zip(x, hw_y, hw_lab, hw_off):
    lab_arts.append(ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=off,
                                fontsize=9, fontweight="bold", color="#d62728", zorder=6,
                                ha=ha))
for xi, yi, lab, (off, ha) in zip(x, sim_y, sim_lab, sim_off):
    lab_arts.append(ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=off,
                                fontsize=9, fontweight="bold", color="#1f77b4", zorder=6,
                                ha=ha))

# 说明文字（锚定数据坐标；QA 实测后可调）
t_hw = ax.text(1.8, 5.0, "hw：绝对噪声底（平坦，2.8–4.5 counts）" + chr(10) + "与信号幅度无关",
               fontsize=9.5, color="#d62728", zorder=5, va="bottom", ha="left")
t_sim = ax.text(1.35, 0.096, "sim：噪声 ∝ 信号幅度" + chr(10) + "（跨值域 ~300×，0.004 → 1.40 counts）",
                fontsize=9.5, color="#1f77b4", zorder=5)
t_warn = ax.text(0.60, 0.55, "uint4 信号低于噪声底" + chr(10) + "（hw 拟合无物理意义）",
                 fontsize=8.5, color="#8a6d00", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec="#ccb350", alpha=0.9))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.3, 320)
ax.set_ylim(0.002, 12)
ax.set_xlabel("信号幅度 rms_ideal（counts，log 刻度）", fontsize=10)
ax.set_ylabel("噪声 std（counts，log 刻度）", fontsize=10)
ax.set_title("E5：hw 绝对噪声底 vs sim 信号相关噪声（3 值域）", fontsize=12.5)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.text(0.5, 0.022,
         "据 crossval/fitted_params.json + stats.json E5（3 值域 × 10000 GEMM/侧；counts = raw/255）。"
         "口径与 CROSSVAL_REPORT.md §4/§5.2 一致。",
         ha="center", fontsize=8.5, color="#666666")

fig.canvas.draw()
fig.savefig("figures/e5_noise_vs_signal_domains.png", dpi=160)
print("saved: figures/e5_noise_vs_signal_domains.png")
