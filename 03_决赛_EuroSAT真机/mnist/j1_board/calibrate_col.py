# calibrate_col.py — 逐列 (per-output-channel) 校准, 替代/补充 per-layer 标量校准
#
# 背景: probe 残差分解发现标量 calib (np.polyfit 压平列维度) 之后, 残差里仍有
# 确定性的 per-column 偏移 (std 264-872 raw) 与 per-column 增益 (std 1.2-2.5%)。
# 本脚本对每层每列 c 做一维最小二乘 hw[:,c] ~ alpha_c*ideal[:,c] + beta_c,
# 输出 calib_col.json, 由 run_j1_gazelle.py (J1_CALIB_COL) 折叠进反量化使用。
#
# 环境变量 (compass_sdk 篡改 sys.argv, 禁位置参数):
#   CALIB_COL_PAIRS_DIR  probe pairs 目录 (default /home/uisrc/j1)
#   CALIB_COL_OUT        输出 json (default {PAIRS_DIR}/calib_col.json)
#   CALIB_COL_LAYERS     逗号分隔层名 (default s1a,s2a,s2b,s3a,s3b; 有 pairs 才处理)
#   CALIB_COL_SCALAR     现有标量 calib json (对照用, 可选)
#   CALIB_COL_PREFIX     pairs 文件前缀 (default probe_)
#   CALIB_COL_HOLDOUT=1  留出验证自检 (default 1): 行 50/50 分半,
#                        前半拟合, 后半对比 标量 vs 逐列 残差 std
#
# 本地预检 (无需上板):
#   CALIB_COL_PAIRS_DIR=<pairs目录> CALIB_COL_OUT=calib_col_local.json python3 calibrate_col.py
import os
import json

import numpy as np

PAIRS_DIR = os.environ.get("CALIB_COL_PAIRS_DIR", "/home/uisrc/j1")
OUT = os.environ.get("CALIB_COL_OUT", os.path.join(PAIRS_DIR, "calib_col.json"))
LAYERS = os.environ.get("CALIB_COL_LAYERS", "s1a,s2a,s2b,s3a,s3b").split(",")
SCALAR_FILE = os.environ.get("CALIB_COL_SCALAR", "")
PREFIX = os.environ.get("CALIB_COL_PREFIX", "probe_")
HOLDOUT = os.environ.get("CALIB_COL_HOLDOUT", "1") == "1"


def fit_col(x, y):
    """一维最小二乘 y ~ a*x + b。返回 (a, b, resid_std, se_a, se_b)。x,y: (n,)。"""
    n = x.shape[0]
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    sxx = float((dx * dx).sum())
    if sxx <= 0:
        return 1.0, float(ym), float(y.std()), 0.0, 0.0
    a = float((dx * (y - ym)).sum() / sxx)
    b = float(ym - a * xm)
    r = y - (a * x + b)
    s2 = float((r * r).sum()) / max(n - 2, 1)
    se_a = np.sqrt(s2 / sxx)
    se_b = np.sqrt(s2 * (1.0 / n + xm * xm / sxx))
    return a, b, float(r.std()), float(se_a), float(se_b)


def eval_resid_std(ideal, hw, alphas, betas):
    """用给定 (alpha, beta) (标量或 (m,) 向量) 求残差 std。"""
    r = hw - (ideal * np.asarray(alphas, dtype=np.float64) + np.asarray(betas, dtype=np.float64))
    return float(r.std())


def main():
    scalar_ref = {}
    if SCALAR_FILE and os.path.exists(SCALAR_FILE):
        scalar_ref = json.load(open(SCALAR_FILE))
        print(f"scalar 对照: {SCALAR_FILE}")

    calib = {}
    summary_rows = []
    for name in LAYERS:
        fi = os.path.join(PAIRS_DIR, f"{PREFIX}{name}_ideal.npy")
        fh = os.path.join(PAIRS_DIR, f"{PREFIX}{name}_hw.npy")
        if not (os.path.exists(fi) and os.path.exists(fh)):
            print(f"{name}: pairs 缺失, 跳过")
            continue
        ideal = np.load(fi).astype(np.float64)
        hw = np.load(fh).astype(np.float64)
        n, m = ideal.shape

        # ---- 全量拟合 ----
        sa, sb = np.polyfit(ideal.ravel(), hw.ravel(), 1)
        alphas = np.zeros(m)
        betas = np.zeros(m)
        col_std = np.zeros(m)
        se_a = np.zeros(m)
        se_b = np.zeros(m)
        for c in range(m):
            alphas[c], betas[c], col_std[c], se_a[c], se_b[c] = \
                fit_col(ideal[:, c], hw[:, c])

        entry = {
            "alpha": [round(float(v), 8) for v in alphas],
            "beta": [round(float(v), 4) for v in betas],
            "col_resid_std": [round(float(v), 2) for v in col_std],
            "scalar_alpha": float(sa),
            "scalar_beta": float(sb),
            "n_samples": int(n),
            "se_alpha_median": float(np.median(se_a)),
            "se_beta_median": float(np.median(se_b)),
        }
        if name in scalar_ref:
            entry["scalar_ref_alpha"] = scalar_ref[name].get("alpha")
            entry["scalar_ref_beta"] = scalar_ref[name].get("beta")

        # 列结构真实性: per-col 参数跨列离散度 vs 估计标准误 (应 >>1 才是真结构)
        snr_a = float(alphas.std() / max(np.median(se_a), 1e-12))
        snr_b = float(betas.std() / max(np.median(se_b), 1e-12))
        entry["col_struct_snr_alpha"] = round(snr_a, 1)
        entry["col_struct_snr_beta"] = round(snr_b, 1)

        # ---- 留出验证: 前半拟合, 后半评估 ----
        if HOLDOUT:
            rng = np.random.RandomState(0)
            perm = rng.permutation(n)
            h1, h2 = perm[: n // 2], perm[n // 2:]
            ha, hb = np.polyfit(ideal[h1].ravel(), hw[h1].ravel(), 1)
            fa = np.zeros(m)
            fb = np.zeros(m)
            for c in range(m):
                fa[c], fb[c], _, _, _ = fit_col(ideal[h1, c], hw[h1, c])
            std_scalar = eval_resid_std(ideal[h2], hw[h2], ha, hb)
            std_col = eval_resid_std(ideal[h2], hw[h2], fa, fb)
            red_std = 100.0 * (1.0 - std_col / std_scalar)
            red_var = 100.0 * (1.0 - (std_col / std_scalar) ** 2)
            entry["holdout"] = {
                "n_fit": int(h1.shape[0]),
                "n_eval": int(h2.shape[0]),
                "scalar_resid_std": round(std_scalar, 2),
                "col_resid_std": round(std_col, 2),
                "std_reduction_pct": round(red_std, 2),
                "var_reduction_pct": round(red_var, 2),
            }
            print(f"{name}: n={n} cols={m} | scalar {std_scalar:.1f} -> col {std_col:.1f} "
                  f"(std -{red_std:.1f}%, var -{red_var:.1f}%) | "
                  f"alpha_c std={alphas.std():.5f} (SE {np.median(se_a):.2e}, SNR {snr_a:.0f}) "
                  f"beta_c std={betas.std():.1f} (SE {np.median(se_b):.1f}, SNR {snr_b:.0f})",
                  flush=True)
            summary_rows.append((name, std_scalar, std_col, red_std, red_var))
        else:
            print(f"{name}: n={n} cols={m} alpha_c std={alphas.std():.5f} "
                  f"beta_c std={betas.std():.1f} resid_std(mean col)={col_std.mean():.1f}",
                  flush=True)

        calib[name] = entry

    json.dump(calib, open(OUT, "w"), indent=2)
    print("saved", OUT)
    if summary_rows:
        print("\n==== 留出验证汇总 (标量 -> 逐列, 后半数据) ====")
        for name, ss, cs, rs, rv in summary_rows:
            print(f"  {name:5s}  {ss:9.1f} -> {cs:9.1f}   std -{rs:5.1f}%  var -{rv:5.1f}%")


if __name__ == "__main__":
    main()
