#!/usr/bin/env python3
"""A3: 权重分布 x per-column 残差相关性分析

检验预测: dW 是绝对扰动(int8 counts) -> 列内小权重承受更大相对扰动 ->
重尾权重分布的列残差更大(相对口径) -> wd 压权重向零在噪声口径下更有害。

输出:
  - 每层全局 hw ~ alpha*ideal+beta 拟合 + per-column 残差强度
  - per-column 权重分布统计 (kurtosis / frac|w|<10 / CV / rms / entropy)
  - 跨列 Spearman/Pearson 相关 (~416 列)
  - dW 矩阵回归 (R[:,j] ~ x @ dW[:,j]) 并检验 |dW| vs |w| (绝对 vs 相对扰动)
结果打印到 stdout, 同时写 x0/results/a3_weight_dist.json
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats  # scipy 不可用则退回手写 spearman

ROOT = Path(__file__).resolve().parents[2]  # eurosat_research/
PAIRS = ROOT / "x0/data/probe_pairs"
DEPLOY = ROOT / "runs/c3d_J1_v8probe15_local/deploy_weights"
OUT = ROOT / "x0/results/a3_weight_dist.json"

LAYERS = ["s1a", "s2a", "s2b", "s3a", "s3b"]


def spearmanr(a, b):
    return stats.spearmanr(a, b)


def pearsonr(a, b):
    return stats.pearsonr(a, b)


def col_weight_stats(w):
    """w: (n_out, k) int8 -> dict of per-column (per output-channel) stats."""
    wa = np.abs(w.astype(np.float64))
    n = w.shape[0]
    out = {}
    out["rms"] = np.sqrt((wa**2).mean(axis=1))
    out["maxabs"] = wa.max(axis=1)
    out["meanabs"] = wa.mean(axis=1)
    out["cv_abs"] = wa.std(axis=1) / (wa.mean(axis=1) + 1e-9)
    out["frac_small10"] = (wa < 10).mean(axis=1)
    out["frac_zero"] = (wa == 0).mean(axis=1)
    # excess kurtosis (Fisher)
    m = wa.mean(axis=1, keepdims=True)
    s = wa.std(axis=1, keepdims=True) + 1e-9
    out["kurt_abs"] = (((wa - m) / s) ** 4).mean(axis=1) - 3.0
    # entropy of |w| histogram (16 bins over [0,128])
    ent = np.zeros(n)
    for j in range(n):
        h, _ = np.histogram(wa[j], bins=16, range=(0, 128))
        p = h / h.sum()
        p = p[p > 0]
        ent[j] = -(p * np.log2(p)).sum()
    out["entropy_abs"] = ent
    return out


def fit_layer(x, ideal, hw):
    """全局 hw ~ alpha*ideal + beta, 返回 alpha, beta, R (n,m)."""
    A = np.vstack([ideal.ravel(), np.ones(ideal.size)]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, hw.ravel(), rcond=None)
    R = hw - (alpha * ideal + beta)
    return alpha, beta, R


def regress_dW(x, R):
    """对每列 j: R[:,j] ~ x @ dW[:,j] + c. 返回 dW (k, m), 以及估计噪声尺度.

    x: (n, k) uint8. 中心化后最小二乘. 同时给出每个系数的标准误估计.
    """
    n, k = x.shape
    m = R.shape[1]
    xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    Rc = R - R.mean(axis=0, keepdims=True)
    # (k,k) 可能病态, 加微小 ridge
    G = xc.T @ xc + 1e-6 * np.trace(xc.T @ xc) / k * np.eye(k)
    dW = np.linalg.solve(G, xc.T @ Rc)  # (k, m)
    Rhat = Rc - xc @ dW
    dof = max(n - k - 1, 1)
    sigma2 = (Rhat**2).sum(axis=0) / dof  # per column residual variance
    # 系数标准误: se_ik = sqrt(sigma2_j * Ginv_ii)
    Ginv_diag = np.diag(np.linalg.inv(G))
    se = np.sqrt(np.outer(Ginv_diag, sigma2))  # (k, m)
    return dW, se, Rhat


def main():
    all_col_stats = []   # 跨层列级记录
    dW_records = {}      # 每层 dW 分析
    layer_summary = {}

    for layer in LAYERS:
        x = np.load(PAIRS / f"probe_{layer}_xint.npy")
        ideal = np.load(PAIRS / f"probe_{layer}_ideal.npy").astype(np.float64)
        hw = np.load(PAIRS / f"probe_{layer}_hw.npy").astype(np.float64)
        w = np.load(DEPLOY / f"{layer}_w.npy")  # (n_out, k) int8
        assert w.shape == (ideal.shape[1], x.shape[1]), (layer, w.shape, ideal.shape, x.shape)

        alpha, beta, R = fit_layer(x, ideal, hw)
        n, m = R.shape
        col_off = R.mean(axis=0)
        col_std_raw = R.std(axis=0)
        col_std_in = (R - col_off).std(axis=0)
        sig_rms = np.sqrt((ideal**2).mean(axis=0))  # 每列信号强度
        rel_std = col_std_in / (sig_rms + 1e-9)

        wstats = col_weight_stats(w)

        # ---- dW 回归 (x 域, 单位换算到 int8 counts: dW_counts = dW_est/alpha) ----
        dW_est, se, Rhat = regress_dW(x, R)
        dW_counts = dW_est / alpha
        se_counts = se / abs(alpha)

        layer_summary[layer] = dict(
            n=n, m=m, k=x.shape[1], alpha=float(alpha), beta=float(beta),
            resid_std=float(R.std()),
            resid_std_within_col=float(col_std_in.mean()),
            col_off_std=float(col_off.std()),
            dW_rms=float(np.sqrt((dW_counts**2).mean())),
            dW_se_median=float(np.median(se_counts)),
            dW_hat_rms=float(np.sqrt((Rhat**2).mean())),
            x_std=float(x.std()),
        )

        # ---- dW vs |w| 逐元素关系 ----
        wa = np.abs(w.astype(np.float64)).ravel()
        da = np.abs(dW_counts.T.ravel())  # dW_counts is (k, m) -> transpose to (m, k) match w
        # w is (m, k): w.ravel() 顺序是 j 主序; dW_counts.T 也是 (m,k). OK
        sp_all = spearmanr(wa, da)
        pe_all = pearsonr(wa, da)
        # 按 |w| 分 bin 看 rms(dW)
        bins = [0, 2, 5, 10, 20, 40, 80, 128]
        bin_rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (wa >= lo) & (wa < hi)
            if mask.sum() < 5:
                continue
            bin_rows.append(dict(
                bin=f"[{lo},{hi})", n=int(mask.sum()),
                w_med=float(np.median(wa[mask])),
                dW_rms=float(np.sqrt((da[mask]**2).mean())),
                dW_mean=float(da[mask].mean()),
            ))
        # 减去估计噪声后的 dW 功率 (per bin)
        se_flat = se_counts.T.ravel()
        for row in bin_rows:
            lo, hi = row["bin"].strip("[])").split(",")
            mask = (wa >= float(lo)) & (wa < float(hi))
            row["dW_rms_denoised"] = float(np.sqrt(max((da[mask]**2 - se_flat[mask]**2).mean(), 0.0)))

        dW_records[layer] = dict(
            spearman=dict(rho=float(sp_all.statistic), p=float(sp_all.pvalue)),
            pearson=dict(r=float(pe_all.statistic), p=float(pe_all.pvalue)),
            bins=bin_rows,
        )

        for j in range(m):
            rec = dict(layer=layer, col=j,
                       col_std_in=float(col_std_in[j]),
                       col_std_raw=float(col_std_raw[j]),
                       rel_std=float(rel_std[j]),
                       sig_rms=float(sig_rms[j]),
                       off=float(col_off[j]))
            for kname, arr in wstats.items():
                rec[kname] = float(arr[j])
            all_col_stats.append(rec)

    # ================= 跨列相关性 =================
    metrics = ["rms", "maxabs", "meanabs", "cv_abs", "frac_small10", "frac_zero",
               "kurt_abs", "entropy_abs"]
    targets = ["col_std_in", "col_std_raw", "rel_std"]
    corr_all = {}
    for t in targets:
        corr_all[t] = {}
        for met in metrics:
            a = np.array([r[met] for r in all_col_stats])
            b = np.array([r[t] for r in all_col_stats])
            sp = spearmanr(a, b)
            pe = pearsonr(a, b)
            corr_all[t][met] = dict(spearman=float(sp.statistic), sp_p=float(sp.pvalue),
                                    pearson=float(pe.statistic), pe_p=float(pe.pvalue))

    # 层内相关 (控制层间差异): 每层内 spearman, 再取中位
    corr_within = {}
    for t in targets:
        corr_within[t] = {}
        for met in metrics:
            rhos = []
            for layer in LAYERS:
                rows = [r for r in all_col_stats if r["layer"] == layer]
                a = np.array([r[met] for r in rows])
                b = np.array([r[t] for r in rows])
                if np.std(a) < 1e-12:
                    continue
                rhos.append(float(spearmanr(a, b).statistic))
            corr_within[t][met] = dict(median_rho=float(np.median(rhos)), rhos=rhos)

    # dW 汇总: 全部层拼接 (按每层 |w| bin 的 dW_rms_denoised 已存)
    # 全局 |w| bin (所有层元素拼接太贵, 用每层 bin 的 n 加权)
    pooled_bins = {}
    for layer in LAYERS:
        for row in dW_records[layer]["bins"]:
            pooled_bins.setdefault(row["bin"], []).append((row["dW_rms"], row["n"]))
    pooled = []
    for bname, lst in pooled_bins.items():
        num = sum(r * n for r, n in lst)
        den = sum(n for _, n in lst)
        pooled.append(dict(bin=bname, n=den, dW_rms_wavg=num / den))
    pooled.sort(key=lambda r: float(r["bin"].strip("[])").split(",")[0]))

    result = dict(
        layer_summary=layer_summary,
        dW_vs_w=dW_records,
        corr_pooled_columns=corr_all,
        corr_within_layer=corr_within,
        pooled_dW_bins=pooled,
        n_columns=len(all_col_stats),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---------- 打印摘要 ----------
    print("=== 层摘要 ===")
    for l in LAYERS:
        s = layer_summary[l]
        print(f"{l}: alpha={s['alpha']:.4f} resid_std={s['resid_std']:.2f} "
              f"within_col_std={s['resid_std_within_col']:.2f} off_std={s['col_off_std']:.2f} "
              f"dW_rms={s['dW_rms']:.3f}counts (se_med={s['dW_se_median']:.3f}) x_std={s['x_std']:.1f}")
    print("\n=== dW vs |w| (逐元素, |dW| 单位 int8 counts) ===")
    for l in LAYERS:
        d = dW_records[l]
        print(f"{l}: spearman={d['spearman']['rho']:+.3f} (p={d['spearman']['p']:.2e}) "
              f"pearson={d['pearson']['r']:+.3f}")
        for row in d["bins"]:
            print(f"    |w|{row['bin']:>8} n={row['n']:>6}  dW_rms={row['dW_rms']:.3f} "
                  f"(denoised {row['dW_rms_denoised']:.3f})")
    print("\n=== 池化 |w| bin (全层 n 加权) ===")
    for row in pooled:
        print(f"  |w|{row['bin']:>8} n={row['n']:>7}  dW_rms={row['dW_rms_wavg']:.3f}")
    print("\n=== 跨列相关 (全部 416 列) ===")
    for t in targets:
        print(f"-- target: {t}")
        for met in metrics:
            c = corr_all[t][met]
            cw = corr_within[t][met]
            print(f"   {met:>12}: spearman={c['spearman']:+.3f} (p={c['sp_p']:.1e}) "
                  f"层内中位rho={cw['median_rho']:+.3f}")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
