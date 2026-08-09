#!/usr/bin/env python3
"""A2: 非线性残余再建模 — 从 probe pairs 学 R_nl 的函数形式,评估可学性。

流水线(每层):
  1. 全局回归 hw ~ alpha*ideal + beta -> R
  2. 去 per-column 偏移(列均值) + per-column 增益(列内 R~ideal 线性)
  3. 去线性 δW 分量(列内 R ~ [x,1] 岭回归) -> R_nl
  4. R_nl 可学性评估(全部用留出集 R²,随机 80/20 按行切分):
     M0 条件均值查表: per-column x ideal 分箱 -> 均值表
     M1 RFF 基线: per-column, cos(w*ideal+phi) 随机特征 + 岭回归
     M2 MLP(ideal): per-column 1-D 输入小 MLP
     M3 MLP(ideal + x 摘要): ||x||^2, max(x), nnz(x), mean(x)
     M4 MLP(ideal + 全量 x): 输入驱动非线性的上限探测
     M5 全局共享 MLP(ideal + x 摘要 + 列 one-hot): 跨列共享是否可行
  5. iid 噪声估计: 最佳模型残余 std / 查表 bin 内 std
输出: 打印每层数字表 + JSON 汇总到 stdout(由调用方存档)。
"""
import json
import sys
import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

DATA = "/Users/ms.chen/Projects/2607-ciciec/Ltsimulator-test/eurosat_research/x0/data/probe_pairs"
LAYERS = ["s1a", "s2a", "s2b", "s3a", "s3b"]
RNG = np.random.RandomState(0)
N_BINS = 32
N_RFF = 64
MLP_MAX_ROWS = 60000  # per-column MLP 采样上限(控制 CPU 时间)


def ridge_fit_pred(Xtr, ytr, Xte, alpha=1.0):
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def r2(y, yhat):
    v = np.var(y)
    return 1.0 - np.mean((y - yhat) ** 2) / v if v > 0 else 0.0


def bin_ids(v, n_bins, lo=None, hi=None):
    if lo is None:
        lo, hi = v.min(), v.max()
    edges = np.linspace(lo, hi, n_bins + 1)
    return np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1), (lo, hi)


def x_summaries(x):
    xf = x.astype(np.float64)
    return np.stack(
        [xf.mean(1), (xf**2).mean(1), xf.max(1), (xf > 0).mean(1)], axis=1
    )


def eval_layer(L):
    x = np.load(f"{DATA}/probe_{L}_xint.npy").astype(np.float64)
    ideal = np.load(f"{DATA}/probe_{L}_ideal.npy").astype(np.float64)
    hw = np.load(f"{DATA}/probe_{L}_hw.npy").astype(np.float64)
    N, K = x.shape
    C = ideal.shape[1]
    print(f"\n===== {L}: N={N} K={K} C={C} =====")
    hw_int = np.allclose(hw, np.round(hw))
    print(f"hw integer-valued: {hw_int}")

    # --- step 1: global alpha/beta ---
    a, b = np.polyfit(ideal.ravel(), hw.ravel(), 1)
    R = hw - (a * ideal + b)
    var_R = R.var()
    print(f"global fit: alpha={a:.6f} beta={b:.2f} resid_std={R.std():.2f}")

    # --- step 2: per-column offset + gain (on train rows only) ---
    idx = RNG.permutation(N)
    ntr = int(0.8 * N)
    tr, te = idx[:ntr], idx[ntr:]

    R2 = R.copy()
    col_gain = np.zeros(C)
    for j in range(C):
        off = R2[tr, j].mean()
        R2[:, j] -= off
        g = np.polyfit(ideal[tr, j], R2[tr, j], 1)[0]
        col_gain[j] = g
        R2[:, j] -= g * ideal[:, j]
    print(f"after col offset+gain: resid_std={R2.std():.2f} "
          f"(var removed {100*(1-R2.var()/var_R):.1f}%)")

    # --- step 3: linear dW removal: per-column R2 ~ x ridge ---
    R3 = R2.copy()
    for j in range(C):
        R3[te, j] = R2[te, j] - ridge_fit_pred(
            x[tr], R2[tr, j], x[te], alpha=1.0)
        # train 侧用 in-sample 拟合残差(仅用于 bin 内统计,不进 R²)
        m = Ridge(alpha=1.0).fit(x[tr], R2[tr, j])
        R3[tr, j] = R2[tr, j] - m.predict(x[tr])
    print(f"after linear dW removal: test resid_std={R3[te].std():.2f} "
          f"(cumul var removed {100*(1-R3[te].var()/var_R):.1f}%)")
    R_nl = R3  # 非线性残余
    var_Rnl_te = R_nl[te].var()

    results = {"layer": L, "N": N, "K": K, "C": C,
               "alpha": float(a), "beta": float(b),
               "std_R": float(R.std()),
               "std_after_colfit": float(R2[te].std()),
               "std_Rnl": float(R_nl[te].std()),
               "col_gain_abs_med": float(np.median(np.abs(col_gain)))}

    # ---------- M0: per-column x ideal-bin 查表 ----------
    pred = np.zeros_like(R_nl[te])
    bin_var_sum, bin_cnt = 0.0, 0
    for j in range(C):
        btr, (lo, hi) = bin_ids(ideal[tr, j], N_BINS)
        bte, _ = bin_ids(ideal[te, j], N_BINS, lo, hi)
        table = np.zeros(N_BINS)
        cnt = np.zeros(N_BINS)
        np.add.at(table, btr, R_nl[tr, j])
        np.add.at(cnt, btr, 1)
        table = np.where(cnt > 0, table / np.maximum(cnt, 1), 0.0)
        pred[:, j] = table[bte]
        # bin 内方差(train): 真 iid 噪声估计
        resid_in = R_nl[tr, j] - table[btr]
        bin_var_sum += resid_in.var()
        bin_cnt += 1
    r2_M0 = r2(R_nl[te], pred)
    sigma_iid_lookup = np.sqrt(bin_var_sum / bin_cnt)
    print(f"M0 lookup(col x {N_BINS}bins): holdout R2={r2_M0:.4f}, "
          f"in-bin sigma={sigma_iid_lookup:.2f}")
    results["M0_lookup_R2"] = float(r2_M0)
    results["sigma_iid_lookup"] = float(sigma_iid_lookup)

    # ---------- M1: RFF 基线 (cos(w*ideal+phi), per-column, 共享 w) ----------
    omega = RNG.normal(0, 1.0 / ideal.std(), size=N_RFF)
    phi = RNG.uniform(0, 2 * np.pi, N_RFF)
    pred = np.zeros_like(R_nl[te])
    for j in range(C):
        Ftr = np.cos(np.outer(ideal[tr, j], omega) + phi)
        Fte = np.cos(np.outer(ideal[te, j], omega) + phi)
        pred[:, j] = ridge_fit_pred(Ftr, R_nl[tr, j], Fte, alpha=1.0)
    r2_M1 = r2(R_nl[te], pred)
    print(f"M1 RFF(ideal, {N_RFF}feat): holdout R2={r2_M1:.4f}")
    results["M1_rff_R2"] = float(r2_M1)

    # ---------- 准备 MLP 数据(采样) ----------
    S = x_summaries(x)
    def subsample(tr_idx, cap):
        if len(tr_idx) <= cap:
            return tr_idx
        return RNG.choice(tr_idx, cap, replace=False)

    def mlp_per_col(Ftr_all, Fte_all, tag, hidden=(32, 16), cap=MLP_MAX_ROWS):
        pred = np.zeros_like(R_nl[te])
        t0 = time.time()
        for j in range(C):
            trs = subsample(tr, cap)
            m = MLPRegressor(hidden_layer_sizes=hidden, max_iter=300,
                             early_stopping=True, random_state=j)
            m.fit(Ftr_all[trs], R_nl[trs, j])
            pred[:, j] = m.predict(Fte_all)
        r = r2(R_nl[te], pred)
        print(f"{tag}: holdout R2={r:.4f} ({time.time()-t0:.0f}s)")
        return r

    # M2: MLP(ideal) per-column
    F_id = ideal  # per-column 取第 j 列
    pred = np.zeros_like(R_nl[te])
    t0 = time.time()
    for j in range(C):
        trs = subsample(tr, MLP_MAX_ROWS)
        m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300,
                         early_stopping=True, random_state=j)
        m.fit(ideal[trs, j:j+1], R_nl[trs, j])
        pred[:, j] = m.predict(ideal[te, j:j+1])
    r2_M2 = r2(R_nl[te], pred)
    print(f"M2 MLP(ideal) per-col: holdout R2={r2_M2:.4f} ({time.time()-t0:.0f}s)")
    results["M2_mlp_ideal_R2"] = float(r2_M2)

    # M3: MLP(ideal + x 摘要) per-column
    F3 = np.concatenate([ideal, S], axis=1)  # 取列 j 时拼 ideal_j + S
    pred = np.zeros_like(R_nl[te])
    t0 = time.time()
    for j in range(C):
        trs = subsample(tr, MLP_MAX_ROWS)
        Xtr = np.concatenate([ideal[trs, j:j+1], S[trs]], axis=1)
        Xte = np.concatenate([ideal[te, j:j+1], S[te]], axis=1)
        m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300,
                         early_stopping=True, random_state=j)
        m.fit(Xtr, R_nl[trs, j])
        pred[:, j] = m.predict(Xte)
    r2_M3 = r2(R_nl[te], pred)
    print(f"M3 MLP(ideal+x摘要) per-col: holdout R2={r2_M3:.4f} ({time.time()-t0:.0f}s)")
    results["M3_mlp_ideal_xsum_R2"] = float(r2_M3)

    # M4: MLP(ideal + 全量 x) per-column (上限探测, 强采样)
    pred = np.zeros_like(R_nl[te])
    t0 = time.time()
    cap4 = min(MLP_MAX_ROWS, 40000)
    for j in range(C):
        trs = subsample(tr, cap4)
        Xtr = np.concatenate([ideal[trs, j:j+1], x[trs]], axis=1)
        Xte = np.concatenate([ideal[te, j:j+1], x[te]], axis=1)
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                         early_stopping=True, random_state=j)
        m.fit(Xtr, R_nl[trs, j])
        pred[:, j] = m.predict(Xte)
    r2_M4 = r2(R_nl[te], pred)
    print(f"M4 MLP(ideal+x全量) per-col: holdout R2={r2_M4:.4f} ({time.time()-t0:.0f}s)")
    results["M4_mlp_ideal_xfull_R2"] = float(r2_M4)

    # M5: 全局共享 MLP(ideal + x 摘要 + 列 one-hot), 随机 (行,列) 对采样
    t0 = time.time()
    M = 300000
    rows = RNG.randint(0, N, M)
    cols = RNG.randint(0, C, M)
    eye = np.eye(C)
    Xg = np.concatenate([ideal[rows, cols, None], S[rows], eye[cols]], axis=1)
    yg = R_nl[rows, cols]
    perm = RNG.permutation(M)
    ntr_g = int(0.8 * M)
    gtr, gte = perm[:ntr_g], perm[ntr_g:]
    m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                     early_stopping=True, random_state=0)
    m.fit(Xg[gtr], yg[gtr])
    r2_M5 = r2(yg[gte], m.predict(Xg[gte]))
    print(f"M5 全局共享MLP(ideal+x摘要+onehot): holdout R2={r2_M5:.4f} ({time.time()-t0:.0f}s)")
    results["M5_mlp_global_R2"] = float(r2_M5)

    # ---------- M6: tile 配对行交叉依赖 (硬件 8x2 tile 每次处理 2 行) ----------
    # 假设数据按跑批顺序排列, 行 (2i,2i+1) 同 batch。检查 R_nl 是否依赖配对行输入。
    even = np.arange(0, N - 1, 2)
    odd = even + 1
    # 配对划分 train/test (按对切, 避免泄漏)
    npairs = len(even)
    pidx = RNG.permutation(npairs)
    ptr, pte = pidx[: int(0.8 * npairs)], pidx[int(0.8 * npairs):]
    r2_M6_list = []
    for j in range(C):
        # R_nl[偶数行, j] ~ [x[奇数行], ideal[奇数行,j]]
        Ftr = np.concatenate([x[odd[ptr]], ideal[odd[ptr], j:j+1]], axis=1)
        Fte = np.concatenate([x[odd[pte]], ideal[odd[pte], j:j+1]], axis=1)
        p = ridge_fit_pred(Ftr, R_nl[even[ptr], j], Fte, alpha=1.0)
        r2_M6_list.append(r2(R_nl[even[pte], j], p))
    r2_M6 = float(np.median(r2_M6_list))
    print(f"M6 配对行ridge(even~partner x,ideal): med holdout R2={r2_M6:.4f}")
    results["M6_partner_ridge_R2_med"] = r2_M6
    # 配对行残差相关(无模型, 直接看结构): corr(R_nl[2i,j], R_nl[2i+1,j])
    pair_corrs = [np.corrcoef(R_nl[even, j], R_nl[odd, j])[0, 1] for j in range(C)]
    results["pair_resid_corr_med"] = float(np.median(pair_corrs))
    print(f"配对行残差相关 med={np.median(pair_corrs):.4f}")

    # ---------- M7: 跨列耦合 R_nl[:,j] ~ ideal 全列 ridge ----------
    r2_M7_list = []
    for j in range(C):
        p = ridge_fit_pred(ideal[tr], R_nl[tr, j], ideal[te], alpha=1.0)
        r2_M7_list.append(r2(R_nl[te, j], p))
    r2_M7 = float(np.median(r2_M7_list))
    print(f"M7 跨列ridge(R_nl~ideal全列): med holdout R2={r2_M7:.4f}")
    results["M7_crosscol_ridge_R2_med"] = r2_M7

    # ---------- 行间时序结构 (run 内漂移?) ----------
    # R_nl 行均值序列的一阶自相关 & 与前 5%/后 5% 行均值差
    rowmean = R_nl.mean(axis=1)
    ac1 = np.corrcoef(rowmean[:-1], rowmean[1:])[0, 1]
    head = rowmean[: max(N // 20, 10)].mean()
    tail = rowmean[-max(N // 20, 10):].mean()
    print(f"run-order: rowmean lag-1 autocorr={ac1:.4f}, head-tail mean diff={head - tail:.2f}")
    results["runorder_autocorr1"] = float(ac1)
    results["runorder_head_tail_diff"] = float(head - tail)

    # ---------- 条件均值曲线形状诊断 ----------
    # E[R_nl | ideal] 曲线: 用全体数据, 报告曲线的"可解释方差占比"与形状指标
    shape = {}
    ev_fracs = []
    for j in range(C):
        bids, _ = bin_ids(ideal[:, j], N_BINS)
        means = np.zeros(N_BINS); cnt = np.zeros(N_BINS)
        np.add.at(means, bids, R_nl[:, j]); np.add.at(cnt, bids, 1)
        means = means / np.maximum(cnt, 1)
        pred_all = means[bids]
        ev = 1 - np.var(R_nl[:, j] - pred_all) / np.var(R_nl[:, j])
        ev_fracs.append(ev)
    shape["E[Rnl|ideal]_explained_var_frac_med"] = float(np.median(ev_fracs))
    shape["E[Rnl|ideal]_explained_var_frac_p90"] = float(np.percentile(ev_fracs, 90))
    results["shape"] = shape
    print(f"cond-mean curve explains var (in-sample, med/p90 over cols): "
          f"{np.median(ev_fracs):.3f}/{np.percentile(ev_fracs,90):.3f}")

    # 导出 s 层首列的条件均值曲线供报告引用
    j0 = int(np.argmax(ev_fracs))
    bids, (lo, hi) = bin_ids(ideal[:, j0], N_BINS)
    means = np.zeros(N_BINS); cnt = np.zeros(N_BINS)
    np.add.at(means, bids, R_nl[:, j0]); np.add.at(cnt, bids, 1)
    means = means / np.maximum(cnt, 1)
    centers = np.linspace(lo, hi, N_BINS + 1)[:-1] + (hi - lo) / (2 * N_BINS)
    results["example_curve"] = {
        "col": j0,
        "centers": [round(float(v), 1) for v in centers],
        "means": [round(float(v), 2) for v in means],
        "counts": [int(v) for v in cnt],
    }

    # ---------- 汇总 ----------
    best = max(r2_M0, r2_M1, r2_M2, r2_M3, r2_M4, r2_M5, r2_M6, r2_M7)
    results["best_holdout_R2"] = float(best)
    results["predictable_std"] = float(np.sqrt(best * var_Rnl_te))
    results["unpredictable_std"] = float(np.sqrt((1 - best) * var_Rnl_te))
    print(f"BEST holdout R2={best:.4f} -> predictable_std="
          f"{results['predictable_std']:.2f}, residual(iid?)_std="
          f"{results['unpredictable_std']:.2f}")
    return results


def main():
    all_results = []
    for L in LAYERS:
        all_results.append(eval_layer(L))
    print("\n===== JSON =====")
    print(json.dumps(all_results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
