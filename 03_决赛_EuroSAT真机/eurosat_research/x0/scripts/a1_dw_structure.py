#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1: δW 内部结构分解（Round X0）

输入: x0/data/probe_pairs/probe_{layer}_{xint,ideal,hw}.npy  (5 层, 只读)
输出: stdout 汇总 + x0/results/a1_dw_structure.json

流程（每层）:
  1. hw ~ alpha*ideal + beta 全局回归 -> 残差 R
  2. R 按输出列做 per-column 增益/偏移回归: R[:,j] = g_j*ideal[:,j] + c_j -> R2
     （复现 C3 三组分分解: offset/gain 方差占比, 作为 sanity check）
  3. R2[:,j] = X @ dW[:,j] + int_j  逐列 lstsq -> dW_hat (k x n)
     -  split-half (随机对半, seed 固定): dW1, dW2
     -  分量能量用 split-half 交叉内积估计（无偏, 估计噪声在两半间独立）:
        E_comp = <comp(dW1), comp(dW2)>_F
  4. 结构检验:
     a. SVD 谱 vs 行置换零假设 (R2 行打乱后重估 dW_null) -> 低秩性
     b. dW 列均值 / 行均值能量占比 (split-half 交叉)
     c. 行 RMS 差异 (某些输入通道是否更"脆")
     d. 8x2 tile 块均值方差 vs 矩阵内打乱零假设 -> 块相关
     e. 诱导输出误差的对齐比 A = Var(X2 @ dW1) / Var(X2 @ shuffle(dW1))
        (out-of-sample; >1 说明 dW 与真实激活相关方向对齐)
  5. 稀释律直接测量: 随机行子集 S (|S|=k'), 误差能量 V(k') = E[(X[:,S]@dW[S,:])^2],
     log-log 拟合指数 gamma: iid -> 0.5 (sigma ~ sqrt(k)), 相干 -> 1.0 (sigma ~ k)
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "probe_pairs")
OUT_JSON = os.path.join(HERE, "..", "results", "a1_dw_structure.json")
LAYERS = ["s1a", "s2a", "s2b", "s3a", "s3b"]
TILE_R, TILE_C = 8, 2  # Gazelle 8x2 tile
N_PERM_NULL = 3        # 零假设置换次数
N_SUBSET = 24          # 稀释律每个 k' 的随机子集数
N_SHUF_ALIGN = 5       # 对齐比 shuffle 次数
SEED = 0


def load(layer):
    x = np.load(os.path.join(DATA, f"probe_{layer}_xint.npy")).astype(np.float64)
    ideal = np.load(os.path.join(DATA, f"probe_{layer}_ideal.npy")).astype(np.float64)
    hw = np.load(os.path.join(DATA, f"probe_{layer}_hw.npy")).astype(np.float64)
    return x, ideal, hw


def fit_dw(Xc, R2):
    """R2[:,j] = Xc @ dW[:,j] + int_j, Xc 为按列中心化的激活 (全局列均值)。
    中心化使 dW 与截距可分离 (R2 列均值~0 时 int~0)。返回 dW (k,n), int (n,)"""
    N, k = Xc.shape
    G = np.hstack([Xc, np.ones((N, 1))])
    coef, *_ = np.linalg.lstsq(G, R2, rcond=None)  # 容忍常量通道导致的共线
    return coef[:k], coef[k]


def svd_spectrum(M):
    return np.linalg.svd(M, compute_uv=False)


def scaling_exponent(X, M, rng, sizes):
    """V(k') = E[(X[:,S] @ M[S,:])^2], log-log 斜率 -> gamma (std ~ k^gamma)"""
    N, k = X.shape
    vs, ks = [], []
    for kp in sizes:
        if kp > k:
            continue
        acc = 0.0
        for _ in range(N_SUBSET):
            S = rng.choice(k, size=kp, replace=False)
            E = X[:, S] @ M[S, :]          # (N, n)
            acc += np.mean(E ** 2)
        vs.append(acc / N_SUBSET)
        ks.append(kp)
    ks = np.array(ks, float)
    vs = np.array(vs, float)
    gamma_var = np.polyfit(np.log(ks), np.log(vs), 1)[0]  # Var ~ k^(2*gamma_std)
    return 0.5 * gamma_var, list(zip(ks.tolist(), vs.tolist()))


def analyze(layer):
    rng = np.random.RandomState(SEED)
    X, ideal, hw = load(layer)
    N, k = X.shape
    n = ideal.shape[1]
    res = {"layer": layer, "N": N, "k": k, "n": n}

    # 1. 全局回归
    A = np.vstack([ideal.ravel(), np.ones(ideal.size)]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, hw.ravel(), rcond=None)
    R = hw - (alpha * ideal + beta)
    var_R = R.var()
    res["alpha"], res["beta"] = float(alpha), float(beta)
    res["resid_std"] = float(R.std())

    # 2. per-column 增益/偏移
    G = np.zeros(n)
    C = np.zeros(n)
    for j in range(n):
        Aj = np.vstack([ideal[:, j], np.ones(N)]).T
        (g, c), *_ = np.linalg.lstsq(Aj, R[:, j], rcond=None)
        G[j], C[j] = g, c
    R2 = R - (G[None, :] * ideal + C[None, :])
    res["frac_offset"] = float(C.var() / var_R)          # 列偏移(列间)方差占比
    res["frac_gain"] = float((G[None, :] * ideal).var() / var_R)
    res["gain_std_milli"] = float(G.std() * 1e3)
    res["var_R2_over_R"] = float(R2.var() / var_R)

    # 激活按列中心化 (dW 相对均值激活的偏差建模)
    xbar = X.mean(axis=0)
    Xc = X - xbar[None, :]

    # sanity: 不去 gain 时, R 的 x-线性可解释方差 (对齐 C3 的 21-50% 口径)
    dW_onR, _ = fit_dw(Xc, R)
    res["frac_xlinear_of_R"] = float(1.0 - (R - Xc @ dW_onR).var() / var_R)

    # 3. dW 回归: 全量 + split-half + 置换零假设
    dW_full, int_full = fit_dw(Xc, R2)
    pred = Xc @ dW_full + int_full[None, :]
    R3 = R2 - pred
    res["frac_dW_explained_of_R2"] = float(1.0 - R3.var() / R2.var())
    res["frac_dW_explained_of_R"] = float(pred.var() / var_R)
    res["frac_nonlin_of_R2"] = float(R3.var() / R2.var())
    res["dW_rms"] = float(np.sqrt((dW_full ** 2).mean()))

    perm = rng.permutation(N)
    h1, h2 = perm[: N // 2], perm[N // 2:]
    dW1, _ = fit_dw(Xc[h1], R2[h1])
    dW2, _ = fit_dw(Xc[h2], R2[h2])
    # split-half 可靠性 & 真实 dW 能量
    v1, v2 = dW1.ravel(), dW2.ravel()
    reliab = float(np.corrcoef(v1, v2)[0, 1])
    res["split_half_reliability"] = reliab
    E_total = float(np.dot(v1, v2))  # 交叉内积 = 真实能量无偏估计
    res["E_true_over_E_hat"] = float(E_total / np.dot(dW_full.ravel(), dW_full.ravel()))

    null_dWs = []
    for p in range(N_PERM_NULL):
        Rp = R2[rng.permutation(N)]
        dWn, _ = fit_dw(Xc, Rp)
        null_dWs.append(dWn)

    # ---- 分量能量分解 (split-half 交叉内积) ----
    def decomp(M):
        mu = M.mean()
        a = M.mean(axis=1) - mu          # 行均值 (per-输入通道)
        b = M.mean(axis=0) - mu          # 列均值 (per-输出列)
        Madd = M - mu - a[:, None] - b[None, :]
        return mu, a, b, Madd

    mu1, a1, b1, M1 = decomp(dW1)
    mu2, a2, b2, M2 = decomp(dW2)
    E_mu = float(k * n * mu1 * mu2)
    E_row = float(n * np.dot(a1, a2))
    E_col = float(k * np.dot(b1, b2))
    E_add = float((M1 * M2).sum())
    for name, E in [("E_grandmean", E_mu), ("E_rowmean", E_row),
                    ("E_colmean", E_col), ("E_addresid", E_add)]:
        res[name] = E
        res[name + "_frac"] = float(E / E_total) if E_total > 0 else float("nan")

    # ---- 4a. SVD 谱 vs 零假设 ----
    s_real = svd_spectrum(dW_full)
    s_null = np.median(np.stack([svd_spectrum(m) for m in null_dWs]), axis=0)
    e_real, e_null = s_real ** 2, s_null ** 2
    res["svd_top1_frac"] = float(e_real[0] / e_real.sum())
    res["svd_top4_frac"] = float(e_real[:4].sum() / e_real.sum())
    res["svd_top8_frac"] = float(e_real[:8].sum() / e_real.sum())
    res["svd_null_top1_frac"] = float(e_null[0] / e_null.sum())
    res["svd_null_top8_frac"] = float(e_null[:8].sum() / e_null.sum())
    res["eff_rank_real"] = float(e_real.sum() ** 2 / (e_real ** 2).sum())
    res["eff_rank_null"] = float(e_null.sum() ** 2 / (e_null ** 2).sum())
    # 加性残差(M)中超出零假设谱的低秩能量 (split-half 交叉版本)
    U1, s1, Vt1 = np.linalg.svd(M1, full_matrices=False)
    s_cross = np.array([U1[:, i] @ M2 @ Vt1[i] for i in range(len(s1))])
    below = np.where(s1 < s_null)[0]          # half1 谱跌入零假设谱之下的首个奇异值
    r_cut = int(below[0]) if len(below) else len(s1)
    E_lowrank = float(np.sum(np.clip(s_cross[:r_cut], 0, None) ** 2))
    res["svd_r_above_null"] = r_cut
    res["E_lowrank_frac_of_addresid"] = float(E_lowrank / E_add) if E_add > 0 else float("nan")
    res["E_lowrank_frac_total"] = float(E_lowrank / E_total) if E_total > 0 else float("nan")
    res["E_iid_frac_total"] = float((E_add - E_lowrank) / E_total) if E_total > 0 else float("nan")

    # ---- 4b/4c. 行/列 RMS 结构 (与零假设比) ----
    row_rms_real = np.sqrt((dW_full ** 2).mean(axis=1))
    row_rms_null = np.mean([np.sqrt((m ** 2).mean(axis=1)) for m in null_dWs], axis=0)
    res["row_rms_cv_real"] = float(row_rms_real.std() / row_rms_real.mean())
    res["row_rms_cv_null"] = float(row_rms_null.std() / row_rms_null.mean())
    res["row_rms_max_over_med"] = float(row_rms_real.max() / np.median(row_rms_real))
    col_rms_real = np.sqrt((dW_full ** 2).mean(axis=0))
    col_rms_null = np.mean([np.sqrt((m ** 2).mean(axis=0)) for m in null_dWs], axis=0)
    res["col_rms_cv_real"] = float(col_rms_real.std() / col_rms_real.mean())
    res["col_rms_cv_null"] = float(col_rms_null.std() / col_rms_null.mean())

    # ---- 4d. 8x2 tile 块结构 ----
    def block_mean_var(M):
        Mr = M[: k // TILE_R * TILE_R, : n // TILE_C * TILE_C]
        B = Mr.reshape(k // TILE_R, TILE_R, n // TILE_C, TILE_C)
        return B.mean(axis=(1, 3)).var()
    bm_real = block_mean_var(dW_full)
    bm_null = float(np.mean([block_mean_var(m) for m in null_dWs]))
    res["block8x2_mean_var_real"] = float(bm_real)
    res["block8x2_mean_var_null"] = bm_null
    res["block8x2_excess_ratio"] = float(bm_real / bm_null) if bm_null > 0 else float("nan")

    # ---- 4e. 对齐比 (out-of-sample: dW1 在 h2 上) ----
    Xte = Xc[h2]
    E_real_te = Xte @ dW1
    v_real = float(E_real_te.var())
    v_sh = []
    for _ in range(N_SHUF_ALIGN):
        sh = dW1.ravel()[rng.permutation(dW1.size)].reshape(dW1.shape)
        v_sh.append((Xte @ sh).var())
    res["align_ratio"] = float(v_real / np.mean(v_sh))
    # 零假设对齐比 (估计噪声本身也有反对齐, 需作基线)
    v_null, v_null_sh = [], []
    for m in null_dWs:
        v_null.append((Xte @ m).var())
        for _ in range(2):
            sh = m.ravel()[rng.permutation(m.size)].reshape(m.shape)
            v_null_sh.append((Xte @ sh).var())
    res["align_ratio_null"] = float(np.mean(v_null) / np.mean(v_null_sh))

    # ---- 5. 稀释律指数 gamma (std_err ~ k^gamma) ----
    sizes = sorted(set(max(2, int(round(k * f))) for f in
                       [0.125, 0.25, 0.375, 0.5, 0.75, 1.0]))
    mu_f, a_f, b_f, M_f = decomp(dW_full)
    comps = {
        "full": dW_full,
        "rowmean": (a_f[:, None] * np.ones((1, n))),
        "colmean": (np.ones((k, 1)) * b_f[None, :]),
        "addresid": M_f,
    }
    res["gamma"] = {}
    curves = {}
    for cname, M in comps.items():
        g, curve = scaling_exponent(Xc, M, rng, sizes)
        res["gamma"][cname] = float(g)
        curves[cname] = curve
    # 零假设 gamma: 估计噪声自身的稀释指数 (基线)
    g_nulls = []
    null_curve_acc = None
    for m in null_dWs:
        gn, cn = scaling_exponent(Xc, m, rng, sizes)
        g_nulls.append(gn)
        if null_curve_acc is None:
            null_curve_acc = {kk: [] for kk, _ in cn}
        for kk, vv in cn:
            null_curve_acc[kk].append(vv)
    res["gamma"]["null_full"] = float(np.mean(g_nulls))
    # 超额能量稀释指数: V_ex(k') = V_real - V_null 的 log-log 斜率
    null_curve_med = {kk: float(np.median(vv)) for kk, vv in null_curve_acc.items()}
    xs, ys = [], []
    for kk, vv in curves["full"]:
        ex = vv - null_curve_med.get(kk, 0.0)
        if ex > 0:
            xs.append(np.log(kk))
            ys.append(np.log(ex))
    res["gamma"]["excess_full"] = float(np.polyfit(xs, ys, 1)[0] / 2.0) if len(xs) >= 3 else float("nan")
    # 参考: iid 随机矩阵 (期望 gamma=0.5) 与全 1 相干方向 (测量激活跨通道相关性)
    g, _ = scaling_exponent(Xc, rng.standard_normal((k, n)), rng, sizes)
    res["gamma"]["iid_ref"] = float(g)
    g, _ = scaling_exponent(Xc, np.ones((k, n)) / k, rng, sizes)
    res["gamma"]["coherent_ref"] = float(g)
    res["x_mean"] = float(X.mean())
    return res


def main():
    allres = [analyze(L) for L in LAYERS]
    hdr = (f"{'layer':5s} {'N':>6s} {'k':>4s} {'n':>4s} {'xlin%R':>7s} {'gain%R':>7s} {'off%R':>6s} {'dW%R2':>7s} {'rely':>6s} "
           f"{'row%':>6s} {'col%':>6s} {'lowrk%':>7s} {'iid%':>6s} "
           f"{'blk/null':>8s} {'align':>6s} {'al_nul':>6s} {'g_full':>6s} {'g_null':>6s} {'g_exc':>6s} {'g_add':>6s} {'g_iidr':>6s} {'g_cohr':>6s}")
    print(hdr)
    for r in allres:
        print(f"{r['layer']:5s} {r['N']:6d} {r['k']:4d} {r['n']:4d} "
              f"{100*r['frac_xlinear_of_R']:6.1f}% {100*r['frac_gain']:6.1f}% {100*r['frac_offset']:5.1f}% "
              f"{100*r['frac_dW_explained_of_R2']:6.1f}% "
              f"{r['split_half_reliability']:6.3f} "
              f"{100*r['E_rowmean_frac']:5.1f}% {100*r['E_colmean_frac']:5.1f}% "
              f"{100*r['E_lowrank_frac_total']:6.1f}% {100*r['E_iid_frac_total']:5.1f}% "
              f"{r['block8x2_excess_ratio']:8.2f} {r['align_ratio']:6.3f} {r['align_ratio_null']:6.3f} "
              f"{r['gamma']['full']:6.3f} {r['gamma']['null_full']:6.3f} {r['gamma']['excess_full']:6.3f} "
              f"{r['gamma']['addresid']:6.3f} "
              f"{r['gamma']['iid_ref']:6.3f} {r['gamma']['coherent_ref']:6.3f}")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(allres, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
