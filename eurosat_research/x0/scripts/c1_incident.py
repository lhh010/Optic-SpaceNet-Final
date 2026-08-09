#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1-incident: 2026-08-09 真机抢占事件本地归因分析（Round X0）

问题：他队 server_gazelle.py 抢占光器件（~05:46-06:00Z）后，我方同一校准窗口内
跑批轨迹系统性偏低（93.3%@104 -> 84.1%@208 vs 昨日同段 91.5%@200）。
判决三假设：
  (i)  校准态被改写（重 cali 可恢复）
  (ii) 硬件物理慢漂/温漂（重 cali 不能立即恢复）
  (iii) 软件/驱动配置残留

输入（x0/data/incident_20260809/）:
  pairs_c3d_w1/probe_c3d_{s1a,s2a,s2b,s3a,s3b}_{xint,ideal,hw}.npy   # 抢占前 05:42 pairs
  pairs_c2c_old/probe_{layer}_{ideal,hw}.npy                          # 昨日 c2c pairs（对照 x0/data/probe_pairs/）
  probe_post_*/probe_post_{layer}_{ideal,hw}.npy                      # 抢占后小探针
  calib/*.json                                                        # 三份 calib json
  runs/*                                                              # 污染窗口跑批轨迹/logits + 昨日同形状 run
  *.md/*.txt/*.log                                                    # 日志、他队 server 信息、EBR 快照、MANIFEST

输出: stdout 汇总 + x0/results/c1_incident.json（供 C1_incident_analysis.md 引用）

方法（复用 a1_dw_structure.py 的分解口径）:
  每层每时间点: hw ~ alpha*ideal + beta 全局标量回归 -> R
               R 逐列回归 alpha_c/beta_c -> R2（iid+δW+非线性残余）
  自然起伏口径: T1 pairs 内部行 bootstrap (B=400, 50% 子采样) 估计
               alpha/beta/逐列参数的抽样噪声，作为"同窗口自然起伏"基线。
  判决量:
    z_alpha, z_beta      = |Δ(T1->T2)| / 自然起伏 std        （校准态判决）
    corr(beta_c T1,T2)   跨时间列结构相关                     （列指纹稳定性）
    Δbeta_c 形态          全局平移 vs 列特异改写               （(i) vs (ii)/(iii)）
    iid 底噪比 T2/T1     R2 的行内 std 之比                   （物理层变化判据）
"""
import json
import os
import re
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X0 = os.path.join(HERE, "..")
INC = os.path.join(X0, "data", "incident_20260809")
OLD_FALLBACK = os.path.join(X0, "data", "probe_pairs")   # 昨日 c2c pairs 本地已有副本
OUT_JSON = os.path.join(X0, "results", "c1_incident.json")

LAYERS = ["s1a", "s2a", "s2b", "s3a", "s3b"]
SEED = 0
N_BOOT = 400          # 自然起伏 bootstrap 次数
BOOT_FRAC = 0.5       # 每次子采样比例
rng = np.random.RandomState(SEED)


def try_load_npy(path):
    """容忍未写完的 npy（scp 进行中）: 失败返回 None"""
    try:
        return np.load(path, allow_pickle=False).astype(np.float64)
    except Exception as e:
        print(f"  [warn] 无法加载 {path}: {e}")
        return None


def find_pairs(prefix, dirs):
    """在 dirs 中按 prefix 找 {prefix}{layer}_{ideal,hw}.npy。返回 {layer: (ideal, hw)}"""
    out = {}
    for d in dirs:
        for layer in LAYERS:
            fi = os.path.join(d, f"{prefix}{layer}_ideal.npy")
            fh = os.path.join(d, f"{prefix}{layer}_hw.npy")
            if layer not in out and os.path.exists(fi) and os.path.exists(fh):
                a, b = try_load_npy(fi), try_load_npy(fh)
                if a is not None and b is not None and a.shape == b.shape:
                    out[layer] = (a, b)
    return out


def fit_scalar(ideal, hw):
    """hw ~ alpha*ideal + beta 全局回归。返回 alpha, beta, R(N,n)"""
    A = np.vstack([ideal.ravel(), np.ones(ideal.size)]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, hw.ravel(), rcond=None)
    R = hw - (alpha * ideal + beta)
    return float(alpha), float(beta), R


def fit_columns(ideal, R):
    """R[:,j] = g_j*ideal[:,j] + c_j 逐列回归 -> (G, C, R2)"""
    N, n = R.shape
    G = np.zeros(n)
    C = np.zeros(n)
    for j in range(n):
        Aj = np.vstack([ideal[:, j], np.ones(N)]).T
        (g, c), *_ = np.linalg.lstsq(Aj, R[:, j], rcond=None)
        G[j], C[j] = g, c
    R2 = R - (G[None, :] * ideal + C[None, :])
    return G, C, R2


def analyze_layer(ideal, hw):
    """单时间点单层完整分解"""
    alpha, beta, R = fit_scalar(ideal, hw)
    G, C, R2 = fit_columns(ideal, R)
    var_R = float(R.var())
    # per-column alpha_c = alpha*(1+g_j) 展开: hw[:,j] ≈ (alpha+g_j)*ideal + (beta+c_j)
    # 直接在 hw 上做逐列回归得到绝对口径 alpha_c, beta_c
    N, n = hw.shape
    ac = np.zeros(n)
    bc = np.zeros(n)
    for j in range(n):
        Aj = np.vstack([ideal[:, j], np.ones(N)]).T
        (a, b), *_ = np.linalg.lstsq(Aj, hw[:, j], rcond=None)
        ac[j], bc[j] = a, b
    return {
        "alpha": alpha, "beta": beta, "resid_std": float(R.std()),
        "frac_offset": float(C.var() / var_R),          # 列偏移方差占比
        "frac_gain": float((G[None, :] * ideal).var() / var_R),
        "alpha_c": ac, "beta_c": bc,                     # 绝对口径逐列参数
        "R2_rowstd": R2,                                  # 去列结构后残余（保留用于形态对比）
        "col_mean_R": C,                                  # 逐列偏移（标量 calib 后）
    }


def natural_variability(ideal, hw):
    """T1 pairs 内部 bootstrap: 同窗口自然起伏口径。
    返回 dict(std_alpha, std_beta, std_beta_c(逐列 std 的抽样 std), std_resid_std)"""
    N = ideal.shape[0]
    m = int(N * BOOT_FRAC)
    alphas, betas, rstds = [], [], []
    n = ideal.shape[1]
    bcs = np.zeros((N_BOOT, n))
    idx_all = np.arange(N)
    for b in range(N_BOOT):
        idx = rng.choice(idx_all, size=m, replace=False)
        a, be, R = fit_scalar(ideal[idx], hw[idx])
        alphas.append(a)
        betas.append(be)
        rstds.append(R.std())
        for j in range(n):
            Aj = np.vstack([ideal[idx, j], np.ones(m)]).T
            (aa, bb), *_ = np.linalg.lstsq(Aj, hw[idx, j], rcond=None)
            bcs[b, j] = bb
    return {
        "std_alpha": float(np.std(alphas)),
        "std_beta": float(np.std(betas)),
        "std_resid_std": float(np.std(rstds)),
        "std_beta_c_mean": float(np.mean(np.std(bcs, axis=0))),   # 单列 beta_c 的抽样噪声
        "std_of_col_std_beta_c": float(np.std(np.std(bcs, axis=1))),  # 跨列 std 的抽样噪声
    }


def compare(name, t1, t2, nat=None):
    """两个时间点单层对比 -> 判决量"""
    d = {"pair": name}
    d["d_alpha"] = t2["alpha"] - t1["alpha"]
    d["d_beta"] = t2["beta"] - t1["beta"]
    d["d_alpha_rel"] = d["d_alpha"] / t1["alpha"]
    d["resid_std_ratio"] = t2["resid_std"] / t1["resid_std"]
    if nat:
        d["z_alpha"] = d["d_alpha"] / nat["std_alpha"] if nat["std_alpha"] > 0 else float("nan")
        d["z_beta"] = d["d_beta"] / nat["std_beta"] if nat["std_beta"] > 0 else float("nan")
        d["z_resid_std"] = (t2["resid_std"] - t1["resid_std"]) / nat["std_resid_std"] if nat["std_resid_std"] > 0 else float("nan")
    # 逐列结构相关（去列均值后相关 = 列指纹形态一致性）
    b1, b2 = t1["beta_c"], t2["beta_c"]
    a1, a2 = t1["alpha_c"], t2["alpha_c"]
    d["corr_beta_c"] = float(np.corrcoef(b1 - b1.mean(), b2 - b2.mean())[0, 1])
    d["corr_alpha_c"] = float(np.corrcoef(a1 - a1.mean(), a2 - a2.mean())[0, 1])
    # Δbeta_c 形态: 全局平移成分 vs 列特异成分
    dbc = b2 - b1
    d["dbeta_c_mean"] = float(dbc.mean())
    d["dbeta_c_colstd"] = float(dbc.std())               # 列特异改写的幅度
    dac = a2 - a1
    d["dalpha_c_mean_rel"] = float(dac.mean() / a1.mean())
    d["dalpha_c_colstd_rel"] = float(dac.std() / a1.mean())
    # iid 底噪对比（去列结构后残余的整体 std）
    d["R2_std_t1"] = float(t1["R2_rowstd"].std())
    d["R2_std_t2"] = float(t2["R2_rowstd"].std())
    d["R2_std_ratio"] = d["R2_std_t2"] / d["R2_std_t1"]
    d["frac_offset_t1"] = t1["frac_offset"]
    d["frac_offset_t2"] = t2["frac_offset"]
    return d


def scan_aux_files():
    """读 MANIFEST / 日志 / server 信息 / EBR 快照等文本材料（只收集路径+摘要）"""
    aux = {}
    for pat in ["MANIFEST.md", "*.md", "*.txt", "*.log", "calib/*.json", "server*/*", "server*"]:
        for f in sorted(glob.glob(os.path.join(INC, pat))):
            if os.path.isfile(f):
                aux[os.path.relpath(f, INC)] = os.path.getsize(f)
    return aux


def main():
    print("=" * 72)
    print("C1-incident 归因分析")
    print("=" * 72)
    result = {"incident_dir": INC, "aux_files": scan_aux_files()}

    # ---- 载入三时间点 pairs ----
    T0 = find_pairs("probe_", [os.path.join(INC, "pairs_c2c_old"), OLD_FALLBACK])
    T1 = find_pairs("probe_c3d_", [os.path.join(INC, "pairs_c3d_w1")])
    # 抢占后小探针: 可能命名为 probe_post_s2a_ideal.npy，位置任意
    T2 = {}
    for f in glob.glob(os.path.join(INC, "**", "*post*"), recursive=True):
        for layer in LAYERS:
            if layer in f and f.endswith("_ideal.npy"):
                fh = f[:-len("_ideal.npy")] + "_hw.npy"
                if os.path.exists(fh) and layer not in T2:
                    a, b = try_load_npy(f), try_load_npy(fh)
                    if a is not None and b is not None and a.shape == b.shape:
                        T2[layer] = (a, b)
    print(f"T0(昨日c2c): {sorted(T0.keys())}")
    print(f"T1(抢占前c3d): {sorted(T1.keys())}")
    print(f"T2(抢占后post): {sorted(T2.keys())}")

    # ---- 逐层分解 ----
    ana = {}
    for tag, T in [("T0", T0), ("T1", T1), ("T2", T2)]:
        ana[tag] = {}
        for layer, (ideal, hw) in T.items():
            ana[tag][layer] = analyze_layer(ideal, hw)

    # ---- 自然起伏口径（T1 内部 bootstrap） ----
    nat = {}
    for layer, (ideal, hw) in T1.items():
        nat[layer] = natural_variability(ideal, hw)
        print(f"自然起伏[{layer}]: std_alpha={nat[layer]['std_alpha']:.5f} "
              f"std_beta={nat[layer]['std_beta']:.2f} std_resid={nat[layer]['std_resid_std']:.2f} "
              f"beta_c抽样噪声={nat[layer]['std_beta_c_mean']:.2f}")
    result["natural"] = nat

    # ---- 对比表 ----
    cmps = []
    for layer in LAYERS:
        if layer in ana["T1"] and layer in ana["T2"]:
            cmps.append(compare(f"T1->T2/{layer}", ana["T1"][layer], ana["T2"][layer], nat.get(layer)))
        if layer in ana["T0"] and layer in ana["T1"]:
            cmps.append(compare(f"T0->T1/{layer}", ana["T0"][layer], ana["T1"][layer]))
        if layer in ana["T0"] and layer in ana["T2"]:
            cmps.append(compare(f"T0->T2/{layer}", ana["T0"][layer], ana["T2"][layer]))
    result["comparisons"] = cmps

    hdr = f"{'pair':16s} {'dα':>9s} {'dα%':>7s} {'dβ':>9s} {'zα':>7s} {'zβ':>7s} {'zres':>7s} {'r(βc)':>7s} {'r(αc)':>7s} {'Δβc平移':>9s} {'Δβc列std':>9s} {'iid比':>6s}"
    print("\n" + hdr)
    for c in cmps:
        print(f"{c['pair']:16s} {c['d_alpha']:+9.4f} {100*c['d_alpha_rel']:+6.2f}% {c['d_beta']:+9.1f} "
              f"{c.get('z_alpha', float('nan')):7.1f} {c.get('z_beta', float('nan')):7.1f} {c.get('z_resid_std', float('nan')):7.1f} "
              f"{c['corr_beta_c']:7.3f} {c['corr_alpha_c']:7.3f} {c['dbeta_c_mean']:+9.1f} {c['dbeta_c_colstd']:9.1f} {c['R2_std_ratio']:6.3f}")

    # ---- 标量参数总表 ----
    print(f"\n{'layer':6s} {'T':3s} {'alpha':>9s} {'beta':>10s} {'resid_std':>10s} {'off%':>6s} {'gain%':>6s}")
    scal = {}
    for tag in ["T0", "T1", "T2"]:
        for layer in LAYERS:
            if layer in ana[tag]:
                a = ana[tag][layer]
                scal[f"{tag}/{layer}"] = {k: a[k] for k in ["alpha", "beta", "resid_std", "frac_offset", "frac_gain"]}
                print(f"{layer:6s} {tag:3s} {a['alpha']:9.5f} {a['beta']:10.1f} {a['resid_std']:10.1f} "
                      f"{100*a['frac_offset']:5.1f}% {100*a['frac_gain']:5.1f}%")
    result["scalar"] = scal

    # ---- calib json 对比 ----
    calibs = {}
    for f in sorted(glob.glob(os.path.join(INC, "calib", "*.json"))) + sorted(glob.glob(os.path.join(INC, "**", "calib*.json"), recursive=True)):
        try:
            with open(f) as fp:
                calibs[os.path.relpath(f, INC)] = json.load(fp)
        except Exception as e:
            print(f"  [warn] calib json 读取失败 {f}: {e}")
    result["calib_files"] = list(calibs.keys())

    # ---- calib json 标量参数两两 diff（同层 alpha/beta） ----
    def leaves(d, p=""):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(leaves(v, f"{p}/{k}"))
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            out[p] = float(d)
        return out
    scalar_jsons = {k: leaves(v) for k, v in calibs.items() if "col" not in k}
    names = sorted(scalar_jsons.keys())
    result["calib_pairwise_diff"] = {}
    if len(names) >= 2:
        print(f"\n标量 calib 两两 diff ({names}):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = scalar_jsons[names[i]], scalar_jsons[names[j]]
                common = sorted(set(a) & set(b))
                diffs = {kk: {"a": a[kk], "b": b[kk], "d": b[kk] - a[kk]}
                         for kk in common if abs(b[kk] - a[kk]) > 1e-12}
                result["calib_pairwise_diff"][f"{names[i]} vs {names[j]}"] = diffs
                for kk, vv in diffs.items():
                    print(f"  {kk}: {vv['a']:.5f} -> {vv['b']:.5f}  (Δ{vv['d']:+.5f})")

    # ---- 轨迹分析: 污染窗口 v2 vs 昨日 c2c（同测试集顺序，段对段） ----
    def traj(path):
        out = []
        for line in open(path, errors="ignore"):
            m = re.search(r"\[\s*(\d+)/1000\] acc=([\d.]+)% elapsed=([\d.]+)s", line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        return np.array(out) if out else None

    RUNS = os.path.join(INC, "runs")
    v2 = traj(os.path.join(RUNS, "run_c1_scalar.log"))
    c2c = traj(os.path.join(RUNS, "run_c2c.log"))
    col = traj(os.path.join(RUNS, "run_c1_col.log"))
    if v2 is not None and c2c is not None:
        n = min(len(v2), len(c2c))
        v2, c2c = v2[:n], c2c[:n]
        segs = [(8, 104), (112, 200), (208, 392), (400, 600), (608, 800), (808, 1000)]
        seg_rows = []
        print("\n段对段 batch 精度对比 (v2 污染窗口 vs 昨日 c2c):")
        for lo_, hi in segs:
            mv = (v2[:, 0] >= lo_) & (v2[:, 0] <= hi)
            mc = (c2c[:, 0] >= lo_) & (c2c[:, 0] <= hi)
            vb = np.diff(v2[mv][:, 1] * v2[mv][:, 0]) / 8.0
            cb = np.diff(c2c[mc][:, 1] * c2c[mc][:, 0]) / 8.0
            # 该段的 wall-clock 起止（v2, 由 05:59 起跑）
            t0, t1 = float(v2[mv][0, 2]), float(v2[mv][-1, 2])
            seg_rows.append({"seg": f"{lo_}-{hi}", "v2_batch": float(vb.mean()),
                             "c2c_batch": float(cb.mean()), "diff": float(vb.mean() - cb.mean()),
                             "v2_elapsed": [t0, t1]})
            print(f"  {lo_:4d}-{hi:4d}: v2 {vb.mean():6.2f}  c2c {cb.mean():6.2f}  diff {vb.mean()-cb.mean():+6.2f}  (v2 elapsed {t0:.0f}-{t1:.0f}s)")
        result["trajectory_segments"] = seg_rows
        # 逐 batch diff 序列（供拐点检查）
        bd = (np.diff(v2[:, 1] * v2[:, 0]) - np.diff(c2c[:, 1] * c2c[:, 0])) / 8.0
        result["trajectory_batchdiff"] = {
            "sample_idx": v2[1:, 0].tolist(),
            "elapsed": v2[1:, 2].tolist(),
            "batch_diff_pt": bd.tolist(),
            "mean_diff": float(bd.mean()),
        }
        print(f"  全程 batch diff 均值 {bd.mean():+.2f}pt; 前 1/3 {bd[:len(bd)//3].mean():+.2f}pt "
              f"中 1/3 {bd[len(bd)//3:2*len(bd)//3].mean():+.2f}pt 后 1/3 {bd[2*len(bd)//3:].mean():+.2f}pt")
    if col is not None and v2 is not None:
        mc = col[:, 0] <= 392
        mv = v2[:, 0] <= 392
        kb = np.diff(col[mc][:, 1] * col[mc][:, 0]) / 8.0
        vb2 = np.diff(v2[mv][:, 1] * v2[mv][:, 0]) / 8.0
        cb2 = np.diff(c2c[mv][:, 1] * c2c[mv][:, 0]) / 8.0
        result["col_vs_scalar_392"] = {"col_batch": float(kb.mean()), "scalar_batch": float(vb2.mean()),
                                       "c2c_batch": float(cb2.mean())}
        print(f"\n同段(8-392) batch 均值: 逐列calib(col) {kb.mean():.2f}  v2 scalar {vb2.mean():.2f}  昨日c2c {cb2.mean():.2f}")

    # ---- logits 时间趋势（边际/尺度是否随 run 时间漂移） ----
    lg = os.path.join(RUNS, "logits_c1_scalar.npy")
    labels = os.path.join(X0, "data", "labels_1000.npy")
    if os.path.exists(lg) and os.path.exists(labels):
        L = np.load(lg)
        lab = np.load(labels)
        pred = L.argmax(1)
        correct = pred == lab[:len(pred)]
        srt = np.sort(L, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        rows = []
        print("\nlogits 分段时间趋势:")
        for a_, b_ in [(0, 104), (104, 200), (200, 392), (392, 600), (600, 800), (800, 1000)]:
            rows.append({"seg": f"{a_}-{b_}", "acc": float(correct[a_:b_].mean()),
                         "margin_mean": float(margin[a_:b_].mean()),
                         "logit_rms": float(np.sqrt((L[a_:b_] ** 2).mean()))})
            print(f"  {a_:4d}-{b_:4d}: acc={correct[a_:b_].mean()*100:6.2f} margin={margin[a_:b_].mean():6.2f} logit_rms={rows[-1]['logit_rms']:6.2f}")
        result["logits_trend"] = rows

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
