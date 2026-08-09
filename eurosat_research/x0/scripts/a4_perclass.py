"""
================================================================================
 a4_perclass.py — Round X0 / A4: c3d per-class hw 错误分布分析
================================================================================
 输入:
   runs/c3d_J1_v8probe15_local/hw_logits_1000.npy   (真机 logits, 1000×10)
   runs/c3d_J1_v8probe15_local/fake_logits_1000.npy (离线 FAKE logits, 同序)
 labels 复现:
   EuroSAT ImageFolder 顺序 (sorted 类目录 + sorted 文件名) + eurosat_split
   (RandomState(42).shuffle(list(range(27000))), test = idx[5400:10800][:1000])
 自检:
   1) 与历史导出 Gazelle-national/mnist/j1_board/weights_j1/test_labels_j1.npy 对比
   2) fake acc 应 ≈95-96%
 输出:
   x0/data/labels_1000.npy + stdout 报告 (per-class 表 / 混淆对 / 统计检验)
================================================================================
"""
import os
import sys
import numpy as np

ROOT = "/Users/ms.chen/Projects/2607-ciciec"
ER = os.path.join(ROOT, "Ltsimulator-test", "eurosat_research")
RUN_DIR = os.path.join(ER, "runs", "c3d_J1_v8probe15_local")
DATA_DIR = os.path.join(ROOT, "Ltsimulator-test", "data", "EuroSAT_RGB")
REF_LABELS = os.path.join(ROOT, "Gazelle-national", "mnist", "j1_board",
                          "weights_j1", "test_labels_j1.npy")
OUT_LABELS = os.path.join(ER, "x0", "data", "labels_1000.npy")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED, N_TEST = 42, 1000


def reproduce_labels():
    """复现 torchvision ImageFolder 的样本顺序 → labels。"""
    classes = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_files = []
    for c in classes:
        cdir = os.path.join(DATA_DIR, c)
        fnames = sorted(f for f in os.listdir(cdir)
                        if os.path.splitext(f)[1].lower() in IMG_EXT)
        n_files.append(len(fnames))
    n = sum(n_files)
    # sample 索引 i → 类别: 连续区间 (ImageFolder 先按类再按文件名排序)
    labels_all = np.empty(n, dtype=np.int64)
    off = 0
    for i, cnt in enumerate(n_files):
        labels_all[off:off + cnt] = i
        off += cnt
    # eurosat_split.split_indices: val=0.2 test=0.2, seed=42
    indices = list(range(n))
    np.random.RandomState(SEED).shuffle(indices)
    val_size = int(n * 0.2)
    test_size = int(n * 0.2)
    test_idx = indices[val_size:val_size + test_size]
    sel = test_idx[:N_TEST]
    return labels_all[sel], classes, n, n_files


def main():
    labels, classes, n, n_files = reproduce_labels()
    C = len(classes)
    print(f"[labels] n_total={n}, per-class files={n_files}")
    print(f"[labels] classes={classes}")

    # --- 自检 1: 与历史导出对比 ---
    if os.path.exists(REF_LABELS):
        ref = np.load(REF_LABELS)
        k = min(len(ref), len(labels))
        agree = int((ref[:k] == labels[:k]).sum())
        print(f"[selfcheck] 历史导出 test_labels_j1.npy len={len(ref)}, "
              f"前{k}个一致 {agree}/{k}"
              f"{' [OK]' if agree == k else ' [!! MISMATCH]'}")
    else:
        print("[selfcheck] 历史导出 labels 不存在, 跳过对比")

    # --- 加载 logits ---
    hw = np.load(os.path.join(RUN_DIR, "hw_logits_1000.npy"))
    fake = np.load(os.path.join(RUN_DIR, "fake_logits_1000.npy"))
    assert hw.shape == fake.shape == (len(labels), C), (hw.shape, fake.shape)
    hw_pred = hw.argmax(1)
    fake_pred = fake.argmax(1)

    hw_acc = (hw_pred == labels).mean()
    fake_acc = (fake_pred == labels).mean()
    print(f"[selfcheck] fake acc = {fake_acc*100:.2f}%  (期望≈95-96%)")
    print(f"[overall]  hw acc = {hw_acc*100:.2f}%  (n={len(labels)})")

    # --- 自检 2: fake acc 合理性 ---
    assert 0.93 <= fake_acc <= 0.98, f"fake acc {fake_acc:.4f} 异常, label 顺序可能错"
    np.save(OUT_LABELS, labels)
    print(f"[saved] {OUT_LABELS}")

    # --- per-class 表 ---
    sup = np.bincount(labels, minlength=C)
    hw_ok = (hw_pred == labels)
    fake_ok = (fake_pred == labels)
    hw_err = ~hw_ok
    fake_err = ~fake_ok
    hw_only = hw_err & fake_ok          # hw 错 & fake 对 (噪声所致)
    both_err = hw_err & fake_err        # 两者都错 (模型固有)
    fake_only = hw_ok & fake_err        # hw 反而对

    print("\n=== per-class 精度表 ===")
    print(f"{'class':<22} {'sup':>4} {'hw_err':>6} {'fake_err':>8} "
          f"{'hw_only':>7} {'both':>5} {'hw_acc%':>8} {'fake_acc%':>9} "
          f"{'hw_only%':>8} {'fake_err%':>9}")
    for c in range(C):
        m = labels == c
        print(f"{classes[c]:<22} {sup[c]:>4} {int(hw_err[m].sum()):>6} "
              f"{int(fake_err[m].sum()):>8} {int(hw_only[m].sum()):>7} "
              f"{int(both_err[m].sum()):>5} {hw_ok[m].mean()*100:>8.2f} "
              f"{fake_ok[m].mean()*100:>9.2f} {hw_only[m].mean()*100:>8.2f} "
              f"{fake_err[m].mean()*100:>9.2f}")

    n_hw_only = int(hw_only.sum())
    n_fake_err = int(fake_err.sum())
    print(f"\ntotal: hw_err={int(hw_err.sum())} fake_err={n_fake_err} "
          f"hw_only={n_hw_only} both={int(both_err.sum())} "
          f"fake_only={int(fake_only.sum())}")

    # --- hw-only 错误 per-class 分布 ---
    hw_only_cls = np.array([int(hw_only[labels == c].sum()) for c in range(C)])
    fake_err_cls = np.array([int(fake_err[labels == c].sum()) for c in range(C)])
    print("\n=== hw-only 错误类别分布 ===")
    for c in range(C):
        share = hw_only_cls[c] / max(n_hw_only, 1) * 100
        print(f"{classes[c]:<22} {hw_only_cls[c]:>3}  ({share:5.1f}%)")

    # --- hw-only 混淆对 (真实类 → 预测类) ---
    print("\n=== hw-only 混淆对 (true -> hw_pred, top) ===")
    pairs = {}
    for t, p in zip(labels[hw_only], hw_pred[hw_only]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    for (t, p), cnt in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {classes[t]:>20} -> {classes[p]:<20} : {cnt}")

    # --- fake 错误混淆对 (对照: 模型固有短板方向) ---
    print("\n=== fake 错误混淆对 (true -> fake_pred, top) ===")
    pairs_f = {}
    for t, p in zip(labels[fake_err], fake_pred[fake_err]):
        pairs_f[(int(t), int(p))] = pairs_f.get((int(t), int(p)), 0) + 1
    for (t, p), cnt in sorted(pairs_f.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {classes[t]:>20} -> {classes[p]:<20} : {cnt}")

    # ============ 统计检验 ============
    from scipy import stats
    rng = np.random.RandomState(0)

    print("\n=== 检验 1: hw-only 类别分布 vs 均匀 (chi2 GOF) ===")
    exp_u = np.full(C, n_hw_only / C)
    chi2_u, p_u = stats.chisquare(hw_only_cls, exp_u)
    print(f"chi2={chi2_u:.2f} df={C-1} p={p_u:.4f}  "
          f"(counts={hw_only_cls.tolist()}, 期望格值={n_hw_only/C:.1f})")

    print("\n=== 检验 2: hw-only 率 vs fake 错误率, 逐类同构检验 ===")
    # 列联表: 每类 [hw_only, 非hw_only] vs [fake_err, 非fake_err]
    # H0: 两类错误率按类同分布 (hw-only 是 fake 错误的随机放大)
    tab = np.stack([hw_only_cls, sup - hw_only_cls,
                    fake_err_cls, sup - fake_err_cls])
    chi2_h, p_h, dof_h, _ = stats.chi2_contingency(tab)
    print(f"2 x {C} 同构卡方: chi2={chi2_h:.2f} df={dof_h} p={p_h:.4f}")

    print("\n=== 检验 3: hw-only 类别分布 vs fake 错误分布形状 (多项 GOF) ===")
    # H0: hw-only 各类计数 ~ Multinomial(n_hw_only, fake_err 类别占比)
    exp_f = n_hw_only * fake_err_cls / fake_err_cls.sum()
    # 期望<5 的格子合并以保证卡方近似
    order = np.argsort(exp_f)
    obs_m, exp_m = [], []
    acc_o = acc_e = 0.0
    for c in order:
        acc_o += hw_only_cls[c]
        acc_e += exp_f[c]
        if acc_e >= 5:
            obs_m.append(acc_o); exp_m.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:  # 尾部并入最后一格
        obs_m[-1] += acc_o; exp_m[-1] += acc_e
    obs_m = np.array(obs_m); exp_m = np.array(exp_m)
    chi2_f, p_f = stats.chisquare(obs_m, exp_m)
    print(f"合并后 {len(obs_m)} 格: obs={obs_m.astype(int).tolist()}")
    print(f"               exp={np.round(exp_m,1).tolist()}")
    print(f"chi2={chi2_f:.2f} df={len(obs_m)-1} p={p_f:.4f}")

    print("\n=== 检验 4: bootstrap — 随机放大零假设下各类 hw-only 率 95% CI ===")
    # 零假设: hw-only 是 fake 错误的随机放大 => 在每类内部, 非fake错误的样本
    # 以相同概率 p_c? 更朴素的操作: 在"fake 对的样本"池内, 每类按全局
    # hw-only 率 抽取, 得到各类 hw-only 计数分布 → CI
    B = 20000
    fake_ok_cnt = np.array([int((labels == c).sum() - fake_err_cls[c])
                            for c in range(C)])
    pool = []  # fake 对样本的类别标签
    for c in range(C):
        pool += [c] * fake_ok_cnt[c]
    pool = np.array(pool)
    p_global = n_hw_only / len(pool)
    sim = np.empty((B, C), dtype=np.int32)
    for b in range(B):
        hit = rng.random(len(pool)) < p_global
        sim[b] = np.bincount(pool[hit], minlength=C)
    lo = np.percentile(sim, 2.5, axis=0).astype(int)
    hi = np.percentile(sim, 97.5, axis=0).astype(int)
    print(f"(全局率 {p_global*100:.2f}%, fake对池={len(pool)})")
    print(f"{'class':<22} {'obs':>4} {'CI_lo':>6} {'CI_hi':>6}  out?")
    n_out = 0
    for c in range(C):
        out = hw_only_cls[c] < lo[c] or hw_only_cls[c] > hi[c]
        n_out += out
        print(f"{classes[c]:<22} {hw_only_cls[c]:>4} {lo[c]:>6} {hi[c]:>6}  "
              f"{'YES' if out else ''}")
    print(f"落在 CI 外的类别数: {n_out}/10 "
          f"(期望 ~0.5, ≥2 才有分布结构证据)")

    print("\n=== 混杂排除: hw-only 错误的序列位置 (漂移/批次伪影?) ===")
    # 若 Forest hw-only 错误集中在序列某段, 可能是静态漂移而非类耦合
    idx_hwo = np.where(hw_only)[0]
    bins = np.arange(0, len(labels) + 1, 100)
    hist_all, _ = np.histogram(idx_hwo, bins=bins)
    idx_forest = np.where(hw_only & (labels == classes.index("Forest")))[0]
    hist_f, _ = np.histogram(idx_forest, bins=bins)
    print("位置分桶 (每 100 样本): hw_only_total / Forest_hw_only")
    for b in range(len(hist_all)):
        print(f"  [{b*100:>4},{b*100+100:>4}): {hist_all[b]:>2} / {hist_f[b]:>2}")
    print(f"Forest hw-only 样本下标: {sorted(idx_forest.tolist())}")
    # 置换检验: 34 个 hw-only 位置中 ≥16 个落在 Forest 标签位置的概率
    forest_pos = np.where(labels == classes.index("Forest"))[0]
    B2 = 20000
    cnt = 0
    all_pos = np.arange(len(labels))
    for b in range(B2):
        samp = rng.choice(all_pos, size=n_hw_only, replace=False)
        if np.isin(samp, forest_pos).sum() >= hw_only_cls[classes.index("Forest")]:
            cnt += 1
    print(f"置换检验 P(≥16/34 落在 Forest) ≈ {cnt/B2:.6f} (B={B2})")

    print("\n=== 机制证据: margin 脆弱 vs 方向耦合 ===")
    srt = np.sort(fake, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    resid = hw - fake
    print(f"{'class':<22} {'margin_p10':>10} {'margin_med':>10} {'resid_std':>9}")
    for c in range(C):
        m = fake_ok & (labels == c)
        print(f"{classes[c]:<22} {np.percentile(margin[m], 10):>10.3f} "
              f"{np.median(margin[m]):>10.3f} {resid[labels == c].std():>9.3f}")
    print(f"全局 resid_std: {resid.std():.3f}")
    fi = classes.index("Forest")
    fh = (labels == fi) & fake_ok & hw_err
    rest = (labels == fi) & fake_ok & ~fh
    print(f"Forest hw-only 样本 margin: {np.round(np.sort(margin[fh]), 3).tolist()}")
    print(f"Forest 其余正确样本 margin p10/med: "
          f"{np.percentile(margin[rest], 10):.3f} / {np.median(margin[rest]):.3f}")


if __name__ == "__main__":
    main()
