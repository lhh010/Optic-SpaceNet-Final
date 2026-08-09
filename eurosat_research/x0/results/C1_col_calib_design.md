# C1 逐列校准（per-column calibration）设计文档

Round X0 · C1-prep · 2026-08-09 · 状态：代码 + 本地预检完成，未上板

## 1. 背景与动机

现有部署链路对每层光计算输出做 **per-layer 标量校准**：`ideal = (hw − β)/α`，
(α, β) 由 `np.polyfit(ideal.ravel(), hw.ravel(), 1)` 拟合——列（输出通道）维度被压平。
C3 probe 残差分解（round5 文档 §6-§8）表明标量校准后的残差含确定性、run 间可复现的
**per-column 偏移（std 264–872 raw counts，占残差方差 4–23%）** 与
**per-column 增益（std 1.2–2.5%，占 1–3%）**。逐列校准即对每列单独拟合
`hw[:,c] ≈ α_c·ideal[:,c] + β_c`，把这部分结构化残差消掉。

## 2. 数学

### 2.1 逐列最小二乘

对每层（n 行 pairs × m 列），每列 c 独立一维回归：

- α_c = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²，β_c = ȳ − α_c·x̄，x=ideal[:,c]，y=hw[:,c]
- 估计标准误：SE(α_c)=√(s²/Sxx)，SE(β_c)=√(s²·(1/n+x̄²/Sxx))，s²=Σr²/(n−2)

列结构真实性判据（脚本内打印）：跨列 std(α_c)、std(β_c) 应远大于估计标准误中位数
（SNR >> 1），否则逐列参数只是在拟合测量噪声。实测 SNR：α 3–23，β 7–51，全部成立。

### 2.2 留出验证（防过拟合自检）

pairs 行按固定种子 50/50 分半：前半拟合（标量与逐列各拟合一份），后半评估残差 std。
只有留出数据上逐列残差 std 显著小于标量，逐列 calib 才真有效。实测见 §4。

### 2.3 部署折叠（零额外算子）

原推理路径（标量 calib）：

```
y_corr = (y_hw − β)/α                     # calib_correct
y_f    = x_scale·ws_c·(y_corr − x_zp·col_sum_c)   # 反量化
```

逐列 calib（α_c, β_c 为 (1, C_out) 向量）代入并整理，得**反量化折叠形式**（patch 实现）：

```
y_f = x_scale·(ws_c/α_c)·(y_hw − β_c − α_c·x_zp·col_sum_c)
```

即只把原有 per-channel 量换成 `ws'_c = ws_c/α_c`、`off_c = α_c·x_zp·col_sum_c + β_c`，
乘加次数与标量路径完全相同（已用随机数据数值验证，与"先 calib 再反量化"逐步路径
最大偏差 2e-15，机器精度）。

等价地，若令未校准反量化输出 z_c = x_scale·ws_c·(y_hw − x_zp·col_sum_c)，
则 y_f = (z_c − o_c)/α_c（o_c = x_scale·ws_c·β_c），代入后续 BN
`(y−μ)/√(v+ε)·γ+b` 得 **BN 折叠形式**：

```
μ'_c = o_c + α_c·μ_c          （等效 running_mean）
v'_c = α_c²·(v_c+ε) − ε       （等效 running_var）
γ, b 不变
```

部署采用反量化折叠而非 BN 折叠，原因：(a) h1/h2 FC 层后无 BN，反量化折叠对
conv/FC 统一；(b) 不改动 BN 参数数组，回退路径清晰。

## 3. 实现

- **`Gazelle-national/mnist/j1_board/calibrate_col.py`**（新增）：读 probe pairs
  （`{PREFIX}{layer}_{ideal,hw}.npy`），逐列拟合 + SE + 结构 SNR + 留出验证，
  输出 `calib_col.json`。全环境变量驱动（CALIB_COL_PAIRS_DIR / CALIB_COL_OUT /
  CALIB_COL_LAYERS / CALIB_COL_SCALAR / CALIB_COL_PREFIX / CALIB_COL_HOLDOUT），
  纯 numpy，板上 Python 3.6 可直接跑，也可本地预检。
  输出格式：`{layer: {alpha:[...], beta:[...], col_resid_std:[...], scalar_alpha,
  scalar_beta, n_samples, se_*_median, col_struct_snr_*, holdout:{...}}}`。
- **`Gazelle-national/mnist/j1_board/run_j1_gazelle_colcalib.patch`**（新增）：
  对 `run_j1_gazelle.py` 的 git-style patch（`patch -p1` 从仓库根应用，已验证可干净
  应用 + py_compile 通过）。新增 `J1_CALIB_COL` 环境变量；设置后该层走逐列折叠，
  json 中缺失的层自动回退标量 calib，未设置变量则行为完全不变（后向兼容）。
- **FAKE 自检**：`_load_calib_col` 与标量 calib 一样有 `not FAKE` 守卫，FAKE 模式
  不加载任何 calib。实测：patched 版 + `J1_CALIB_COL=calib_col_local.json` +
  `J1_FAKE=1 J1_LIMIT=32` 与原版 FAKE 结果逐位一致（均 FINAL 100.00%）。

## 4. 本地 pairs 实测（零板上成本预检）

数据：`x0/data/probe_pairs/`（2026-08-08 从板上取回，weights_c2c 工作点，
s1a–s3b 五层，1024–16384 行/层；h1/h2 无 pairs）。标量对照 `calib_c2c.json`。

留出验证（前半拟合 → 后半评估，残差 std raw counts）：

| 层 | n | 标量 resid_std | 逐列 resid_std | std 改善 | **方差改善** | β_c 跨列 std (SE) | α_c 跨列 std (SE) |
|-----|-------|--------|--------|---------|----------|------------|------------|
| s1a | 16384 | 753.3  | 688.9  | −8.5%  | **−16.4%** | 305.2 (5.9)  | 0.0251 (0.0011) |
| s2a | 4096  | 1135.9 | 1016.5 | −10.5% | **−19.9%** | 586.9 (24.7) | 0.0240 (0.0028) |
| s2b | 4096  | 1390.0 | 1300.2 | −6.5%  | **−12.5%** | 480.0 (26.5) | 0.0120 (0.0016) |
| s3a | 1024  | 1799.7 | 1556.9 | −13.5% | **−25.2%** | 979.0 (73.4) | 0.0157 (0.0028) |
| s3b | 1024  | 1346.4 | 1315.4 | −2.3%  | **−4.5%**  | 336.2 (47.3) | 0.0126 (0.0040) |

结论：

- 五层留出方差改善 **4.5%–25.2%**，与 C3 分解预测的列结构占比（5–26%）精确吻合——
  逐列 calib 把预测的确定性列结构基本全部回收，不是过拟合（留出验证 + SNR>>1 双重证据）。
- s3b 改善最小（−4.5%），与其分解中列结构占比最低一致；留出验证仍为严格正收益，无回退风险。
- 预期真机收益：残差方差整体降 ~5–25%/层，按 C3 噪声-精度敏感性外推 **+零点几~1pt**，
  以板上背靠背跑批为准。

## 5. 板上执行 SOP（草案）

1. `sudo compass_cali` 新鲜校准（EBR ≥ 8），MNIST canary 确认板状态正常。
2. （如新权重/新窗口）`PROBE_ROWS=100000 python3 probe_dump.py` 重采 pairs
   （5 层 ×10 万行，m≤2 tiling，约 15–30 min；h1/h2 如部署光计算 head 需补 dump）。
3. `python3 calibrate_col.py`（板上直接跑，<1 min）→ `/home/uisrc/j1/calib_col.json`；
   检查打印的结构 SNR（应 >>1）与留出改善（应显著为正）。
4. 背靠背跑批：先 `J1_CALIB=<标量>` 基线，立即 `J1_CALIB=<标量>
   J1_CALIB_COL=/home/uisrc/j1/calib_col.json` 逐列，同日同窗口对比。
5. 归档：calib_col.json + 两次 run log 取回 `x0/results/`。

calib 增加的板上时间预算：重采 pairs 15–30 min（若无新鲜 pairs）+ 拟合 <1 min +
第二次跑批（与基线相同耗时）。若复用现有 pairs，仅 +1 次跑批时间。

## 6. 风险与后续

- **跨窗口稳定性**：列参数是否随 calib 窗口漂移未知——需同一权重两次独立
  probe_dump + calibrate_col，比较 (α_c, β_c) 相关/差分 std（后续可选项，
  脚本输出含全部参数可直接 diff）。若漂移显著，逐列 calib 须与跑批同窗口采集。
- **h1/h2 无本地 pairs**：本地预检未覆盖 head；板上 SOP 第 2 步按需补采。
- **s3b 边际收益**：改善 −4.5% var 最小，若板上时间紧张可不重采其 pairs
  （复用现有即可），缺失层自动回退标量 calib，混合使用安全。
- pairs 与部署权重必须同源：本地这份是 weights_c2c 工作点，部署 c3d/J1 前
  需用对应 weights 重采（probe_dump 走 J1_WEIGHTS_DIR 环境变量）。
