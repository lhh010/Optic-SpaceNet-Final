# Round X0 — 首轮架构-硬件联合设计

日期：2026-08-09 · 状态：**主体完成**（M9/M10 上板评测进行中）· 性质：首个**架构 × 硬件噪声联合设计**轮次
背景：R1-R8 完成"clean 口径架构搜索"（J1 = ≤2M 局部最优），C1-C3 完成"固定架构上的噪声归因与组分 QAT"（c3d hw 93.80%，gap 1.8pt）。X0 把两条线合并：架构决策与噪声建模/校准**联合**优化，而非串行贪心。

## 0. 理论框架（本轮的三个第一性原理）

1. **GAP 噪声滤波不对称性**：GAP 平均空间维 → iid 快噪声被 √N 抑制，但 per-column 结构化噪声在同通道所有空间位置共享 → **平均不掉**。BagNet+GAP 架构滤掉 iid 后，瓶颈天然是列结构噪声——c3d 的成功与此自洽。
2. **δW 是绝对 count 扰动**（probe 实测 3.7-7.2 int8 counts）：小权重相对误差被不成比例放大 → 光计算偏好"列内幅度平坦/顶满"的权重分布，与 weight decay 的重尾效应冲突；同时 y = Σ(w+δw)x 中 iid δW 按 1/√cin 稀释 → **宽度是 δW 的天然平均器**（与 GAP→iid 同数学，平均轴从空间换成输入通道）。
3. **绝对底噪 → 值域利用率 = SNR**：hw 噪声 σ≈4.5 counts 与信号幅度无关（crossval），顶满量程严格改善 SNR；per-tensor scale 被离群点绑架 → percentile clipping 白捡有效位数。

**下采样算子家族**（RF 贡献 (k_ds−1)·j，节奏 = stride 计划表）：

| 算子 | k | 窗内函数 | MACs | X0 结论 |
|---|---|---|---|---|
| MaxPool 2×2 | 2 | max | 0 | J1 现役基线 |
| patchify+1×1 | 2 | 线性 | 4C²·HW/4 | R6 已测负（只证伪 k=2 线性，**未证伪 shuffle 家族**） |
| stride 1×1 | 1 | 抽取（混叠） | C²·HW/4 | R6 灾难 −10.45 |
| MaxPool 3×3 s2 | 3 | max | **0** | X0 实测**中性**（95.43 vs J1 95.57） |
| BlurPool 3×3 s2 | 3 | 二项式低通 | 0（电侧） | X0 实测**负**（94.31，关闭） |
| **3×3 conv s2** | 3 | 线性可学习 | 9C²·HW/4 | **X0 最大赢家**（见 §1-B1） |

## 1. 执行结果与结论

### A 组 · 噪声结构分析（纯本地，probe pairs）

- **A1（δW 结构）**：宽度稀释 δW 证伪——深层 γ≈0.45-0.5 无稀释；浅层 s1a 存在 tile 块共变（21×）。v10 噪声模型建议：浅层加块相关+低秩，深层维持 iid。
- **A2（非线性）**："确定性非线性"是 RFF 过拟合伪影（8 模型留出 R²≤0.005）→ v10 删 RFF，改加 tile 行对随机共模（corr 0.12-0.22）；iid σ 应从部署残差反推。
- **A3（权重分布）**：δW 绝对扰动证实（斜率 0.08-0.23）；列级主变量是信号幅度/SNR 而非分布形状 → B2 只剩 wd 扫描（动机=SNR）。
- **A4（per-class）**：Forest 一类占 hw-only 错误 47%（logit 残差 std 0.826 全局最大），混淆对 Forest→Pasture；修复上限≈95.4%。

### B1 · 下采样家族（X0 单阶段 v8 160ep 口径，对照 x0_ctrl=J1 95.57）

| 臂 | MACs | test | 结论 |
|---|---|---|---|
| pool3（max3×3 s2 全程） | 1.378M | 95.43 | 中性 |
| blurpool | 1.378M | 94.31 | 负，关闭 |
| pool4（第 4 次 pool，j=32） | 1.378M | 94.81 | 负，关闭 |
| dsconv3（conv3s2 下采样） | 2.557M | 96.22 / 96.67（双 seed） | 确认有效 |
| **w075ds3**（w0.75+conv3s2，**≤2M 预算内**） | **1.522M** | **95.87 / 95.69** | **≤2M 新冠军 → M9** |
| **ds3pool3**（conv3s2+stem max3） | **2.557M** | **96.76 / 96.56** | **总冠军 → M10** |

判读：space-to-depth/shift 家族未被证伪（R6 只杀了 k=2 线性 patchify）；**每步降采样引入的感受野**是关键自由度，conv3s2 用一份 MACs 同时答下采样+RF+通道混合三问。

### B3 · v8 口径 Pareto（6 个 R 系模型单阶段重测 + M5-M8 join）

前沿（含 M 系，M 系为独立复测值）：**M7 94.98 (0.86M) → J1 95.57 (1.38M) → w075ds3 95.78 (1.52M) → M8 96.20 (2.16M) → ds3pool3 96.66 (2.56M) → w200 96.74 (4.62M) → Model4E 96.81 (17M)**

- v8 口径整体比 v5 低 0.3-0.9pt，排序基本保持；**重要反转：w200 (96.74) 反超 rf_s2k3 (96.39)**——v5 口径"RF 路线长训续涨"在 v8 噪声下不成立，rf_s2k3 被 2.56M 的 ds3pool3 严格支配。
- **J1 已非甜点**：+10% MACs 的 w075ds3 与 +57% 的 rf_stem5 均更优。
- **v8 噪声下规模收益急剧饱和**：Model4E 17M 仅比 ds3pool3 2.56M 高 0.15pt；甜点区间 2.5-4.6M。
- 图：`docs/plot_pareto_v8.png`（脚本 `plot_pareto_v8.py`）；数据表 `x0/results/pareto_v8.md`。

### C1 · 逐列校准（per-column calib）—— 真机 SOTA

同窗口 ABA 背靠背（各 1000 样本）：scalar 92.60 / **col 94.60** / scalar′ 93.60 → **+1.5pt，真机历史最佳 94.60%**（旧 c3d 93.80）。错误级分析修复/新增比 ~3:1，非噪声抖动。详见 `x0/results/C1_board_results.md`。

### C2 · M9/M10 上板（**进行中**）

w075ds3/ds3pool3 已导出板端权重包，新 runner（`run_ds3_gazelle.py`，支持 conv3s2 光计算 + m≤2 tiling + 逐列 calib 折叠）本地 FAKE 对拍通过（corr 0.9999，pred 一致 99.9%，无 head bias/stem 坑）。放行判据全过，同窗口 probe→calib→跑批进行中。结果将补入 `x0/results/C2_ds3_board.md`。

## 2. 新上板方法论（X0 沉淀的部署 SOP）

1. **他人使用侦测**：跑批前 `ps aux | grep -i gazelle` + 器件占用检查；发现他队进程不对抗、不动对方文件，等释放。
2. **四项放行判据**（fresh `compass_cali` 后全过才开跑）：EBR≥8 + error_std 相对基准 ±2%（或 <10% 宽限）+ MNIST canary + 200 样本 mini-run。
3. **同窗口背靠背**：probe pairs → calibrate_col → 跑批必须在同一 fresh cali 窗口连续完成；calib json **不可跨窗口复用**（列结构占比跨窗口波动，w1 vs w2 留出改善 −24.4% → −7.8% 实证）。
4. **抢占处置**：保存现场（scp 回 `x0/data/incident_<date>/`）→ 等物理恢复（实测 ~40-75min 瞬态自行恢复）→ 重走放行判据 → 重做全部实验；被污染窗口数据标记不采信。
5. **部署链路数值自检**：新 runner 上板前本地 FAKE 对拍（logits corr + pred 一致率 + stem 逐位），防 head bias 丢失/stem 不一致（C2 轮历史坑）。
6. **铁律沿用**：部署脚本禁位置参数（compass_sdk 篡改 sys.argv，一律 env）；sudo 用 `sudo env VAR=...`；m≤2 tiling（FPGA 行回绕 bug）。

## 3. 模型注册表（eurosat_research/weights/）

| 模型 | 架构 | MACs | v8 口径 test | 权重文件 | 真机状态 |
|---|---|---|---|---|---|
| M5 | rf_s2k3+stem k5 | 5.31M | 96.61（两阶段，复测确认） | `m5_j1rf_stem5_v8probe15.pth` | 已注册 |
| M6 | J1 (head128) | 1.38M | 95.28（两阶段，复测确认） | `m6_j1_v8probe15.pth` | 已注册 |
| M7 | J1 w0.75 (C0=12) | 0.86M | 94.98（两阶段，复测确认） | `m7_j1w075_v8probe15.pth` | 未注册 |
| M8 | rf_stem5 | 2.16M | 96.20（两阶段，复测确认） | `m8_rf_stem5_v8probe15.pth` | 未注册 |
| **M9** | **J1 w0.75 + conv3s2 下采样**（≤2M 冠军） | 1.52M | 95.87/95.69（单阶段双 seed） | `m9_j1w075ds3_v8probe15.pth` | **上板评测中（C2）** |
| **M10** | **J1 + conv3s2 + stem max3**（总冠军） | 2.56M | 96.76/96.56（单阶段双 seed） | `m10_ds3pool3_v8probe15.pth` | **上板评测中（C2）** |

注：M5-M8 为队友两阶段协议（clean160→v8 60ep）权重，X0 已独立复测确认（报告−复测 ≤0.06pt，MACs 逐位一致，见 `x0/results/M_validate.md`）；M 系与 X0 同档架构差异均在种子噪声量级，不提供新信息量。M9/M10 为 X0 单阶段 v8 160ep、seed42 best ckpt。

## 4. 产物索引

- 分析：`x0/results/A1_dW_structure.md` / `A2_nonlinear_residual.md` / `A3_weight_dist.md` / `A4_perclass_errors.md`
- 训练：`x0/results/B1_arms_ready.md` / `B1_train_wave1.md` / `B1_train_wave3.md`；configs `eurosat_research/configs/x0_*.json`、`x0r_*_160.json`；架构旋钮 `src/models.py`（max3/blur/conv3s2/extra_pool/stem_pool_mode）
- Pareto：`x0/results/pareto_v8.md` + `docs/plot_pareto_v8.{py,png}`
- 上板：`x0/results/C1_col_calib_design.md` / `C1_board_results.md`；脚本 `x0/scripts/{export_ds3,run_ds3_gazelle,probe_dump_ds3,calibrate_any_ds3,check_fake_ds3}.py`；板端 `mnist/j1_board/{calibrate_col.py,run_j1_gazelle_colcalib.patch}`（Gazelle-national）
- 事件：`x0/results/C1_incident_analysis.md` / `C1_board_forensics.md`

## 5. 纪律（沿用 C3 方法论闭环）

- 架构对比必须 **160ep 决赛口径**（R7 head256 反转教训）
- proxy_v8 只粗筛；高 iid 端失真（c2j 教训）；选模以 hw 为准
- 禁止 iid 全量注入 resid_std（C2 训崩教训）；结构化≠iid（proxy 79.0% 反例）
