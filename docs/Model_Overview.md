# 模型总览（Model 1/4/5/6/7/8/9/10）— Optic-SpaceNet 决赛

> 队伍 CICC1003564 · 更新：2026-08-17（**真机全量验证收官**）
> 口径：QAT clean = v5 输出噪声（r=0.0392）；EuroSAT test 5400（eurosat_split, seed 42）

## 一、模型清单（最终结果 = 训练 + 真机全量）

| 模型 | 设计 | 参数量 | 计算量 | clean test | v8 test | **真机全量 5400** | 状态 |
|---|---|---|---|---|---|---|---|
| **Model 1**<br>Baseline VGG | VGG 风格（flat+BN），7 光计算层（6×3×3 Conv + fc1/fc2）；变体 A：conv1_1 电计算 | 2.39M | **156.6M** MACs/张 | v4.1 val A 97.98% | — | **20/20 = 100%**（板端抽样，全量 94s/张不可行）| ✅ 抽样验证完成（08-17）|
| **Model 4**<br>MiniVGG-GAP | stem(3×3,s2) FP32 电算 → 3 stages × 2×(3×3) → GAP → Linear(96,10)；7 光算层 | 260K | **17.03M** MACs/张 | test 96.17% | — | **94.19%**（5086/5400，gap −1.55）| ✅ 全量完成（08-11）|
| **Model 5**<br>J1-RF+ | stem k5 + stage2 3×3×2（RF 49px），全 MaxPool，C0=16 | 100,250 | **5.31M** | 96.81% | 96.65% | **90.17%**（gap −6.3，3×3 层瓶颈）| ✅ 全量完成（08-11）|
| **Model 6**<br>纯 J1 | 全 1×1 (1,2,2)，RF 17px，≤2M 预算 | 50,330 | **1.38M** | 96.39% | 95.22% | **92.78%**（5010/5400，gap −2.7）| ✅ 全量完成（08-11）|
| **Model 7**<br>J1-w075 | 宽度 ×0.75（C0=12），全 1×1，**MaxPool 下采样** | 32,246 | **0.86M** | 95.56% | 95.00% | **81.7%**（板端 [0:1800]，深诊断后暂停）| ❌ **架构不匹配（最终判定）**：MaxPool 噪声有偏传播，与 M9（同宽 conv3s2）差 12.1pt |
| **Model 8**<br>rf_stem5 | J1 + stem k5（C0=16），全 1×1 | 51,098 | **2.16M** | 96.17% | 96.26% | **87.78%**（gap −8.0，20h+ 连续使用后时段）| ✅ 全量完成（08-12）|
| **Model 9**<br>J1-w075ds3 | w0.75 + **conv3s2 下采样**（C0=12 + 可学习下采样）| 54,886 | **1.52M** | — | 95.87/95.69 | **94.43%**（canonical 完整复跑，301 错，gap −1.44 vs GPU QAT 95.87）| ✅ 全量完成（08-16）· **最终交付** |
| **Model 10**<br>ds3pool3 | conv3s2 下采样 + stem max3 | 96,602 | **2.56M** | — | 96.76/96.56 | **95.33%**（5400 张，gap −1.43 vs GPU QAT 96.76）| ✅ 全量完成（08-16）· **最终交付 · 全模型真机 SOTA** |

> 参数量/MACs 取自训练日志实测：M5 `100,250 / 5,309,696`；M6 `50,330 / 1,377,536`；
> M7 `32,246 / 861,440`；M8 `51,098 / 2,163,968`（logs/m{5,6,7,8}_*.log 启动行）。
> M4 逐层合计 ≈17.03M（模型头注释 ~17M）。
> gap 口径（本表）：M4-M8 = 真机 − 同量化 numpy 干净参考（M4 精确 95.74；M5/M6/M8 分段近似）；M9/M10 = 真机 − GPU QAT 部署 seed42（95.87/96.76）；M1 = 100% − QAT int8 val 97.87。C2 同窗 1000 张抽样（94.90/96.40）非全量口径。

## 二、M4-M10 结构细节（逐层 MACs）

> 全部为光计算第一性原理设计：模型尺寸无限、激活计算量（总 MACs）高度受限；
> 快速下采样到低分辨率 + 低分辨率宽层；conv bias=False；MaxPool 零 MACs（R6 证实 max 最优）。

### M4 — MiniVGG-GAP（≈17.03M MACs, ~260K params）

```
stem   Conv(3→32, 3×3, s2) + BN+ReLU + MaxPool      64×64 → 32×32 → 16×16   0.88M
stage1 Conv(32→48,3×3) → Conv(48→48,3×3) + MaxPool  16×16 → 8×8             3.54M + 5.31M
stage2 Conv(48→72,3×3) → Conv(72→72,3×3) + MaxPool  8×8 → 4×4               1.99M + 2.99M
stage3 Conv(72→96,3×3) → Conv(96→96,3×3)            4×4                     1.00M + 1.33M
head   GAP → Linear(96,10)                                                    0.001M
```
- 光计算层 7 个：stage1.0 / stage1.3 / stage2.0 / stage2.3 / stage3.0 / stage3.3（6×3×3 conv）+ head Linear(96,10)；
  **stem FP32 电计算**（训练 MODEL4_FIRST_CONV_FP32=1 默认，与部署 keep_first_conv_electronic 一致；retrain_v41_m4.log 确认）
- v4.1 语义：激活 TIA/ADC 噪声 + uint8+zp 量化；head FC 有 bias（光算时反量化后加回）
- 参数：conv 258,336 + BN 928 + head 970 ≈ 260K

### M5 — J1-RF+（5.31M MACs, 100,250 params）⭐ 中高档

```
stem   Conv(3→16, 5×5, s2) + MaxPool                64×64 → 32×32 → 16×16   1.23M
stage1 1×1 Conv(16→32) + MaxPool                    16×16 → 8×8             0.13M
stage2 3×3 Conv(32→64) → 3×3 Conv(64→64) + MaxPool  8×8 → 4×4               1.18M + 2.36M ← RF 49px 核心
stage3 1×1 Conv(64→128) → 1×1 Conv(128→128)         4×4                     0.13M + 0.26M
head   GAP → Linear(128→10)                                                    ~0.01M
```
- kernels [1,3,1]：唯一在低分辨率 8×8 上放 3×3 的模型（R6/R8 RF 路线，R8 160ep 96.93 为全谱最优）；C0=16 保持

### M6 — 纯 J1（1.38M MACs, 50,330 params）

```
stem   Conv(3→16, 3×3, s2) + MaxPool                64×64 → 32×32 → 16×16   0.44M
stage1 1×1 Conv(16→32) + MaxPool                    16×16 → 8×8             0.13M
stage2 1×1 Conv(32→64) → 1×1 Conv(64→64) + MaxPool  8×8 → 4×4               0.13M + 0.26M
stage3 1×1 Conv(64→128) → 1×1 Conv(128→128)         4×4                     0.13M + 0.26M
head   GAP → Linear(128→10)                                                    ~0.01M
```
- 全 1×1、深度 (1,2,2)（fast_downsample：stage1 单层）、RF 17px
- R7 决赛口径 7 个候选全部未超 J1（96.30）；head256 在 160ep 是负优化 −0.23pt，已剔除
- clean 正身 = 官方 r3_J1_long ckpt（复评 96.39%）；作废记录见 `logs/m5_m8_summary.md`

### M7 — J1-w075（0.86M MACs, 32,246 params）
- 结构与 M6 相同，仅 channels 全 ×0.75：**[12, 24, 48, 96]**（C0=12），stem k3 s2，head [128]
- 微型档；R6 80ep 95.30 / R8 170ep 95.56 → 本队 clean 160ep 95.56（复现）

### M8 — rf_stem5（2.16M MACs, 51,098 params）
- 结构与 M6 相同，仅 **stem 换 5×5**（C0=16 保持）：stem 占 1.23M，其余 1×1 层共 ~0.93M
- R6 效率第一（0.67pt/M）；R7 证实 C0≤14 + stem5 会崩，故必须 C0=16（不用 M7 的 C0=12）

### M9 — J1-w075ds3（1.52M MACs, 54,886 params）≤2M 预算新冠军（X0）

```
stem   Conv(3→12, 3×3, s2) + MaxPool                     电算 (stem_fp32)
stage1 1×1 Conv(12→24) → 3×3 s2 Conv(24→24)             16×16 → 8×8     0.07M + 0.33M
stage2 1×1 Conv(24→48) → 1×1 Conv(48→48) → 3×3 s2       8×8 → 4×4       0.07M + 0.15M + 0.33M
stage3 1×1 Conv(48→96) → 1×1 Conv(96→96)                4×4             0.07M + 0.15M
head   GAP → Linear(96→128) → Linear(128→10)            电算 (head_fp32)
```
- channels [12,24,48,96]（w0.75）+ stage1/stage2 两处下采样点换 **conv3s2**（3×3 s2 光算层，各 0.33M）；
  光计算层 **7**：stage1.0 / stage1.3.0 / stage2.0 / stage2.3 / stage2.6.0 / stage3.0 / stage3.3（5×1×1 + 2×3×3s2）
- X0 B1：conv3s2 一份 MACs 同时答"下采样 + RF + 通道混合"三问，v8 口径效率 0.55 pt/M；
  零成本 RF（pool3 max3×3）无收益 → RF 必须可学习才兑现（R6 rf_s2k3 互证）

### M10 — ds3pool3（2.56M MACs, 96,602 params）v8 总冠军（X0）
- 结构同 M9，channels 回全宽 **[16,32,64,128]** + **stem 池化换 MaxPool 3×3 s2**（stem_pool_mode=max3，电侧零 MACs）
- 光计算层同 M9（7 层）；v8 口径总冠军（96.76/96.56），2.5–4.6M 甜点区代表；
  严格支配 rf_s2k3（4.52M, 96.39）与 M5（5.31M, 96.61）——更便宜更准

## 三、架构谱系（MACs-精度，真机全量口径，2026-08-17 终版）

```
0.86M  M7  (板端 81.7%)                    ← ❌ MaxPool 噪声有偏传播（判定关闭）
1.38M  M6  (真机 92.78%)                  ← 全 1×1 稳健
1.52M  M9  (真机 94.43%)                  ← ≤2M 冠军 · 最终交付
2.16M  M8  (真机 87.78%)                  ← 受时段影响
2.56M  M10 (真机 95.33%)                  ← 全模型真机 SOTA · 最终交付
5.31M  M5  (真机 90.17%)                  ← 3×3 层瓶颈
17M    M4  (真机 94.19%)                  ← 复赛主力延续
156.6M M1  (板端抽样 20/20=100%)          ← 大 VGG 基准
```

- **真机排序**：M10 95.33 > M9 94.43 > M4 94.19 > M6 92.78 > M5 90.17 > M8 87.78 > M7 81.7（关闭）
- **决定性发现（M7 vs M9 对照）**：同 C0=12、同通道同层形，唯一差异 MaxPool vs conv3s2 光下采样 → 差 12.1pt（M9 94.43% vs M7 修复校准后最佳 82.33%；三段累计 [0:1800] 81.72% → 差 12.7pt）。
  三重验证（板端 FAKE 95.04% corr 0.9998 / 权重 md5 一致 / probe v2 深层 SNR 10-25）确认部署链正确：
  **MaxPool 取 max 对加性噪声有偏传播（选择噪声尖峰）；卷积光层线性平滑噪声**
- v8 训练口径关键反转（X0）：w200 反超 rf_s2k3；conv3s2 家族（M9/M10）全面压制 MaxPool 家族；
  MACs 效率 dsconv3 = 0.55 pt/M

## 四、设计依据（R2-R8 架构探索结论）

- **M5** = rf_s2k3（R8: 160ep 96.93，RF 路线长训持续涨）+ stem k5（R6: MACs 效率第一 0.67pt/M）；C0=16 保持（R7: C0≤14 即崩）
- **M6** = 纯 J1（R7 决赛口径：7 个候选全部未超 J1 96.30；**head256 在 160ep 是负优化 −0.23pt，已剔除**）
- **M7** = w075（R6/R8: 0.86M 微型档 95.30@80ep / 95.56@170ep）
- **M8** = rf_stem5@C0=16（R6 效率第一改动；避免 w075 的 C0=12 + stem5 组合——R7 证实 C0≤14 崩）
- **M9** = w075ds3（X0 B1）：J1 w0.75 + stage1/2 下采样点换 **conv3s2**；pool3（max3×3）零成本 RF 无收益（−0.14），RF 必须可学习
- **M10** = ds3pool3（X0 B1）：M9 通道回全宽 + **stem max3**；v8 总冠军，严格支配 rf_s2k3/M5
- 共性：全 MaxPool（R6: avg/patchify 均亏）、深度 (1,2,2)（R7: 恰好）、无全局旁路（R6 Q3 证伪）、SGD+Momentum（R1: 最优）；X0 增补：blurpool/pool4 关闭（v8 负收益）

## 五、量化与上板（v8 漂移鲁棒 QAT）

- **真机误差归因**（round5 §7/§8）：hw 误差 95% 是 run 间可复现的**结构化分量**（per-column 偏移 4-23% + per-column 增益 1-3% + per-element δW 21-50%），iid 噪声模型过悲观（差 6.4pt）
- **v8 配方**（c3d 冠军，真机 93.80%，gap 1.8pt）：60ep 从 clean ckpt 初始化，lr 0.01 + warmup 2ep，SGD，LS 0.05；训练时 per-batch 重采样三组分 + iid 260，幅度取板上 probe 实测 × **1.5 余量**（col_off 396–1308 / col_gain 0.018–0.038 / δW rms 5.6–10.9）
- M5-M8 v8 全部完成：M5 96.65（−0.16pt 几乎无损）、M6 95.22（官方 J1 init）、M7 95.00、M8 96.26（唯一反超）
- 部署链路（M5-M8）：stem/head FP32 电计算 + **5 光计算层**（stage1.0 / stage2.0 / stage2.3 / stage3.0 / stage3.3）；head bias 保留（C2 部署 bug 教训）；M4 为 stem FP32 电算 + 7 光计算层（v4.1）
- **M9/M10（X0 单阶段 v8 160ep，双 seed 42/43）**：M9 95.87/95.69、M10 96.76/96.56；C2 上板评测进行中
  （放行判据全过，probe → calib ×2 → col 跑批 ×2 同窗口；结果补 `x0/results/C2_ds3_board.md`）
- M9/M10 部署：板端 numpy runner `x0/scripts/run_ds3_gazelle.py`（**7 光算 conv**：s1a/s1ds/s2a/s2b/s2ds/s3a/s3b + h1/h2 光算可 `DS3_HEAD_ELEC=1` 电算；m≤2 tiling；`DS3_CALIB_COL` 逐列折叠同 C1 补丁）；权重导出 `x0/scripts/export_ds3.py`
- 上板配置对照与缺口清单：`docs/Board_Deploy_Config.md`（M1/M4/M5-M8 vs calibrate_col）
- 真机预期：M5 ~95%（推算）、M6 ~93.8%（c3d 同架构实测）

## 六、关键文件

- 训练代码：`train-test/eurosat_research/`（共用 `src/runner.py` + 变体 `configs/m{5,6,7,8}_*.json`、`x0*_160.json`）
- 权重：`train-test/eurosat_research/weights/`（M5-M10 v8 全部入库；M1/M4 在 `train-test/weights/`）；v8 权重同步于 `train-test/weights/`（部署统一位置）
- 日志：`train-test/logs/m{5,6,7,8}_*.log` + `runs/*/metrics.jsonl`（每 epoch 结构化）；X0 训练在容器 gazelle_sim
- 汇总：`train-test/logs/retrain_v41_summary.md`（M1-M4）、`train-test/logs/m5_m8_summary.md`（M5-M8）、`x0/results/pareto_v8.md`（X0 单阶段）+ `M_validate.md`（M5-M8 独立复测）
- 探索记录：`train-test/eurosat_research/docs/round{2,3,6,7,8}_notes.md`、`round5_c2_drift_robust.md`、`round_x0_arch_hw_codesign.md`（X0 总览）
- 帕累托图：`train-test/eurosat_research/docs/plot_perf_vs_macs.py`（QAT clean）+ `plot_pareto_v8.py`（v8 口径，含 M9/M10）+ `plot_pareto_hw.py`（真机）

## 七、待办（2026-08-17 收官状态）

- [x] M4/M5/M6/M8 真机全量 5400 完成（94.19/90.17/92.78/87.78）
- [x] M9/M10 真机全量完成（**94.43 / 95.33**，最终交付模型，canonical 复跑口径）
- [x] M7 判定关闭：板端 81.72% + 三重验证（FAKE/md5/probe-SNR）→ **MaxPool 噪声有偏传播**（vs M9 conv3s2 差 12.1pt = 94.43 − 82.33 修复校准后最佳）
- [x] M1 板端抽样 20/20=100%（全量 94s/张不可行）
- [x] 器件调度硬规则固化（AGENTS.md 第 9 条：单次调用 <30min / 冷却 ≥5min / 热崩溃 ≥1h 恢复）
- [ ] 帕累托真机曲线更新（plot_pareto_hw.py，含 M9/M10）
- [ ] 错误样本混淆分析出图（err npz 已全量入库）
- [ ] 决赛答辩材料（讲解 PPT 大纲）
