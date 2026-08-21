# Round 5 — C2 迭代：部署 bug 修复 + 漂移鲁棒 QAT（v6 split-noise）

日期：2026-08-07 · 目标：缩小 QAT(96.3%) → 真机(90.6%) 的 5.7pt gap

## 1. Code review 发现的部署链路 bug（round4 之后复查）

1. **head 两层 bias 丢失**：`export_j1.py` 只导出权重不导出 bias，`run_j1_gazelle.py` 的
   `optical_fc` 反量化后不加 bias（head Linear bias=True，h2 bias 参与 logits）。
   修复后 FAKE 96.50% → 96.70%。
2. **stem 训练/部署不一致**：`prepare_model_v5` "全层光计算无 stem 特判"，训练时 stem 也
   QAT（量化+噪声），部署却是电计算 float。v6 中 `stem_fp32=True` 对齐。
3. **QAT 噪声结构错误**：`inject_output_noise` 的 σ = ratio × 当前 batch y.std()，实为
   **信号相关相对噪声**；crossval 结论是真机为绝对加性底噪 + 慢漂主导。
4. **`is_last` 从未设置**：h2 logits 层训练部署都吃光噪声。
5. 交付清单与实物不符：`calibrate_j1.py`/`calib_j1*.json`/`weights_j1`/runs 均不在仓库
   （板上 `~/j1/` + 训练容器 gazelle_sim），本轮已归档本地 `mnist/j1_board/`。

## 2. 关键实证：iid 全量噪声模型过悲观 ~6.4pt

numpy 代理（`mnist/sim_noise_proxy.py`，5 个光计算 conv raw 域注入 iid N(0, resid_std)，
resid 来自 calib_j1_real.json，n=1000 × 3 seeds）：

| 权重 | clean | iid-full proxy |
|---|---|---|
| champion (v5, r3_J1_long) | 96.70 | 84.23 |
| c2b iid260 | 96.10 | 86.03 |
| **c2c split (iid260+gain2%+off500)** | **96.40** | **91.23** |
| c2d resid/3 | 96.00 | 84.43 |

- champion 真机 90.60%（round4）vs iid 代理 84.23% → **resid_std 里大部分是结构化分量**
  （静态 LUT 误差 = 确定性扰动 + 慢漂），网络对结构化扰动的容忍度远高于 iid。
- **v6 全量 iid 重训（sigma=resid_std）直接训崩**：best_val 90.37%（clean 口径损失 ~6pt）。
  教训：不能把 resid_std 当 iid σ 注入。
- **split-noise（c2c）**：iid 快噪声 260 raw（crossval E6 σ_dynamic≈1.02 counts）
  + 增益抖动 2%（慢漂 alpha）+ 共模偏移抖动 500 raw（慢漂 beta）。
  代理 +7.0pt，clean 基本无损。gain/offset 抖动训出的鲁棒性对 iid 同样泛化。

## 3. 真机结果（同日同一块板，fresh compass_cali + canary + per-layer calib）

| 配置 | hw 1000 |
|---|---|
| champion + 修复部署（bias + head-elec） | 88.70%（漂移主导：前 72 样本 94.4%，单调下滑） |
| **c2c split + 修复部署** | **91.20%** ← 新最佳 |
| c2j iid500 + 修复部署 | 89.20%（proxy 94.93 但 hw 更低——见下） |

- champion 88.70 < round4 的 90.60：非模型回退，是**跑批窗口漂移**（FAKE per-window 平直，
  hw 前段 94.4% → 后段 ~87%）。head-elec + bias 本身无害（FAKE 验证 96.70%）。
- 流程教训：evb/canary/calib 每一步都在消耗漂移预算；calib → 跑批必须背靠背。
- 漂移缓解备选：`J1_OFFSET` 分段跑批 + 段间重 calib（已实现，本轮未启用）。

## 3b. 二轮扫参（proxy 选优，hw 验证两点标定 proxy↔hw 映射）

| 变体 | 配方 | clean (torch test) | iid-full proxy |
|---|---|---|---|
| c2c | gain2%/off500/iid260, 60ep | 96.22 | 91.23 → **hw 91.20** |
| c2e | gain3%/off800/iid260 | 96.24 | 80.53（过强抖动崩） |
| c2f | c2c + 120ep | 96.50 | 83.83（长训丢 iid 鲁棒） |
| c2g | gain1.5%/off400/iid260 | 96.31 | 91.43 |
| c2h | gain2.5%/off600/iid260 | 96.39 | 84.20 |
| c2i | gain2%/off500/**iid400** | 95.37 | 93.87 |
| c2j | gain2%/off500/**iid500** | 93.72 | **94.93**（proxy>clean） |
| c2m | iid550 | 93.09 | 94.73（平台） |
| c2k | iid600 | 92.04 | 93.50（过峰） |

- proxy↔hw 标定点：champion 84.23→88.70（+4.5，champion 未训 iid 鲁棒，hw 结构化噪声
  比 iid 仁慈）；c2c 91.23→91.20（±0，训过 iid 鲁棒后 proxy≈hw）。模型越 iid-鲁棒，
  proxy 越准。
- **proxy 在高 iid 端失真**：c2j (iid500) proxy 94.93 全场最高，hw 仅 89.20。原因：
  重 iid 训练牺牲 clean（93.9 vs 96.4），而 hw 噪声以结构化/漂移为主，iid 鲁棒性
  用不上；clean 天花板降低 + 漂移下压 = 净亏。**选模应以 hw 为准，proxy 只用于粗筛。**
- iid 维度 proxy 甜点 500-550，但 hw 甜点在 260（c2c）；jitter 维度甜点 gain 1.5-2%/
  off 400-500；两个维度都过强即崩。

## 5. 结论与下一步

- **C2 最终结果：c2c split（gain2%/off500/iid260, 60ep, init_from champion）hw 91.20%**，
  较 round4 的 90.60% +0.6pt，较同日同窗口 champion 修复部署 88.70% +2.5pt。
  完整链路：FP32 96.65 → clean FAKE 96.40 → **hw 91.20**。
- 修复部署两件套的贡献被漂移掩盖（champion 88.70 < 90.60 是漂移运气差，非修复回退，
  FAKE 96.70 证实代码正确）。
- 剩余 gap（hw 91.20 vs clean 96.40 ≈ 5.2pt）归因实验（用户决策：不测漂移曲线，
  直接试药）：**`J1_OFFSET` 分段跑批 + 段间重 calib**。判读二分：
  - 分段后明显涨 → 跑批内漂移是主因 → 采纳分段 SOP + 试 test-time BN adaptation
  - 分段后不动 → iid + LUT 静态误差为主 → 多次 cali 采样 LUT 残差图谱做
    domain randomization 训练
- 后续叠加：c2c 邻域细扫（gain 1.5-2%/off 400-500/iid 260，每点 hw 验证）。
- 板端文件注意：`~/j1/weights_c2c` 被误覆盖为 c2j 权重（c2c 权重在
  `eurosat_research/runs/c2c_J1_split_8f8d7be7/best.pth` + `/tmp/weights_c2c_J1_split_8f8d7be7`，
  需要时重新 export 上传）

## 4. 产出物

- `eurosat_research/src/qat_v6.py`：QATConv2d_v6（per-layer raw 域噪声 + gain/off 抖动，
  stem/head FP32）+ `prepare_model_v6`；runner 支持 `qat_version=v6` + `init_from`
- `eurosat_research/configs/c2*.json`：v6 全量 iid（崩）/ iid260 / split / resid/3 四臂
- `run_j1_gazelle.py`：bias 修复 + `J1_HEAD_ELEC` + `J1_OFFSET`；`export_j1.py`：导出
  head bias + float head 权重；`sim_noise_proxy.py`：numpy 噪声代理基准
- 训练：gazelle_sim 容器 A800，`runs/c2*_*/`（c2c = c2c_J1_split_8f8d7be7）

## 6. 归因实验（2026-08-07 晚，c2c 权重）

**结论先行：跑批内"漂移"假说被证伪，剩余 gap 主因是噪声底本身 + 分钟级非平稳波动。**

| 实验 | 结果 | 判读 |
|---|---|---|
| 段间重 calib 探针对比（25 min 间隔，中间夹 500 样本重负载） | alpha 变化 ≤0.08%，beta 变化 <3%，resid_std 变化 <1% | **增益/偏移在 25 分钟尺度上稳定**，per-layer calib 没有可修正的漂移 |
| 分段跑批（500+重 calib+500） | 88.00 / 90.20，合计 **89.10%** | **低于单跑 91.20%**，分段重 calib 无收益，假说证伪 |
| 段内轨迹分析 | seg1 中段（136→344）真实窗口精度 ~83.6%，首尾 ~94/88.5；champion/c2c 单跑同样 ±3-4pt 摆动 | 精度在 ~5-10 min 尺度上**非平稳波动**，但探针（本身数分钟平均）看不到 |
| TTBN（batch 统计量替代 running stats） | FAKE 81.90% vs 96.40% | **排除**：B=8 batch 统计量被类别组成污染，且 per-layer calib 已是更优的同功能机制 |

- 新假设：剩余 gap = iid/LUT 噪声底 + 分钟级非平稳噪声（板载/主机环境干扰未排除）。
- 下一步：3× 背靠背重复跑批（不重 calib）+ logits 落盘 → 测 run-to-run 方差；
  若波动是时间独立抽样，**多次重复 logits 投票**可同时作为诊断和治疗。

## 7. 重复性实验 + 结构化误差归因（2026-08-08，c2c 权重，calib_c2c_seg2 不重 calib）

**结论先行：hw 误差 95% 是"给定输入即可复现"的结构化分量；投票治疗无效；
唯一出路是把结构化误差建模进 QAT（domain randomization）。**

| 实验 | 结果 | 判读 |
|---|---|---|
| 3× 背靠背 500 样本重复跑批（同权重同 calib） | rep1 89.60 / rep2 87.60 / rep3 87.00 | run-to-run 摆动 ±1.3pt，均值 88.07 |
| 两两预测一致率 | 95.6 / 95.4 / 94.2% | 错误高度可复现 |
| 3-logits 累积投票 | 89.00%（≤ 单次最佳 89.60），oracle 上界仅 91.00 | **投票无治疗价值，否决** |
| hw 稳定错误 vs FAKE clean（96.60%） | 三跑共错 48/500 中仅 15 个 FAKE 也错，**33 个是 hw 特有结构化错误** | gap 主体不是模型固有难样本 |
| logit 级误差分解（z-score 后 corr(hw1−fake, hw2−fake)） | corr=0.934，**结构化方差占比 95.2%** | iid 快噪声只占 hw 误差的 ~5%，与 crossval σ_dyn/σ_total 口径一致 |

- 物理图像修正：分钟级"非平稳波动"（§6 段内摆动）主要是**样本组成**造成的表象；
  真正的 hw 误差是输入的确定性函数（LUT 残差/列间不一致），不是时间漂移。
- qat_v7（per-channel 常量 offset/gain，σ_static 全量）已训练（c3a/c3b），
  但结构化 proxy 预检无增益（88.2/87.9 vs c2c 89.0）→ **误差不是"每通道常量偏移"结构**，
  更可能是输入依赖的 LUT 残差（线性化 ≈ 等效权重扰动 δW）。

## 8. probe_dump 残差结构分解 + v8 组分噪声模型（2026-08-08 下午）

板上 `probe_dump.py`：5 层各 1-10 万行真实激活 (x_int, ideal, hw) pairs（subsample）。
对全局 alpha/beta 回归后的残差 R 做结构分解（本地分析）：

| 结构假设 | 解释方差占比 | 实测量级（per layer） |
|---|---|---|
| per-column 常量偏移 off_c | 4-23% | std 264-872 raw（s1a 281 / s2a 478 / s2b 459 / s3a 872 / s3b 264） |
| per-column 增益 g_c | 1-3% | std 1.2-2.5%（0.0251/0.0240/0.0120/0.0157/0.0126） |
| per-element 等效权重扰动 δW（R~x 线性回归） | **21-50%** | dW_rms 3.7-7.2 int8 counts（3.72/6.49/5.00/4.67/7.23） |
| x² 非线性 / E[R\|ideal,col] 输出非线性 | 再加 ~1-11pt / 7-29% | 低阶模型不可分解的残余仍占 ~50-75% |

- 残差 std 随 \|ideal\| 五分位从 660→958（s1a）温和增长 → 绝对底噪 + 弱信号相关混合。
- **proxy_v8**（`Gazelle-national/mnist/sim_noise_proxy_v8.py`）：组分噪声（列偏移+列增益+δW，
  整跑一次采样）+ iid 260。标定：c2c proxy **89.24** vs hw 3 跑均值 **88.07±1.3** — 模型复现真机。
  （若把不可分解残余也当 iid 全量注入则 79.0%，再次实证"结构化≠iid"。）
- v7 权重在 proxy_v8 下复测：c3a 86.86 / c3b 88.00 < c2c 89.24 → v7（错误结构+过量偏移）确实有害。
- **qat_v8**（`eurosat_research/src/qat_v8.py`）：训练时 per-batch 重采样上述三组分 + iid 260，
  幅度全部取 probe 实测值。c3c=1.0× 实测 / c3d=1.5× 余量（覆盖未建模非线性分量）。
- 训练（60ep，init=c2c best）：c3c test 95.85 / c3d 95.30。
- proxy_v8 预检（9 seeds）：c3c 90.12 / **c3d 93.00**（c2c 89.24）→ c3d 上板验证。
- **qat_v9**（`eurosat_research/src/qat_v9.py`）：v8 + RFF 随机非线性残差
  （eps(x)=A·cos(Bx+φ)，per-batch 重采样 B/A/φ，幅度 sqrt(RESID_AL²-260²)），
  覆盖 probe 分解不掉的 50-75% "确定性非线性" 分量。c3e=v9 全组分：clean test 91.70
  （噪声重，clean 掉 4.7pt，预期内）。
- RFF proxy（最接近真机的噪声模型，5 seeds）：c2c 87.08 / c3d 91.12 / **c3e 92.24**。
- 二轮扫参（RFF proxy，5 seeds）：**c3h(0.5×RFF+1.5×线性) 93.62** ≈ c3j(0.75×RFF+1.5×线性) 93.50
  ≈ c3f(0.5×RFF) 93.44 > c3i(0.25×RFF) 91.76；平台期在 0.5-0.75×RFF + 1.5×线性。
- **hw 验证（同一 fresh compass_cali 窗口，1000 样本）**：
  - **c3d 93.80%**（FAKE clean 95.60 → gap 仅 **1.8pt**；旧纪录 c2c 91.20 / gap 5.2pt）。
    hw-only 错误 34 个（与 c2c 的 33 个量级相当，但基数不同）。
  - 机理确认（前 500 样本，跨窗口对照）：c2c 的 37 个 hw-only 结构化错误被 c3d 修掉 **25 个（68%）**，
    仅新增 9 个；c3d hw-only 错误降到 17 个 < 其 FAKE 固有错误 24 个 ——
    **结构化训练精准消除了可复现 hw 误差，残余 gap 已接近模型固有错误底**。
  - c2c 同窗复测 / c3h / c3f：**已完成**（w2 窗口，fresh compass_cali，1000 样本）：
    c2c **89.60%** / c3f **93.20%** / c3h **93.50%**。v8/v9 家族跨窗口稳定领先 c2c ~4pt。
  - **最终冠军：c3d 93.80%**（gap 1.8pt）；c3h 93.50%（gap 2.3pt）为二号种子
    （0.3pt 差距在 run-to-run 噪声 ±1.3pt 内）。
  - 同窗误差重叠：c3h/c3f/c3d 分别修掉 c2c 错误 56/49/68 个，新增 17/13/26 个 ——
    修复/新增比 ~3:1，全家族方向一致。

## 9. C3 总结（2026-08-08 收官）

| 模型 | 噪声训练 | FAKE clean | hw 真机 | gap |
|---|---|---|---|---|
| c2c (v6 split) | iid260+标量 gain/off | 96.40 | 89.60-91.20 | 5.2pt |
| **c3d (v8, 1.5×probe 组分)** | 列偏移+列增益+δW+iid | 95.60 | **93.80** | **1.8pt** |
| c3h (v9, 0.5×RFF+1.5×线性) | +RFF 非线性 | 95.80 | 93.50 | 2.3pt |
| c3f (v9, 0.5×RFF) | +RFF 非线性 | 95.90 | 93.20 | 2.7pt |

- 方法论闭环：probe_dump 归因 → 组分噪声建模 → domain randomization QAT →
  proxy_v8 预选（复现 hw）→ hw 同窗验证。proxy 与 hw 的排名/量级一致性已三次验证。
- 残余 gap（1.8-2.7pt）≈ iid 快噪声 + 未建模非线性的不可约分量 + 模型固有错误；
  进一步收益需更大模型容量或片上 LUT 修正，超出本轮范围。
- 冠军工件：`eurosat_research/runs/c3d_J1_v8probe15_local/`（best.pth + 部署权重 + hw/fake logits）；
  板上 `~/j1/weights_c3d` + `calib_c3d.json`。
