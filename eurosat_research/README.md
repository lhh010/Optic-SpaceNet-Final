# EuroSAT 光计算研究存档（auto_research R1/R2/R3 + R4/R5/C2/C3 + R6/R7/R8）

> 队伍 CICC1003564 · 整理自 `Ltsimulator-test/auto_research/`（2026-08-08 二次更新：C2/C3 真机 SOTA）
> 数据集：EuroSAT 遥感 10 分类（64×64 RGB，test 全量 5400）

## 内容

```
eurosat_research/
├── src/        # config-driven 训练引擎 + QAT v5（及后续 v6-v9）+ 模型族 + Muon
├── configs/    # R1/R2/R3/R6/R7/R8 全部实验配置（46 个 JSON）
├── scripts/    # 可视化（plot_runs.py）+ 容器跑批（run_queue.sh）+ 汇总（collect_r6.py）
├── docs/       # 各轮探索记录 + 1×1/BagNet 讨论 + perf vs MACs 图
└── weights/    # R1-R3 两个参数档位的 SOTA 权重（R6-R8 权重未归档，存本地 auto_research/runs/）
```

## 两个参数档位的 SOTA 权重

| 档位 | 权重 | params | MACs/张 | QAT clean (test 5400) |
|---|---|---|---|---|
| 大档（Model 4 E） | `weights/model4e_r1_v5_sgd_best.pth` | 269.5K | 17.04M | **97.43%** |
| 小档（J1） | `weights/j1_r3_J1_long_best.pth` | 50.3K | 1.38M | **96.30%** |
| 小档·真机 SOTA（c3d） | `weights/j1_c3d_v8probe15_best.pth` | 50.3K | 1.38M | 95.30%（clean）|

- 大档：R1 冠军 `r1_v5_sgd`（QAT v5 + SGD+Momentum，80ep），源 run `r1_v5_sgd_4d16eae4`，MD5 `54c663c1cd1c95e4a4551437e52f9151`
- 小档：R3 冠军 `r3_J1_long`（J1 架构 160ep 长训），源 run `r3_J1_long_3b6c03f6`，MD5 `7c9752735a6fbb2bf2c021538e45ec69`
- 各权重附 `*_summary.json`（params/MACs/best_val 等档案信息）
- MACs→精度权衡：17.04M→1.38M（↓12×），精度仅降 1.13pt——J1 是 Pareto 最优点
- **真机 SOTA**：c3d（qat_v8，probe 实测组分噪声 1.5× 余量），**Gazelle 真机 93.80%**（1000 样本，fresh compass_cali），train-val gap 仅 1.8pt；附 `j1_c3d_v8probe15_summary.json`、`j1_c3d_hw_logits_1000.npy`、int8 部署权重 `deploy_c3d/`

## 三轮结论速览

- **R1（QAT 范式）**：SGD+Momentum 最优（97.43%）> Muon（97.22%）> AdamW（96.93%）；绝对加性噪声（σ=0.0392，crossval 真机口径）价值小但为正；强增强反伤（−1.26pt）
- **R2（架构搜索 ≤2M MACs）**：全 1×1 kernel 系统性最优（对齐 8×2 tile，展平无浪费）；stem stride=2 > stride=4；更宽≠更好（G3x 3.08M 反不如 H1 1.21M）。冠军 **J1**：1.38M MACs / 95.52%（80ep）
- **R3（冠军精调）**：长训 160ep +0.78pt → **96.30%**；加宽 GAP head 有效（95.94%）；SWA 无效（avg < best）

详见 `docs/round1_notes.md` / `round2_notes.md` / `round3_notes.md` 与 `docs/perf_vs_macs_qat.png`。

## R6/R7/R8（2026-08-08）：架构三问 + 严格预算冲刺 + scaling ladder

- **R6（14 runs 消融，80ep）**：① Pooling 敏感，**MaxPool 显著最优**（avg −0.74 / patchify 等 MACs −0.43 / stride-2 1×1 −10.45 灾难）；② **宽度 > 感受野 > 深度**（宽度单调 +0.5pt/倍增；rf_s2k3 RF 17→49px +0.92 且增益精准落 Highway/River 结构类；深度无效）；stem k5 是 MACs 效率最高的单点改动（0.67pt/M，机制=底层纹理滤波器质量）；③ 全局旁路中性证伪（结构类缺中等 RF 非全局布局）。详见 `docs/round6_notes.md`
- **R7（13 runs，≤2M 严格预算，160ep 决赛 + 多种子）**：**7 个候选全部未能超过 J1**（160ep 96.30/96.24，稳健局部最优）；head256 的 80ep 优势在 160ep 反转 −0.23——架构对比必须在最终训练时长口径下做。详见 `docs/round7_notes.md`
- **R8（scaling ladder，160ep）**：rf_s2k3（4.52M）**96.93** / w200（4.62M）96.39——预算外架构长训后确认超越 J1_long；rf_s2k3 距 Model 4 E（17.04M, 97.43，80ep 口径）仅 0.5pt 而 MACs 仅 1/4
- 架构旋钮（`src/models.py`）：`pool_mode`(max/avg/stride1x1/patchify)、`stem_kernel`、`stage_depths`、`bypass_dim`，全部后向兼容
- R6-R8 权重不进本仓库：存 `Ltsimulator-test/auto_research/runs/r6_*/r7_*/r8_*/`

## 用法

```bash
# 训练（需 GPU；原运行在容器 gazelle_sim / A800）
python src/runner.py --config configs/r3_J1_long.json --gpu 0

# 可视化 runs 目录
python scripts/plot_runs.py <runs_dir> --curves val_acc

# 复现 perf vs MACs 图
python docs/plot_perf_vs_macs.py

# 复现真机上板实验 Pareto 图（docs/pareto_hw_acc.png：MACs vs Gazelle 真机 acc，
# 覆盖 MNIST MLP / OpticSpaceNet Model 1a-4 / J1 QAT 系列，数据源见脚本头注释）
python docs/plot_pareto_hw.py
```

## J1 架构（小档 SOTA）

```
stem:  Conv3x3 s2 3→16 + BN + ReLU + MaxPool2      [电计算]
stage1: Conv1x1 16→32 + BN + ReLU + MaxPool2        [光计算]
stage2: Conv1x1 32→64 + BN + ReLU
        Conv1x1 64→64 + BN + ReLU + MaxPool2        [光计算]
stage3: Conv1x1 64→128 + BN + ReLU
        Conv1x1 128→128 + BN + ReLU                 [光计算]
head:   GAP → FC 128→128 ReLU → FC 128→10
```

## R4/R5/C2/C3（2026-08-07/08）：真机部署 + 结构化噪声 QAT → 真机 93.80%

- **R4（真机部署链路）**：J1 部署到 Gazelle 真机（`mnist/run_j1_gazelle.py`，stem/head 电计算 + 5 个光计算 1×1 conv，m≤2 tiling 规避 FPGA 回绕 bug，per-layer alpha/beta 校准）。详见 `docs/round4_hw_deploy.md`
- **R5/C2（qat_v6 split-noise）**：部署 bug 修复（head bias/stem 不一致）+ iid260/标量 gain/off 抖动 → c2c 真机 91.20%。详见 `docs/round5_c2_drift_robust.md` §1-§5
- **C3（结构化归因 + v8/v9，真机 SOTA）**：
  1. 归因：分段跑批/探针证伪"漂移"；3× 重复跑批证明 **hw 误差 95.2% 是 run 间可复现的结构化分量**（投票治疗证伪）
  2. 板上 probe_dump 逐层分解出三组分：per-column 偏移（4-23% 方差）+ per-column 增益（1-3%）+ per-element δW（21-50%），残余确定性非线性用 RFF 建模（v9）
  3. 产出 **qat_v8（probe 实测组分 domain randomization）/ qat_v9（+RFF）**；`mnist/sim_noise_proxy_v8.py` 标定复现真机（proxy 89.2 vs hw 88.1）
  4. 同窗终验（1000 样本）：**c3d 93.80%（冠军）** / c3h 93.50 / c3f 93.20 / c2c 89.60；机理：c2c 的 hw-only 结构化错误 68% 被 v8 修掉
  5. 详见 `docs/round5_c2_drift_robust.md` §6-§9；端到端流程 `docs/DELIVERY_END_TO_END.md`
