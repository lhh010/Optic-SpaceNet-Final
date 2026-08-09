# M5-M8 训练汇总（J1 家族，2026-08-08 起）

> 代码: `eurosat_research/`（共用代码 src/runner.py + 变体 configs/*.json）
> 数据口径: EuroSAT test 5400（eurosat_split, seed 42）；QAT clean = v5（输出噪声 r=0.0392）
> 日志: `logs/m{5,6,7,8}_*.log`（实时）+ `eurosat_research/runs/*/metrics.jsonl`（每 epoch 结构化）

## 干净阶段（160ep）结果

| 模型 | 架构 | MACs | params | best val | **test acc** | macro_f1 | 完成时间 | 状态 |
|---|---|---|---|---|---|---|---|---|
| M5 | rf_s2k3 + stem k5 | 5.31M | 100,250 | 96.89% | **96.81%** | 0.9672 | 2026-08-08 23:28 | ✅ 完成 |
| M6 | 纯 J1（head128） | 1.38M | 50,330 | 96.31% | **96.30%**（本地复评 96.39%） | 0.9615 | — | ✅ 复用 r3_J1_long ckpt |
| M8 | rf_stem5（C0=16 k5） | 2.16M | 51,098 | — | — | — | 23:2x 启动至 ep10 后暂停 | ⏸️ 用户指示今天不训 |
| M7 | w075（C0=12） | 0.86M | ~29K | 待填 | 待填 | — | — | ⏳ 待启动 |

## v8 漂移鲁棒阶段（60ep, probe 组分噪声 ×1.5 余量）

| 模型 | 配置 | 状态 |
|---|---|---|
| M5 | `configs/m5_j1rf_stem5_v8probe15.json`（init weights/m5_j1rf_stem5_clean.pth）| ⏳ 等干净阶段后 |
| M6 | `configs/m6_j1_v8probe15.json`（init weights/j1_r3_J1_long_best.pth）| ⏳ |
| M7 | `configs/m7_j1w075_v8probe15.json` | ⏳ |
| M8 | `configs/m8_rf_stem5_v8probe15.json` | ⏳ |

## 架构依据（R2-R8 探索，已核查）

- M5 = rf_s2k3 (R8: 160ep 96.93) + stem5 (R6: 效率 0.67pt/M)，C0=16 保持
- M6 = 纯 J1 (R7 决赛口径最优 96.30; **head256 160ep 是负优化 -0.23pt, 已剔除**)
- M7 = w075 (R6/R8: 0.86M 档 95.30@170ep)
- M8 = rf_stem5@C0=16 (R6 效率第一; **C0≤14 即崩, 不用 w075 的 C0=12**)

## M6 更正记录（2026-08-09）

此前 M6 clean 报的 95.65% 是一次**作废 run**：用了已删除的旧脚本
`train_m6_j1_head256.py`（非统一 runner + config），训练轨迹异常，ckpt 本身训差。
核查与复评（`src/eval_ckpt.py`，当前代码 + 同一 EuroSAT_RGB + v5 确定性推理）：

- 超参逐条核对与 r3_J1_long **完全一致**（lr 0.05 / warmup 5 / SGD nesterov wd 5e-4
  / batch 64 / ls 0.05 / aug standard / v5 噪声 r=0.0392 / arch head128 纯 J1）
- 旧 run ckpt `weights/m6_j1_clean.pth` 本地复评 **test 95.59%** —— ckpt 本身差，
  非评测口径问题；该 ckpt 作废，v8 阶段不得以其初始化
- 我们的 `runs/r3_J1_long_3b6c03f6/best.pth`（同架构同配方，即 M6 clean 正身）
  本地复评 **test 96.39%**（原始记录 96.30%），结论维持
- 已入库：`weights/j1_r3_J1_long_best.pth`（MD5 `7c9752735a6fbb2bf2c021538e45ec69`，
  与 README 档案值一致）+ `j1_r3_J1_long_summary.json`；
  `configs/m6_j1_v8probe15.json` 的 `init_from` 已指向该 ckpt
