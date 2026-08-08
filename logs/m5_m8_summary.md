# M5-M8 训练汇总（J1 家族，2026-08-08 起）

> 代码: `eurosat_research/`（共用代码 src/runner.py + 变体 configs/*.json）
> 数据口径: EuroSAT test 5400（eurosat_split, seed 42）；QAT clean = v5（输出噪声 r=0.0392）
> 日志: `logs/m{5,6,7,8}_*.log`（实时）+ `eurosat_research/runs/*/metrics.jsonl`（每 epoch 结构化）

## 干净阶段（160ep）结果

| 模型 | 架构 | MACs | params | best val | **test acc** | macro_f1 | 完成时间 | 状态 |
|---|---|---|---|---|---|---|---|---|
| M5 | rf_s2k3 + stem k5 | 5.31M | 100,250 | 96.89% | **96.81%** | 0.9672 | 2026-08-08 23:28 | ✅ 完成 |
| M6 | 纯 J1（head128） | 1.38M | ~50K | 95.72% | **95.65%** | 0.9550 | 2026-08-08 23:48 | ✅ 完成 |
| M8 | rf_stem5（C0=16 k5） | 2.16M | 51,098 | — | — | — | 23:2x 启动至 ep10 后暂停 | ⏸️ 用户指示今天不训 |
| M7 | w075（C0=12） | 0.86M | ~29K | 待填 | 待填 | — | — | ⏳ 待启动 |

## v8 漂移鲁棒阶段（60ep, probe 组分噪声 ×1.5 余量）

| 模型 | 配置 | 状态 |
|---|---|---|
| M5 | `configs/m5_j1rf_stem5_v8probe15.json`（init weights/m5_j1rf_stem5_clean.pth）| ⏳ 等干净阶段后 |
| M6 | `configs/m6_j1_v8probe15.json` | ⏳ |
| M7 | `configs/m7_j1w075_v8probe15.json` | ⏳ |
| M8 | `configs/m8_rf_stem5_v8probe15.json` | ⏳ |

## 架构依据（R2-R8 探索，已核查）

- M5 = rf_s2k3 (R8: 160ep 96.93) + stem5 (R6: 效率 0.67pt/M)，C0=16 保持
- M6 = 纯 J1 (R7 决赛口径最优 96.30; **head256 160ep 是负优化 -0.23pt, 已剔除**)
- M7 = w075 (R6/R8: 0.86M 档 95.30@170ep)
- M8 = rf_stem5@C0=16 (R6 效率第一; **C0≤14 即崩, 不用 w075 的 C0=12**)
