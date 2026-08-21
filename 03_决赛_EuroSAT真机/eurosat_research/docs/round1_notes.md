# Round 1 — QAT 范式验证记录

> 日期: 2026-08-07 · 状态: **完成**

## 结果汇总 (修复版, 全 test 5400)

| 配置 | test acc | macro F1 | ECE | vs FP32 |
|---|---|---|---|---|
| FP32 基线 | 96.61% | 96.53% | 0.019 | — |
| **v5 + SGD** | **97.43%** | **97.38%** | — | **+0.82%** |
| **v5 + Muon** | **97.22%** | 97.17% | 0.037 | +0.61% |
| v5 + AdamW | 96.93% | 96.89% | 0.035 | +0.32% |
| v5 无噪声 | 96.74% | 96.69% | — | +0.13% |
| v4 风格 | 96.81% | 96.76% | — | +0.20% |
| v5 强增强 | 95.35% | 95.33% | — | −1.26% |

## Round 1 洞察

1. **SGD+Momentum 最优 (97.43%)**: LSQ 配 SGD 的经验在 QAT+噪声下复现, AdamW 非最优。
   SGD 对量化噪声鲁棒 (动量平均抑制噪声梯度)。
2. **Muon 次之 (97.22%)**: 正交化动量在 QAT 下不仅不降级 (文献 Beyond Outliers 担忧),
   反而更鲁棒。低成本高收益。
3. **绝对加性噪声价值小但为正** (+0.19% vs 无噪声): 与 crossval 的 σ_total≈4.4 counts
   量级相符 — 噪声小, 主要靠量化对齐。
4. **v5 量化方案 vs v4**: AdamW 口径 96.93 vs 96.81, v5 略优。
5. **强增强反伤** (95.35%): Rot90+ResizedCrop 破坏 QAT 稳定性。卫星遥感增强需温和:
   保留 HFlip+Rot10, 或 QAT 训练后期再开强增强。

## 关键事件

### 1. QAT v5 开发中发现的训练崩溃 bug (已修复)

**现象**: 所有 v5 QAT 实验精度冻结在 11.22% (≈随机 1/9), loss 恒 2.303。

**根因**: `_forward_qat` 先做 STE 量化得到 `x_q`/`w_q` (已是反量化浮点, 含 scale 因子),
再按 osimulator 整数域 dequant 公式 `x_scale·w_scale·(y_int − x_zp·col_sum)` 计算,
其中 `y_int = conv(x_q, w_q)` 混用了两个域 → 双重缩放 + 灾难性消减,
输出量级被压扁 ~2000× (out std 0.0012 vs 正常 0.56)。

**修复**: 改用浮点域路径 `y = conv(x_dq, w_dq)` — 数学上 ≡ osimulator 整数域 dequant
(数值验证 max diff 2e-6), 但数值稳定、梯度自然流动。修复后真实数据 5 batch
loss 2.302→2.006, acc 14%→31% 正常收敛。

### 2. crossval 逆向结果 → 噪声模型修正 (重大)

来源: `gazelle-crossval/report/CROSSVAL_REPORT.md` (2026-08-07 完成)

| 发现 | 含义 |
|---|---|
| **真机噪声 = 纯绝对加性底噪** | σ_total ≈ 4.49 counts (uint8) / 3.85 (uint4x16), 与信号幅度无关 |
| osimulator 噪声 = 底噪 + 信号相关 | 跨 regime 变化 ~300×, 与真机**结构性不同** |
| **QAT 建议: tia_noise_std = 0.0392** | = σ_total/rms_ideal (uint8 全值域随机 GEMM 口径) |
| 噪声必须全局固定 σ, 非 per-channel 相对 | 注入 `eps ~ N(0, σ)`, σ 与输出幅度无关 |
| uint4 直接上板不可行 | 信号 ±1.5 counts < 噪声底 4.4 counts (SNR<1) |
| σ_total 是"校准后新鲜状态"噪声 | 漂移靠 SOP (新鲜 compass_cali + canary), 不靠训练吸收 |

**对 QAT v5 的变更**:
- `inject_output_noise`: per-channel 相对 → **全局绝对加性** `y + N(0, ratio×RMS(y))`
- `output_noise_ratio`: 0.0457 (osim 口径) → **0.0392** (crossval hw 口径)
- 对比实验保留: `r1_v5_no_noise` (无噪声) 作为消融下界

## 实验矩阵 (Round 1)

| 配置 | 变量 | 目的 |
|---|---|---|
| r1_fp32_base | FP32 无 QAT | 上界参考 (96.63% val) |
| r1_v5_full | v5 + 绝对噪声 0.0392 + 12bit 输出量化 | 主配方 |
| r1_v5_no_noise | v5 无噪声 | 噪声价值消融 |
| r1_v4_style | per-channel signed 激活 + 权重噪声 (旧 v4 风格) | 量化方案对照 |
| r1_v5_muon | Muon 优化器 | 优化器对照 |
| r1_v5_sgd | SGD+Momentum | 优化器对照 |
| r1_v5_strong_aug | Rot90+ResizedCrop+ColorJitter | 增强对照 (失败) |

## 下一步 (Round 2)

- 架构搜索 ≤2M MACs, 用 **SGD 配方** (97.43% 冠军配置)
- 增强策略修正: 温和增强, 不做 Rot90/ResizedCrop 激进变换
- 验证低分辨率宽层 (1×1/3×3 混合 kernel) 在 ≤2M MACs 下能否保持 ~97%
