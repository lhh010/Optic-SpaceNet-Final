# Round 3 — J1 冠军精调记录

> 日期: 2026-08-07 · 状态: 进行中

## 结果 (部分)

| 配置 | MACs | epochs | test acc | 备注 |
|---|---|---|---|---|
| J1 基线 (r2) | 1.38M | 80 | 95.52% | Round 2 冠军 |
| **J1_long** | 1.38M | 160 | **96.30%** | 长训练 +0.78pt |
| **J1_head** | 1.44M | 80 | **95.94%** | 加宽 GAP head +0.42pt |
| J1_swa | 1.38M | 120 | 待重跑 | SWA bug 修复后 |
| J1_swa_head | 1.44M | 160 | 待重跑 | SWA bug 修复后 |

## 洞察 (进行中)

1. **长训练有效**: 80→160 epochs +0.78pt (95.52→96.30)。QAT+噪声训练需要更长
   收敛时间 (噪声是正则化, 需要更多 epoch 达到最优)。
2. **加宽 head 有效**: params 免费 (45.9K→116K), MACs 仅 +0.06M, 精度 +0.42pt。
   验证了"params 免费, activation compute 受限"的第一性原理。
3. **SWA bug 记录**: `swa_accum[k] /= swa_count` 遇到 BN 的
   `num_batches_tracked` (Long 类型) 崩溃。修复: 累积时只保留浮点张量
   (`v.dtype.is_floating_point`)。

## 目标

1.38M MACs 内逼近 97% (当前 96.30%, 距 Model 4 E 的 97.43% 差 1.1pt,
但 MACs 只有 1/12)。
