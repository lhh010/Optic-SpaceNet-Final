# Round 6 — 架构三个问题：Pooling / 宽深·RF / 全局旁路

日期：2026-08-08 · 状态：进行中 · 指标：**QAT clean (test 5400)**，配方统一 r2_J1（SGD lr0.05, 80ep, QAT v5, seed 42）
对照组：`r6_ctrl` = r2_J1 复刻（历史值 95.52%，同 run 复核 seed 稳定性）

## 问题与设计

### Q1 Pooling 方式敏感性
固定 J1 骨架（stem k3s2 + 全 1×1 + GAP），只换下采样算子：

| arm | 下采样 | 宽度 | MACs（估） | 意图 |
|---|---|---|---|---|
| r6_ctrl | MaxPool2d(2) | 16/32/64/128 | 1.38M | 对照 |
| r6_pool_avg | AvgPool2d(2) | 同 | 1.38M | 等宽等 MACs，纯算子对比 |
| r6_pool_s1x1 | 1×1 conv s2 | 同 | ~1.97M | 可学习下采样（丢 3/4 位置） |
| r6_pool_patch | PixelUnshuffle(2)+1×1 mix | 同 | ~2.9M | patchify：保全部位置 + 通道混合 |
| r6_pool_patch_n | 同上 | 12/24/48/96 | ~1.3M | patchify 等 MACs 公平对比 |

读法：avg vs max → 池化统计量敏感性；patch vs patch_n → 把"多出的 MACs"归因（算子本身 vs 容量）；patch_n vs ctrl → 等预算下 patchify 是否优于 maxpool。

### Q2 BagNet 结构下宽度 vs 深度 vs 感受野
- 宽度（depth 固定 5 conv，maxpool）：`r6_w075` [12/24/48/96] ~0.78M / ctrl 1.38M / `r6_w150` [24/48/96/192] ~3.1M / `r6_w200` [32/64/128/256] ~5.5M → 宽度-精度曲线
- 深度（宽度固定）：`r6_d133` stage_depths=(1,3,3) ~1.74M / `r6_d244` (2,4,4) ~2.5M → 深度-精度曲线
- RF（深度、宽度都固定，只放大空间足迹）：
  - `r6_rf_stem5`：stem k5s2（RF 3→5@j2，低层大足迹）~2.2M
  - `r6_rf_s2k3`：stage2 换 3×3（kernels=[1,3,1]，RF 17→25）~2.3M
- b1 判读：同 MACs 增量下，d133（深度↑RF 不变）vs rf_s2k3（RF↑深度不变）谁涨得多 → 深度关键还是 RF 关键

### Q3 全局信息旁路
stem 输出（16ch@16×16）→ AvgPool4× → Flatten(256) → Linear→ReLU → concat 到 GAP(128) 后 → head：
- `r6_glb64` bypass_dim=64（+~40k MACs）
- `r6_glb128` bypass_dim=128
读法：bag-of-local-features 缺全局布局，若 Highway/River 等结构类 F1 回升而 MACs 近零 → 性价比极高的补丁。

## 工程

- models.py 新增：`pool_mode`(max/avg/stride1x1/patchify)、`stem_kernel`、`stage_depths`、`bypass_dim`，全部后向兼容
- 旁路 Linear 与 head 一样吃 v5 QAT（全层口径一致）
- 容器双 A800 两队列跑批，日志 auto_research/logs/r6_*.log
- 后续 R7：综合 findings 设计 ≤2M（目标 ~1.4M）改进架构 1-2 个，80ep 筛选 + 160ep 冲刺
