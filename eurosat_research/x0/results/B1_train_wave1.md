# X0 · B1 首批训练结果（wave-2，2026-08-09）

口径：QAT **v8 1.5× 全程 160ep 从零训**，SGD lr0.05，test 5400 clean。单 seed。
对照 **x0_ctrl = J1 原架构（pool_mode=max）同配方**：**95.57%**（run `x0_pool3_160_021d51d7`，
name 字段沿用 pool3，已修正本地 config；与 pool3 臂 run `x0_pool3_160_d9a8eeef` 是两个目录）。

| 臂 | 改动 | MACs | test | Δ vs ctrl | 判读 |
|---|---|---|---|---|---|
| x0_ctrl | J1（max pool 2×2） | 1.378M | **95.57** | — | 对照（c3d 60ep 续训 clean 95.30 旁证合理） |
| x0_pool3 | MaxPool 3×3 s2（RF/步 翻倍） | 1.378M | 95.43 | −0.14 | ≈ 种子噪声内，**零成本 RF 无收益** |
| x0_blurpool | BlurPool 抗混叠 | 1.378M(+电64K) | 94.31 | **−1.26** | 明确负：固定低通损害纹理极值统计，此 regime 抗混叠不成立 |
| x0_pool4 | +第4次 pool（j=32，GAP 16→4 位置） | 1.378M | −0.76 → 94.81 | **−0.76** | 负：节奏改动代价（GAP 位置↓/RF 过冲）超过 RF 收益 |
| **x0_dsconv3** | 下采样点换 Conv3×3 s2（stem pool 保留 max） | **2.557M** | **96.22** | **+0.65** | 唯一正臂；效率 0.55 pt/M，预算外 |

## 结论

1. **"零成本 RF"（pool3）在 v8 部署配方下无收益**——RF 增益必须通过可学习算子（dsconv3）才兑现，
   与 R6 v5 口径"rf_s2k3 +0.92"互证：RF 的价值在特征质量而非几何感受野本身。
2. **dsconv3（3×3 s2 下采样）是当前 MACs 效率最高的预算外改动**：+1.18M → +0.65pt（0.55 pt/M），
   优于 R6 的 w150（0.39）/w200（0.31），接近 rf_stem5（0.67，v5 80ep 口径）。
   被 R6"pooling 线关闭"错误连坐的臂，正式平反。
3. 节奏轴（pool4）与抗混叠轴（blurpool）关闭（单 seed −0.76/−1.26，超出噪声带）。

## 下一批（wave-3 候选）

- dsconv3 **seed 43 复核**（+0.65 是否真效应，种子内噪声 ~0.1-0.15）
- dsconv3 **预算合法化**：仅 stage1/stage2 下采样点用 conv3s2、或配 w075 宽度压回 ≤2M
- dsconv3 + max3 stem pool 组合臂
- （依赖 C1/A 组）v10 噪声模型臂：删 RFF + 浅层 tile 块相关 + 行对共模
