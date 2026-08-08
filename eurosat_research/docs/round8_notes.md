# Round 8 — scaling ladder：预算外两强 160ep 长训

日期：2026-08-08 · 状态：**完成** · 指标 QAT clean (test 5400)，J1 配方 160ep
背景：R7 证明 ≤2M 内无架构超 J1；R6 的两个预算外赢家（w200 / rf_s2k3）80ep 已超 J1_long，
本轮回合 160ep 确认 scaling ladder 形态。不用于部署，仅作预算-精度标尺。

## 结果

| run | MACs | 80ep test | 160ep test | 长训增益 |
|---|---|---|---|---|
| r8_rf_s2k3_160 | 4.52M | 96.44 | **96.93** | +0.49 |
| r8_w200_160 | 4.62M | 96.54 | 96.39 | −0.15 |
| （参考）J1_long | 1.38M | 95.52 | 96.30 | +0.78 |

## Scaling ladder（QAT clean，最终口径）

```
1.05M  Model 2 (int8)     92.20
1.38M  J1 (160ep)         96.30   ← ≤2M 严格预算最优
4.52M  rf_s2k3 (160ep)    96.93   ← RF 路线, 同 MACs 胜宽度路线 0.54pt
4.62M  w200 (160ep)       96.39
17.04M Model 4 E (80ep)   97.43
156.6M Model 1 (int8)     97.89
```

## 判读

1. **RF 路线（rf_s2k3）长训继续涨（+0.49），宽度路线（w200）长训停滞（−0.15）**——
   w200 的 80ep 优势部分是真容量、部分是早停运气；rf_s2k3 的 RF 归纳偏置在
   长训下持续兑现。同预算 RF > 宽度，与 R6 b1 结论互证。
2. rf_s2k3 距 Model 4 E 仅 0.5pt 而 MACs 仅 1/4.5——中高预算段的甜点。
3. 若未来放宽部署预算到 ~4.5M，rf_s2k3（J1 + stage2 3×3，RF 49px）是首选架构；
   严格 ≤2M 仍 J1。

权重：本地 `auto_research/runs/r8_*/best.pth`（未入 national 仓库）。
