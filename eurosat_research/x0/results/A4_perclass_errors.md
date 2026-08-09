# A4 — c3d per-class hw 错误分布分析

Round X0 / 分析子任务 A4 · 2026-08-09
数据：`runs/c3d_J1_v8probe15_local/{hw,fake}_logits_1000.npy`（1000 样本同序）+ 复现 labels（`x0/data/labels_1000.npy`）。
脚本：`x0/scripts/a4_perclass.py`。

## 0. labels 复现与自检

- 复现逻辑：EuroSAT ImageFolder 顺序（sorted 类目录 + sorted 文件名，27000 张）→ `eurosat_split`（`RandomState(42).shuffle(list(range(27000)))`，test = idx[5400:10800]）→ 前 1000。
- 自检 1：与历史导出 `Gazelle-national/mnist/j1_board/weights_j1/test_labels_j1.npy` **1000/1000 完全一致**。
- 自检 2：fake acc = **95.60%**（期望 95–96%，与 c3d 记录吻合）✓
- hw acc = **93.80%**（与记录吻合）✓

## 1. per-class 精度表（n=1000）

| 类别 | 支持数 | hw 错 | fake 错 | hw-only | both 错 | hw acc% | fake acc% | hw-only 率% | fake 错误率% |
|---|---|---|---|---|---|---|---|---|---|
| AnnualCrop | 112 | 5 | 3 | 3 | 2 | 95.54 | 97.32 | 2.68 | 2.68 |
| **Forest** | **115** | **17** | **1** | **16** | 1 | **85.22** | **99.13** | **13.91** | **0.87** |
| HerbaceousVegetation | 111 | 5 | 6 | 2 | 3 | 95.50 | 94.59 | 1.80 | 5.41 |
| Highway | 101 | 17 | 15 | 4 | 13 | 83.17 | 85.15 | 3.96 | 14.85 |
| Industrial | 90 | 2 | 1 | 1 | 1 | 97.78 | 98.89 | 1.11 | 1.11 |
| Pasture | 69 | 1 | 3 | 0 | 1 | 98.55 | 95.65 | 0.00 | 4.35 |
| PermanentCrop | 107 | 4 | 3 | 3 | 1 | 96.26 | 97.20 | 2.80 | 2.80 |
| Residential | 117 | 2 | 1 | 1 | 1 | 98.29 | 99.15 | 0.85 | 0.85 |
| River | 79 | 5 | 7 | 2 | 3 | 93.67 | 91.14 | 2.53 | 8.86 |
| SeaLake | 99 | 4 | 4 | 2 | 2 | 95.96 | 95.96 | 2.02 | 4.04 |

合计：hw 错 62、fake 错 44、**hw-only 34**、both 错 28、fake-only 16。

## 2. hw-only 错误分布

类别分布（共 34）：**Forest 16（47.1%）**，Highway 4，AnnualCrop/PermanentCrop 各 3，HerbaceousVegetation/River/SeaLake 各 2，Industrial/Residential 各 1，Pasture 0。

hw-only 混淆对（true → hw_pred，top）：

| 混淆对 | 次数 |
|---|---|
| **Forest → Pasture** | **6** |
| Forest → River | 3 |
| Forest → AnnualCrop | 3 |
| Highway → River | 3 |
| Forest → HerbaceousVegetation | 2 |
| Forest → SeaLake | 2 |
| River → Pasture | 2 |
| 其余各 1 | … |

对照：fake 固有错误的混淆对集中在 **Highway → Industrial(7) / Highway → River(5) / River → AnnualCrop(5)** —— 结构类互混，与已知"1×1/BagNet 短板在 Highway/River"一致。Forest 在 fake 侧几乎不错（1/115），且 fake 侧 Forest 的混淆方向与 hw-only 完全不同。

## 3. 分布形状：跟随 fake 还是独立结构？

- hw-only 分布**不跟随** fake 错误分布：fake 错误主要在 Highway(15)/River(7)/HerbVeg(6)，hw-only 主要在 Forest(16)。两者的峰值类别不重叠。

## 4. 统计检验

1. **hw-only 类别分布 vs 均匀**（chi2 GOF）：chi2 = 55.41, df = 9, **p < 1e-8** → 显著非均匀。
2. **hw-only 率 vs fake 错误率逐类同构**（2×10 列联表）：chi2 = 87.80, df = 27, **p < 1e-8** → 两类错误的类别结构显著不同。
3. **hw-only 分布 vs fake 分布形状**（多项 GOF，期望<5 合并后 4 格）：chi2 = 37.01, df = 3, **p < 1e-7** → hw-only 不是 fake 错误的随机放大。
4. **Bootstrap / 置换**（随机放大零假设：在 956 个 fake 对样本池内按全局 3.56% 率抽取）：
   - 各类 hw-only 计数 95% CI 上限均为 6–8，仅 **Forest（obs=16）落在 CI 外**，其余 9 类全部在 CI 内。
   - 置换检验：34 个 hw-only 位置中 ≥16 落在 Forest 标签位置的概率 **p ≈ 0**（B=20000 未观察到一次）。
5. **混杂排除（漂移/批次伪影）**：16 个 Forest hw-only 样本下标为 [70, 106, 117, 181, 190, 192, 282, 429, 446, 472, 594, 644, 650, 724, 725, 949]，散布于 10 个百样本桶中的 8 个，无时间/批次聚集 → 不支持"静态漂移恰好打在序列某段"的解释。
6. **功效说明**：hw-only 共 34 个，除 Forest 外的类别计数 0–4，单类层面的次要差异无检验功效；结论只敢下在 Forest 这个 dominant 信号上。

## 5. 机制证据：margin 脆弱 vs 方向耦合

| 类别 | margin p10 | margin 中位 | hw−fake 残差 std |
|---|---|---|---|
| Forest | 3.218 | 4.274 | **0.826** |
| 全局/其余类 | 1.4–3.7 | 3.8–5.3 | 0.27–0.68（全局 0.541） |

- Forest 样本的 fake top1−top2 margin **并不小**（p10=3.22，中位 4.27，处于中上水平）；被翻转的 16 个 Forest 样本 margin 分布 2.52–5.22，与未翻转 Forest 样本（p10=3.31，中位 4.36）基本重叠 → **不是 margin 脆弱性**。
- Forest 样本的 hw−fake logit 残差 std = **0.826，为 10 类最大，约全局均值（0.541）的 1.5 倍** → hw 噪声在 Forest 样本上的有效幅度系统性偏大，即噪声与 Forest 的特征/判别方向发生耦合（与 C3 归因的 per-column 增益 + δW 结构化分量一致：Forest 的判别方向恰好落在增益大的列上）。

## 6. 结论

**存在明确的"噪声-特征方向耦合"证据，且就落在 Forest 一个类上。**

1. c3d 的 62 个 hw 错误中，28 个是模型固有（fake 也错，集中 Highway/River 结构类，正常），34 个 hw-only 中 **16 个（47%）集中在 Forest**，Forest hw 精度从 fake 的 99.13% 跌到 85.22%（−13.9pt）——单一类别贡献了 hw-vs-FAKE gap（1.8pt）的约一半。
2. 该集中不能用"随机放大 fake 错误"解释（3 个独立检验 p < 1e-7，置换 p≈0），也不能用序列漂移/批次伪影解释（无时间聚集），也不能用 Forest 决策 margin 小来解释（margin 健康）。
3. 直接机制证据：Forest 样本的 logit 域 hw 残差幅度显著大于其他类 → 列结构噪声恰好打在 Forest 的判别方向上。**Forest → Pasture（6 次）是第一大混淆对**，可作为后续校准/建模的优先靶点。
4. 含义：残余 gap 的一半不是模型固有短板的延伸，而是可通过"按类/按方向建模噪声"或针对性校准消除的；若能修复 Forest 通道，hw 精度上限约 95.4%（93.80 + 16/1000），逼近 FAKE 的 95.60%。
