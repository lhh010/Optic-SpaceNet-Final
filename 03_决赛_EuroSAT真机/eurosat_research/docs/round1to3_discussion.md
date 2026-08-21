三个问题分开答，先纠正一个小误会：J1 不是"对输入 element-wise 地 Linear"——stem 的 3×3 conv s2 先提取局部特征，之后 1×1 做的是**特征通道混合**（等效 NiN 的 mlpconv），而 3 个 MaxPool 提供空间混合。算一下感受野：stem conv(3,s2) → r=3；pool → r=5；stage1 1×1 + pool → r=9；stage2 + pool → r=17；stride 16，最后 GAP。**J1 实际是一个感受野 ~17px、bag-of-local-features 架构——几乎就是 BagNet-17 的 EuroSAT 版**，不是像素级独立加权。

## 1. 我们是否有 1×1 超越 3×3 的确切证据

有，但边界条件要说清楚。R2 在 stem s4 固定、QAT 配方固定、**MACs 匹配**（1.1-1.4M）下的对照：

| 配置 | kernels | channels | MACs | QAT clean |
|---|---|---|---|---|
| G1 | [3,3,1] | 16/32/64/128 | 1.41M | 93.83% |
| I1 | [1,3,1] | 16/32/64/128 | 1.14M | 93.85% |
| G2 | [3,1,1] | 32/64/128/256 | 1.19M | 94.11% |
| **H1** | **[1,1,1]** | 32/64/128/256 | 1.21M | **94.89%** |

注意这个对比的真实含义：**不是"同宽度下 1×1 kernel 更好"，而是"固定 MAC 预算下，把 3×3 的 9× 开销省下来换成宽度/深度更划算"**——1×1 的 H1 用省下的 MACs 把 channels 翻倍。另有 G3x（[3,3,1] 加宽到 3.08M MACs）93.70% 佐证：3×3 路线加预算也追不上。

三个诚实的保留：
- 每个点单 seed，test 5400 的 SE≈0.3%，~1pt 的差距显著但不巨大；
- 这是 **≤2M MACs + 8a8w12o QAT + 8×2 tile 对齐**这个小预算 regime 下的结论。**反例就在我们自己手里**：17M MACs 的 Model 4 E（全 3×3 MiniVGG）QAT 97.43%，比任何 1×1 小模型高 1.1pt——预算放开后 3×3 重新占优；
- 我们没跑过"J1 同宽度换 3×3"的直接消融（因为那样 MACs 会爆到 ~10M，不在搜索预算内）。

## 2. 这是否说明 EuroSAT 有奇妙特性

是，而且我们自己的 per-class 数据能精确刻画这个特性。比较两个冠军权重的 val per-class F1：

| 类别 | J1 (1×1, 1.38M) | Model4E (3×3, 17M) | 1×1 损失 |
|---|---|---|---|
| Highway | 94.04 | 97.67 | **−3.6** |
| River | 94.30 | 97.03 | **−2.7** |
| Pasture | 95.47 | 95.84 | −0.4 |
| PermanentCrop | 92.74 | 93.66 | −0.9 |
| HerbaceousVegetation | 94.69 | 94.49 | **+0.2** |
| AnnualCrop | 96.40 | 96.13 | **+0.3** |
| Residential/Industrial | 96.96/98.99 | 98.28/98.99 | −1.3/0 |

规律非常干净：**植被/纹理类（HerbVeg、AnnualCrop、Pasture、PermanentCrop）1×1 完全不输甚至反超；损失集中在 Highway/River 这类线状结构类**。即 EuroSAT ≈ 7 个纹理/光谱可分类 + 2 个结构类 + SeaLake/Forest 送分类。64×64 @ 10m 分辨率下，多数地物靠局部光谱纹理统计就够，全局布局几乎无用（GAP head 也印证）。这解释了为什么小感受野 bag-of-features 在这里只付出 ~1pt 总代价。

## 3. 社区/学术工作

我们的"发现"在文献里有完整的对应链条：

- **[BagNets (Brendel & Bethge, ICLR 2019)](https://arxiv.org/abs/1904.00760)**——最直接的对应。把 ResNet-50 的 3×3 换成 1×1，得到 9/17/33px 感受野的 bag-of-local-features 模型，**在 ImageNet 上 33px 仍达 top-5 87.6%**，并证明 VGG-16 等大网的决策与 BagNet logit 高度相关——主流 CNN 本来就主要靠局部弱统计特征决策。J1 本质上是同一个结论在遥感小图上的复现。
- **[Geirhos et al., ICLR 2019 (texture bias)](https://arxiv.org/abs/1811.12231)**：ImageNet 训练的 CNN 强烈偏向纹理而非形状。EuroSAT 多数类是纹理类，正好落在 CNN 的归纳偏置甜区。
- **NiN ([Lin et al. 2014](https://arxiv.org/abs/1312.4400))** 的 mlpconv 和 **[ConvMixer (Trockman & Kolter 2022)](https://arxiv.org/abs/2201.09792)**：patch embed + 1×1 通道混合的有效性早有验证；J1 的 stem s2 + 全 1×1 + GAP 在结构上就是个微型 ConvMixer。
- **EuroSAT 原论文 [Helber et al., JSTARS 2019](https://arxiv.org/abs/1709.00029)**：ResNet-50 即达 98.57%，此后 leaderboard 顶到 ~99.5%——数据集接近饱和是社区共识。[Clay foundation model 的讨论](https://github.com/Clay-foundation/model/discussions/269)里"冻结 encoder + FC 头 5 分钟训练到 98%"是这种饱和度的极端写照。
- **混淆结构与我们的完全吻合**：社区多次报告 EuroSAT 误差集中在 River–Highway 混淆和植被类互混（如 [arXiv:2510.15527](https://arxiv.org/html/2510.15527v1) 的 confusion 分析，River-Highway 27 个误分是 baseline 最大单一混淆）。同一篇的表述可以直接引用："infrastructure classes depend on spatial patterns (shape, layout, directional structure), while land cover classes depend on spectral signatures (color, texture)"——这正是我们 per-class 表显示的分界。
- 遥感领域更一般的共识（如 [Lu & Weng 类综述](https://arxiv.org/pdf/1807.05713v2)、[MDPI Remote Sensing 2020](https://www.mdpi.com/2072-4292/12/18/2907)）：中低分辨率 LULC 分类以光谱+纹理为主导，空间上下文是增量而非必需。

## 结论

1×1 胜出不是 EuroSAT 怪癖的孤立证据，而是三个因素叠加：(a) **MAC 约束下的预算重分配**——9× kernel 开销换成宽度更值（这是主因，也是可迁移的工程结论）；(b) **EuroSAT 纹理主导**——除 Highway/River 外局部特征足够，bag-of-features 代价小（BagNet 结论的复现）；(c) **硬件对齐**——1×1 展平=通道数，天然 8 倍数，对 8×2 tile 零浪费。三者里只有 (b) 是数据集特性，(a)(c) 换任何数据集在同样预算/硬件下大概率仍成立——但如果预算放开到 17M MACs，我们自己的数据已经表明 3×3 会赢回来。