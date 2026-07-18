# Compute vs Accuracy 散点图 — 数据与提示词

## 图意

横轴：单张图片计算量（MOPs，对数坐标）
纵轴：Top-1 准确率（%）
每模型三个点：◇ FP32 基准 / ● int8 QAT / ▲ osim 真机
点之间用垂直虚线连接，表示同一模型的不同精度阶段。

---

## 数据

| 模型 | 颜色 | MOPs/张 | FP32 基准 | int8 QAT (val) | osim 真机 | 参数量 | 光计算占比 | 样本量 |
|------|------|---------|-----------|----------------|-----------|--------|------------|--------|
| Model 2: SpaceNet V1 | `#E69F00` 暖橙 | 1.05M | 90.15% | 92.06% | **90.43%** | 268K | 90.65% | n=5400 全量 |
| Model 3: SpaceNet V2+KD | `#009E73` 经典绿 | 1.05M | 91.44% | 91.83% | **90.28%** | 268K | 90.65% | n=5400 全量 |
| Model 1-A: VGG+BN | `#0072B2` 深蓝 | 156.6M | 97.17% | 97.87% | **98.15%** | 2.39M | 97.74% | n=650 抽样 |
| Model 1-B: VGG+BN | `#56B4E9` 天蓝 | 156.6M | 97.17% | 98.02% | **97.54%** | 2.39M | 73.64% | n=650 抽样 |

- M2 和 M3 的 MOPs 相同（1.05M），x 位置重合 → 需要 jitter 错开（±7%）
- M1-A 和 M1-B 的 MOPs 相同（156.6M），x 位置重合 → 需要 jitter 错开（±7%）
- osim 标记用 ▲（三角形），加黑边突出；FP32 用 ◇（镂空菱形）；QAT 用 ●（半透明圆）
- 气泡大小 ∝ 参数量

---

## 精确坐标（jitter 后）

```
Model 2 (橙色):  x=9.77e5,  y_fp32=90.15, y_qat=92.06, y_osim=90.43
Model 3 (绿色):  x=1.12e6,  y_fp32=91.44, y_qat=91.83, y_osim=90.28
Model 1-A (深蓝): x=1.46e8, y_fp32=97.17, y_qat=97.87, y_osim=98.15
Model 1-B (天蓝): x=1.68e8, y_fp32=97.17, y_qat=98.02, y_osim=97.54
```

---

## 坐标轴

- x 轴：对数坐标，范围 5×10⁵ ~ 3.5×10⁸
- y 轴：线性坐标，范围 **86% ~ 100%**（不要从 70 开始，数据团在上面）
- MOPs 标注用科学计数法或 M 单位（1.05M / 156.6M）

---

## 标注

1. **≈149× compute reduction**：在 x=1.1M 到 x=150M 之间画双向箭头（高度 y≈87.5）
2. **SpaceNet 聚类卡片**（橙色/绿色区上方）：
   > SpaceNet (native int8 optical)
   > 0.268M params | 90.65% opt. MOPs
3. **VGG 聚类卡片**（蓝色区上方）：
   > VGG baseline
   > 2.39M params | 97.74% / 73.64% opt. MOPs
4. **页脚小字**：
   > osim = real optical-hardware simulation; Model 2/3 n=5400 (full test set), Model 1 n=650 (sampled). Source: EXPERIMENTS.md

---

## 图例（双栏，左下角）

左栏 — 标记类型：
- ◇ FP32 Baseline (Benchmark)
- ● int8 QAT (Quantization-Aware Training)
- ▲ osim (Real Optical-Hardware Inference)

右栏 — 模型：
- Model 2: SpaceNet V1 (0.268M) — 橙色
- Model 3: SpaceNet V2+KD (0.268M) — 绿色
- Model 1-A: VGG+BN (2.39M, 97.74% opt) — 深蓝
- Model 1-B: VGG+BN (2.39M, 73.64% opt) — 天蓝

---

## 视觉规范

- 配色：Okabe-Ito 色盲友好色板（橙 `#E69F00` / 绿 `#009E73` / 深蓝 `#0072B2` / 天蓝 `#56B4E9`）
- 网格：虚线 `#999999`，alpha 0.25
- 字号：标题 13pt bold，轴标签 12pt bold，刻度 11pt
- dpi ≥ 300
- 风格：学术干净风，白底，无背景色块

---

## 标题

> Compute vs. Accuracy — EuroSAT on Optical Computing (int8, Gazelle osimulator)
