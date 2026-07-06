# Phase 4 技术设计: 基于初赛验证方法的 CNN 光计算量化迁移

> 参考: 初赛 MNIST 量化感知训练方案 (STE / LSQ+ / DSQ)
> 目标: 将 EuroSAT CNN 模型以最小精度损失迁移到 int4 光计算硬件

---

## 目录

1. [初赛经验总结](#1-初赛经验总结)
2. [当前方案 vs 初赛方案差距分析](#2-差距分析)
3. [Phase 4 技术方案](#3-phase-4-技术方案)
4. [量化方法详解](#4-量化方法详解)
5. [模型架构修改](#5-模型架构修改)
6. [训练配置](#6-训练配置)
7. [推理管线](#7-推理管线)
8. [预期结果](#8-预期结果)

---

## 1. 初赛经验总结

### 1.1 初赛成果

| 方法 | MNIST NumPy | 光子硬件 | 噪声容差 |
|------|-------------|----------|----------|
| **STE + 噪声注入** | **97.03%** | 96.70% | σ=0.70 |
| LSQ+ | 93.18% | 96.60% | σ=1.00 |
| DSQ | 94.80% | 94.50% | σ=1.50 |

**STE 精度最高 (97.03%)，几乎无量化损失。**

### 1.2 初赛关键技术要素

**要素 1: 非对称量化方案 (最关键!)**

```
激活值 (ReLU 后): uint4 [0, 15]   ← 非负值用无符号，16 级全利用
权重:            int4 [-8, 7]    ← 有正有负用有符号
```

而我们的 Phase 1-3 全部使用 **对称 int4 [-8, 7] 量化激活值**。ReLU 输出全是非负数，对称量化浪费了一半范围 ([-8, -1] 永远用不到)，实际只用了 8 个级别，等价于 int3。

**要素 2: STE + 噪声注入**

```python
# 静态 scale (非动态，稳定)
scale = max_abs / 7

# 训练时向量化后的权重注⼊⾼斯噪声，模拟硬件噪声并增强鲁棒性
noise = torch.randn_like(weight) * 0.05 * scale
weight_noisy = weight + noise
w_q = fake_quantize(weight_noisy, scale)
```

噪声注入起到了正则化作用 → STE 的泛化能力反而最强 (97.03% > 93-95%)。

**要素 3: LSQ+ (可学习 scale + zero-point)**

```python
# LSQ+ = LSQ + zero_point 学习
scale = nn.Parameter(init_from_stats)      # 可学习
zero_point = nn.Parameter(init_from_stats)  # 可学习

# 独立学习率: scale/zp 使用基础 lr 的 0.1 倍
opt_weights = Adam(model.weights, lr=0.005)
opt_quant = Adam([scale, zero_point], lr=0.0005)

# 梯度缩放
grad_scale /= sqrt(N * qmax)
```

**要素 4: 逐层激活值重量化**

光计算推理管线中，每一层的输出 (ReLU 后) 都需要重新量化为 uint4 才能送入下一层。训练时应该在伪量化中模拟这个行为:

```
x_uint4 → MatMul(w_int4) → ReLU → quantize_to_uint4 → 下一层
```

**要素 5: bias=False**

光计算硬件只做 MAC，不支持加法偏置。

### 1.3 为什么 STE+噪声 精度最高？

这是一个反直觉的结果。分析原因:

1. **噪声正则化**: 训练时注入的高斯噪声等价于数据增强，提高了泛化能力
2. **静态 scale 稳定**: 不需要学习 scale，训练更稳定
3. **简单有效**: 对于 MNIST 级别的任务，复杂的 LSQ+ 可能过参数化
4. **硬件一致性**: 静态量化的行为更接近实际硬件

---

## 2. 差距分析: 当前方案 vs 初赛方案

| 维度 | Phase 1-3 (我们的) | 初赛方案 | 影响 |
|------|-------------------|----------|------|
| **激活量化** | int4 [-8,7] 对称 | **uint4 [0,15] 无符号** | 浪费一半量化级别 (8 vs 16) |
| **权重量化** | int4 [-8,7] | int4 [-8,7] | 一致 |
| **scale 方式** | 动态 (max/7 每步) / LSQ | **静态 max/7** (STE) | 动态不稳定 |
| **训练噪声** | 无 | **高斯噪声 std=0.05*scale** | 缺失正则化 |
| **逐层重量化** | 隐式 (下一层输入量化) | **显式 (匹配硬件管线)** | 训练-推理不一致 |
| **量化参数 lr** | 与权重相同 | **0.1x 独立学习率** (LSQ+) | scale 学习不稳定 |
| **bias** | 有 bias | **bias=False** | 硬件不匹配 |
| **BN 处理** | 融合或保留 | MLP 无 BN | 需要额外处理 |

**最大的三个差距**:
1. ❌ 激活量化用 int4 而非 uint4 → 损失 1 bit 精度
2. ❌ 无训练噪声注入 → 泛化差、过拟合
3. ❌ 无逐层输出重量化 → 训练与推理管线不一致

---

## 3. Phase 4 技术方案

### 3.1 整体设计

```
┌─────────────────────────────────────────────────────────┐
│  Phase 4: 初赛验证方法 + CNN 适配                         │
│                                                           │
│  量化方案:                                                 │
│    激活值: uint4 [0, 15] (非对称, 匹配 ReLU 输出)         │
│    权重:   int4 [-8, 7]  (对称, 匹配有符号分布)           │
│                                                           │
│  训练方法 (两种可选):                                     │
│    A. STE + 噪声注入 (推荐首选, 简单稳定)                 │
│    B. LSQ+ (learnable scale + zero_point, 更高上限)      │
│                                                           │
│  关键特性:                                                 │
│    ✓ 逐层输出重量化 (匹配光计算推理管线)                  │
│    ✓ bias=False (匹配光计算硬件)                          │
│    ✓ 训练噪声注入 (正则化 + 硬件鲁棒性)                   │
│    ✓ 量化参数独立学习率 (LSQ+ 模式)                       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 伪量化公式

**激活值量化 (uint4, [0, 15])**:

```
scale = max(x) / 15               # 非对称, 无符号
x_uint4 = clamp(round(x / scale), 0, 15)
x_dq = x_uint4 * scale            # 反量化
```

**权重量化 (int4, [-8, 7])**:

```
scale = max(|w|) / 7              # 对称, 有符号
w_int4 = clamp(round(w / scale), -8, 7)
w_dq = w_int4 * scale
```

**STE 梯度**: `x_dq` 对 `x` 的梯度 = 1 (直通), `x_dq` 对 `scale` 的梯度 = 0 (静态)

**LSQ+ 梯度** (如果使用):
- scale 的梯度: 截断内外分开处理, 乘以 1/sqrt(N*qmax)
- zero_point 的梯度: 类似处理

### 3.3 逐层前向管线 (训练 = 推理)

```
输入图像 (FP32, 归一化)
    │
    ▼ quantize_to_uint4
x0_uint4 [0, 15]
    │
    ▼ MatMul(w0_int4)  ← STE/LSQ+ 伪量化
y0_fp32
    │
    ▼ ReLU
h0_fp32 (≥ 0)
    │
    ▼ quantize_to_uint4    ← 逐层输出重量化!
h0_uint4 [0, 15]
    │
    ▼ MatMul(w1_int4)
y1_fp32
    │
    ▼ ... (重复)
    │
    ▼ 最后一层不量化 (logits 直接输出)
output (FP32)
```

---

## 4. 量化方法详解

### 4.1 方法 A: STE + 噪声注入 (推荐)

```
训练时:
  scale_w = max(|W|) / 7
  W_noisy = W + N(0, 0.05 * scale_w)    ← 噪声注入
  W_q = clamp(round(W_noisy / scale_w), -8, 7) * scale_w

  scale_x = max(X) / 15                   ← 静态 scale
  X_q = clamp(round(X / scale_x), 0, 15) * scale_x

推理时:
  scale_w 固定, 不注入噪声
  scale_x 固定

特点:
  - scale 在训练开始时从权重统计计算, 固定不变
  - 噪声 std = 0.05 * scale_w
  - 简单、稳定、泛化好 (初赛最佳)
```

**超参数**:
- 噪声系数: 0.05
- 训练 epochs: 60-80 (CNN 比 MLP 需要更多)
- 优化器: Adam, lr=0.001

### 4.2 方法 B: LSQ+ (可选, 更高上限)

```
可学习参数:
  - scale (nn.Parameter)
  - zero_point (nn.Parameter) → 支持非对称量化

初始化:
  - scale = (mean + 3*std) / qmax (从数据统计)
  - zero_point = 0 (有符号权重) / -min(x)/scale (无符号激活)

训练:
  x_int = clamp(round(x/scale + zero_point), qmin, qmax)
  x_q = (x_int - zero_point) * scale

梯度:
  - 截断内的值: grad_scale = (x/scale).round() - x/scale
  - 截断外的值: grad_scale = sign(x) * qmax
  - 梯度缩放: 1 / sqrt(N * qmax)
  - 截断区间外 grad_x = 0, 区间内 grad_x = 1

独立学习率:
  - 权重: lr = 0.001
  - 量化参数 (scale, zero_point): lr = 0.0001 (0.1x)
```

---

## 5. 模型架构修改

### 5.1 适配光计算硬件的修改

| 修改 | 原因 |
|------|------|
| **bias=False** (所有 Conv/Linear) | 光计算只做 MAC, 无偏置加法 |
| **BN → 无 / 融合** | 训练时保留 BN (稳定), 推理时融合到权重 |
| **首层输入量化** | RGB [0,255] 归一化后量化为 uint4 |
| **逐层输出重量化** | 每层 ReLU 后量化为 uint4 送入下层 |
| **末层不量化** | 分类 logits 保持浮点精度 |

### 5.2 Model 1 (Baseline VGG) 修改

```python
# 修改前 (Phase 3):
nn.Conv2d(3, 32, 3, padding=1)  # bias 隐含 True
# 修改后:
nn.Conv2d(3, 32, 3, padding=1, bias=False)  # 明确 False

# 首层: float32 (不量化, 处理原始 RGB)
# 末层 (classifier.4): float32 (不量化, 输出 logits)
# 中间层: uint4 激活 + int4 权重 + STE/LSQ+
```

### 5.3 Model 2/3 (SpaceNet) 修改

```python
# BN 处理: 训练时保留, 推理前融合
# 融合时机: 训练完成后
# 融合公式: W_fused = W * γ/σ, b_fused = β - γ*μ/σ
# 融合后 bias 丢弃 (光计算不支持)

# 或: 直接训练 bias=False + 无 BN
```

---

## 6. 训练配置

### 6.1 STE + 噪声模式 (推荐)

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Epochs | 60 | 80 | 100 |
| 优化器 | Adam | Adam | Adam |
| LR (权重) | 0.001 | 0.001 | 0.001 |
| 调度器 | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| 噪声 std | 0.05*scale | 0.05*scale | 0.05*scale |
| 激活量化 | uint4 [0,15] | uint4 [0,15] | uint4 [0,15] |
| 权重量化 | int4 [-8,7] | int4 [-8,7] | int4 [-8,7] |
| 逐层重量化 | ✓ | ✓ | ✓ |
| bias | False | False | False |

### 6.2 LSQ+ 模式 (可选)

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| LR (权重) | 0.001 |
| LR (scale, zp) | 0.0001 (0.1x) |
| 调度器 | CosineAnnealing |
| Epochs | 80/100/120 |
| 初始化 | 首次 forward 从数据统计计算 |

---

## 7. 推理管线

训练完成后，推理分为两步:

### 7.1 PyTorch 伪量化验证 (本地)

```python
model.eval()
# QAT 层自动在 eval 模式下施加 uint4/int4 伪量化
# 测量模拟 int4 精度 (与训练时 eval 一致)
accuracy = evaluate(model, val_loader)
```

### 7.2 光计算模拟器验证 (Docker)

```python
# 1. 提取 int4 权重 + 量化参数
for layer in model.qat_layers:
    w_int4 = (layer.weight / layer.weight_scale).round().clamp(-8, 7)
    save_npy(w_int4, f"w{idx}_int4.npy")
    save_npy(layer.weight_scale, f"s_w{idx}.npy")

# 2. 逐层光计算推理
for batch in test_data:
    # 第一层
    x_uint4 = quantize_input_uint4(batch)  # [0, 15]
    y1 = optical_matmul(x_uint4, w1_int4)  # 光计算 MAC
    y1_fp = y1 * (s_in * s_w1)              # 反量化
    h1 = relu(y1_fp)                         # CPU ReLU
    h1_uint4 = quantize_uint4(h1, s_h1)     # 重新量化

    # 第二层
    y2 = optical_matmul(h1_uint4, w2_int4)
    ...

    # 最后一层 → argmax
    output = argmax(y3_fp)
```

---

## 8. 预期结果

### 8.1 精度预期

基于初赛经验 (STE 在 MNIST 上 FP32→int4 几乎无损) 和 Phase 2 结果，预期:

| 模型 | FP32 | Phase 2 最佳 | Phase 4 STE+noise (预期) | Phase 4 LSQ+ (预期) |
|------|------|-------------|--------------------------|---------------------|
| Model 1 (VGG) | 97.17% | 91.17% | **94-96%** | **93-95%** |
| Model 2 (SN V1) | 90.15% | 81.20% | **86-89%** | **85-88%** |
| Model 3 (SN V2 KD) | 91.44% | 83.26% | **88-91%** | **87-90%** |

### 8.2 关键改进点及预期收益

| 改进 | 预期收益 | 原因 |
|------|----------|------|
| 激活 uint4 | +2-3% | 16 级 vs 8 级, 信息翻倍 |
| 噪声注入 | +1-2% | 正则化, 初赛验证有效 |
| 逐层重量化 | +1-2% | 训练推理一致 |
| bias=False | ~0% | 硬件匹配, 精度影响小 |
| LSQ+ 独立 lr | +1% | 更稳定的 scale 学习 |

---

*文档版本: v1.0 | 参考: 初赛 01_设计报告.pdf, 02_验证报告.pdf, 03_技术数据.pdf*
