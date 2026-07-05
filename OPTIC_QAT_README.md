# Optic-SpaceNet QAT: 光计算量化感知训练迁移

> **从 PTQ (Post-Training Quantization) 到 QAT (Quantization-Aware Training)**
>
> 基于 int4 精度 8×2 光学矩阵乘法器的神经网络迁移方案

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [QAT vs PTQ 核心原理](#2-qat-vs-ptq-核心原理)
3. [技术方案](#3-技术方案)
4. [文件结构](#4-文件结构)
5. [快速开始](#5-快速开始)
6. [详细使用指南](#6-详细使用指南)
7. [Docker 部署](#7-docker-部署)
8. [超参数调优指南](#8-超参数调优指南)
9. [预期结果与基准](#9-预期结果与基准)
10. [FAQ](#10-faq)

---

## 1. 背景与动机

### 1.1 项目目标

将三个 EuroSAT 遥感图像分类模型迁移到 8×2 光学矩阵乘法硬件上运行。光计算具有高能效、低延迟的优势，但精度限制为 **int4** (4-bit)，导致直接量化 (PTQ) 精度大幅下降。

### 1.2 三个模型概览

| 模型 | 架构 | 参数量 | Float32 准确率 | 硬件对齐率 | 训练方式 |
|------|------|--------|----------------|------------|----------|
| Model 1 (Baseline VGG) | Mini-VGG, 3×3 conv | 2,386,986 | 97.17% | ~99.8% | 标准训练 60 epochs |
| Model 2 (SpaceNet V1) | 硬件感知对齐 CNN | 268,210 | 90.15% | ~96.3% | 独立训练 80 epochs |
| Model 3 (SpaceNet V2) | KD 蒸馏 CNN | 268,210 | 91.44% | ~96.3% | KD 100 epochs (教师: ResNet-18, 97.83%) |

### 1.3 第一次迁移失败原因分析

第一次迁移采用了 **PTQ (Post-Training Quantization)** 方案:

```
训练:    Float32 权重 → Float32 训练 → 保存 Float32 权重
推理:    Float32 权重 → 量化为 int4 → 光计算推理 ❌ 精度下降 10-20%
```

**失败原因:**

1. **模型从未见过量化噪声**: FP32 训练时精度远高于 int4，模型在优化时利用了浮点的高精度空间。突然将权重和激活量化到 int4 (仅 16 个离散值)，模型完全无法适应。

2. **量化误差逐层累积**: 6 层卷积 + 2 层全连接，每层 int4 量化都会引入误差，深层误差被逐层放大。

3. **int4 表示能力有限**: 对称 int4 只有 16 个离散值 (-8, -7, ..., 0, ..., 7)，信息损失远大于 int8 (256 个值)。

---

## 2. QAT vs PTQ 核心原理

### 2.1 PTQ (Post-Training Quantization) — 之前的方案

```
┌──────────────────────────────────────────────────┐
│  Float32 Training                                 │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐         │
│  │ Conv2d  │──▶│  ReLU   │──▶│ Conv2d  │──▶ ...  │
│  │ (FP32)  │   │         │   │ (FP32)  │         │
│  └─────────┘   └─────────┘   └─────────┘         │
│                                                     │
│  Save FP32 weights → quantize at inference → INT4  │
│  ⚠️ Model never saw INT4 during training            │
└──────────────────────────────────────────────────┘
```

**问题**: 训练和推理之间存在精度不匹配。模型在 FP32 空间找到的最优解在 INT4 空间可能完全失效。

### 2.2 QAT (Quantization-Aware Training) — 新方案

```
┌──────────────────────────────────────────────────┐
│  QAT Training                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐         │
│  │ Input   │   │  Fake   │   │ Conv2d  │         │
│  │ (FP32)  │──▶│  INT4   │──▶│ (FP32)  │──▶ ...  │
│  └─────────┘   │ Quant   │   └─────────┘         │
│                └────┬────┘                         │
│                     │ STE (梯度直通)                │
│                     ▼                               │
│                Gradient flows to FP32 weights      │
│                                                     │
│  Weights learn to be robust to INT4 quantization   │
│  Save FP32 weights → quantize at inference → INT4  │
│  ✅ Model adapted to INT4 during training           │
└──────────────────────────────────────────────────┘
```

**核心机制:**

1. **Fake Quantization (伪量化)**: 前向传播时，将 FP32 值先量化到 INT4 再反量化回 FP32
   ```
   Forward:  x_fp32 → round(x/scale)·clamp(-8,7) → ×scale → x_fake_quantized
   效果:     x_fake_quantized 的值域与真实 int4 量化一致
   ```

2. **STE (Straight-Through Estimator)**: 反向传播时，梯度直接跳过量化节点
   ```
   Backward: ∂L/∂x = ∂L/∂x_fake_quantized  (identity gradient)
   效果:     梯度更新 FP32 权重，使权重向量化友好的方向移动
   ```

3. **训练结果**: FP32 权重学会在 int4 量化后仍能正确分类

### 2.3 数学推导

**对称 int4 量化:**
- 值域: [-8, 7], 共 16 个离散值
- 量化: `x_int = round(x / scale), clamped to [-8, 7]`
- 反量化: `x_dq = x_int * scale`
- 尺度: `scale = max(|x|) / 7`

**STE 梯度:**
```
∂L/∂x = ∂L/∂x_dq · ∂x_dq/∂x
其中 ∂x_dq/∂x ≈ 1 (STE 假设，忽略四舍五入的不可微性)
```

---

## 3. 技术方案

### 3.1 量化方案设计

| 参数 | 选择 | 理由 |
|------|------|------|
| 位宽 | int4 (对称) | 光计算硬件标准精度 |
| 值域 | [-8, 7] | 对称量化，零中心 |
| 激活量化 | Per-channel (C_in) | 模拟 DAC 逐通道量化 |
| 权重量化 | Per-output-channel (C_out) | 模拟光交叉阵列逐列存储 |
| Scale 计算 | 动态 (每步重算) | 鲁棒，无需学习额外参数 |
| BN 处理 | 融合到 Conv | QAT 标准做法，减少量化节点 |

### 3.2 QAT 层设计

#### QATConv2d

```python
class QATConv2d(nn.Module):
    """
    包裹标准 nn.Conv2d，在前向传播中插入伪 int4 量化。

    Forward:
      1. x_q = fake_int4_quantize(x, per_channel=True, dim=1)  # 量化输入
      2. w_q = fake_int4_quantize(w, per_channel=True, dim=0)  # 量化权重
      3. out = F.conv2d(x_q, w_q, bias)                        # FP32 卷积

    与 OpticConv2d 的关系:
      - QATConv2d (训练):  F.conv2d + 伪量化 → 梯度更新权重
      - OpticConv2d (推理): im2col → 光 matmul → col2im → 实际 int4 推理
    """
```

#### QATLinear

```python
class QATLinear(nn.Module):
    """
    包裹标准 nn.Linear，在前向传播中插入伪 int4 量化。

    Forward:
      1. x_q = fake_int4_quantize(x, per_channel=True, dim=-1) # 量化输入
      2. w_q = fake_int4_quantize(w, per_channel=True, dim=0)  # 量化权重
      3. out = F.linear(x_q, w_q, bias)                        # FP32 全连接
    """
```

### 3.3 QAT 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 加载预训练 FP32 权重                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  model.load_state_dict(torch.load("xxx.pth"))         │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▼                                  │
│  Phase 2: 准备 QAT 模型                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2a. model.eval() + run 1 batch → BN running stats   │   │
│  │  2b. fuse_conv_bn() → 融合 Conv+BN 到一个 Conv       │   │
│  │  2c. replace Conv2d→QATConv2d, Linear→QATLinear      │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▼                                  │
│  Phase 2b: 校准 (可选)                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  跑 3-5 个 batch 预热量化 scale                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▼                                  │
│  Phase 3: QAT 微调                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  for epoch in range(QAT_EPOCHS):                     │   │
│  │      model.train()  # QAT layers auto-fake-quantize  │   │
│  │      output = model(input)                           │   │
│  │      loss = criterion(output, target)                │   │
│  │      loss.backward()  # STE gradients                │   │
│  │      optimizer.step()                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▼                                  │
│  Phase 4: 保存与评估                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  torch.save(model.state_dict(), "xxx_qat.pth")       │   │
│  │  compare_qat_vs_float() → QAT vs Float 精度对比      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 推理时的衔接

QAT 训练产出的是 **FP32 权重**，但权重已经对 int4 量化具有鲁棒性。

推理时有两种方式:

**方式 A — 使用 optic_layers.py (光计算模拟):**
```python
from optic_layers import OpticalEngine, build_optical_model

engine = OpticalEngine()
model = OpticSpaceNetV1()
model.load_state_dict(torch.load("spacenet_v1_qat.pth"))  # QAT 权重
build_optical_model(model, engine)  # 替换为 OpticConv2d/OpticLinear

# 推理: im2col → 实际 int4 量化 → 光 matmul → col2im
output = model(input_image)
```

**方式 B — 直接部署到光硬件:**
```python
# 1. 从 QAT 权重中提取 int4 权重矩阵
w_int4 = (model.weight / scale).round().clamp(-8, 7).to(torch.int32)

# 2. 加载到光计算硬件
optical_hardware.load_weights(w_int4)

# 3. 推理时量化输入
x_int4 = quantize_input(image)  # FP32 → int4

# 4. 光计算推理
output_int = optical_hardware.matmul(x_int4, w_int4)
output_fp32 = dequantize(output_int)
```

---

## 4. 文件结构

```
train-test/
├── optic_qat.py                    # [核心] QAT 模块
│   ├── fake_int4_quantize()        #   STE 伪 int4 量化函数
│   ├── QATConv2d                   #   QAT 卷积层
│   ├── QATLinear                   #   QAT 全连接层
│   ├── fuse_conv_bn()              #   Conv+BN 融合
│   ├── prepare_qat_model()         #   模型 QAT 化
│   ├── enable_qat() / disable_qat()#   QAT 模式开关
│   ├── calibrate_qat_model()       #   量化校准
│   ├── evaluate_model()            #   评估工具
│   ├── compare_qat_vs_float()      #   QAT vs Float 对比
│   └── compute_alignment_ratio()   #   硬件对齐率
│
├── model1_baseline_qat.py          # Model 1 QAT 微调脚本
│   └── BaselineVGG → 加载 baseline_vgg.pth → QAT → baseline_vgg_qat.pth
│
├── model2_spacenet_v1_qat.py       # Model 2 QAT 微调脚本
│   └── OpticSpaceNetV1 → 加载 spacenet_v1.pth → QAT → spacenet_v1_qat.pth
│
├── model3_spacenet_v2_qat.py       # Model 3 QAT 微调脚本
│   ├── Mode A: 标准 QAT (CrossEntropy)
│   └── Mode B: KD+QAT (蒸馏损失 + QAT)
│   └── 加载 spacenet_v2_distilled.pth → QAT → spacenet_v2_qat.pth
│
├── optic_layers.py                 # [保留] 光计算推理模拟 (PTQ 方案)
├── optic_inference.py              # [保留] 光计算推理评估
├── noise_robustness.py             # [保留] 噪声鲁棒性测试
│
├── model1_baseline.py              # [原始] FP32 标准训练
├── model2_spacenet_v1.py           # [原始] FP32 标准训练
├── model3_spacenet_v2.py           # [原始] FP32 KD 训练
│
├── OPTIC_QAT_README.md             # [本文档] QAT 完整说明
└── log.md                          # [原始] 训练日志
```

### 文件依赖关系

```
model1_baseline.py ──→ baseline_vgg.pth ──→ model1_baseline_qat.py ──→ baseline_vgg_qat.pth
                                                                              │
model2_spacenet_v1.py → spacenet_v1.pth → model2_spacenet_v1_qat.py → spacenet_v1_qat.pth
                                                                              │
model3_spacenet_v2.py → spacenet_v2_distilled.pth                             │
                              └──→ model3_spacenet_v2_qat.py ──→ spacenet_v2_qat.pth
                                                                              │
                                                                              ▼
                                                                     optic_inference.py
                                                                     (使用 optic_layers.py
                                                                      加载 QAT 权重推理)
```

---

## 5. 快速开始

### 5.1 环境要求

```bash
# Python 3.8+
pip install torch torchvision numpy matplotlib

# 可选: 光计算模拟器 (仅推理时需要)
# pip install osimulator entrance
```

### 5.2 一键运行 (Model 1 为例)

```bash
# Step 1: 确保已完成 FP32 标准训练 (产出 baseline_vgg.pth)
python model1_baseline.py

# Step 2: QAT 微调
python model1_baseline_qat.py

# Step 3: 使用 QAT 权重进行光计算推理评估
python optic_inference.py  # 需要手动修改加载的权重路径为 *_qat.pth
```

### 5.3 运行所有模型

```bash
# Model 1 - Baseline VGG QAT
python model1_baseline_qat.py

# Model 2 - SpaceNet V1 QAT
python model2_spacenet_v1_qat.py

# Model 3 - SpaceNet V2 QAT (标准模式)
python model3_spacenet_v2_qat.py

# Model 3 - SpaceNet V2 QAT (KD 联合模式, 推荐)
python model3_spacenet_v2_qat.py --use_kd
```

---

## 6. 详细使用指南

### 6.1 optic_qat.py — 核心模块使用

#### 基础用法

```python
import torch
import torch.nn as nn
from optic_qat import (
    fake_int4_quantize,
    QATConv2d,
    QATLinear,
    prepare_qat_model,
    enable_qat,
    disable_qat,
    evaluate_model,
    compare_qat_vs_float,
)

# 1. 创建标准模型
model = YourModel()

# 2. 加载 FP32 预训练权重
model.load_state_dict(torch.load("your_model.pth"))

# 3. 转换为 QAT 模型
model.eval()
# 跑一个 batch 更新 BN running stats
with torch.no_grad():
    model(sample_input)
prepare_qat_model(model, fuse_bn=True)

# 4. QAT 微调
model.train()  # QAT layers auto-apply fake quantization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(20):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)  # 伪量化自动施加
        loss = criterion(output, batch.label)
        loss.backward()        # STE 梯度
        optimizer.step()

# 5. 保存 QAT 权重
torch.save(model.state_dict(), "your_model_qat.pth")

# 6. 诊断: QAT vs Float 对比
comparison = compare_qat_vs_float(model, val_loader, device)
print(f"QAT accuracy: {comparison['qat_mode']['accuracy']:.2%}")
print(f"Float accuracy: {comparison['float_mode']['accuracy']:.2%}")
print(f"Accuracy gap: {comparison['accuracy_gap']:.2%}")
```

#### 高级用法

```python
# 手动控制 QAT 模式
disable_qat(model)  # 使用全精度 (用于诊断)
enable_qat(model)   # 恢复伪量化

# BN 融合 (在 prepare_qat_model 之前手动执行)
from optic_qat import fuse_conv_bn
fused_conv = fuse_conv_bn(original_conv, original_bn)

# 自定义量化函数
from optic_qat import fake_int4_quantize
# 逐张量量化
x_q_per_tensor = fake_int4_quantize(x, per_channel=False)
# 逐通道量化
x_q_per_channel = fake_int4_quantize(x, per_channel=True, ch_dim=1)

# 硬件对齐率分析
from optic_qat import compute_alignment_ratio, print_alignment_detail
ratio = compute_alignment_ratio(model)
print_alignment_detail(model, "MyModel")

# 独立 QAT 层使用
conv = nn.Conv2d(3, 16, 3, padding=1)
qat_conv = QATConv2d(conv)
# 训练时自动伪量化
qat_conv.train()
out = qat_conv(torch.randn(4, 3, 32, 32))
# 推理时可选择关闭
qat_conv.eval()
qat_conv.disable_qat()
out_fp32 = qat_conv(torch.randn(4, 3, 32, 32))
```

### 6.2 Model 1: Baseline VGG QAT

```bash
python model1_baseline_qat.py
```

**特点:**
- 参数量大 (2.39M)，QAT 微调 15 epochs 即可收敛
- 3×3 卷积为主，第一层对齐率低 (27→32) 但 ops 少，影响有限
- 使用 CosineAnnealing 学习率调度

### 6.3 Model 2: SpaceNet V1 QAT

```bash
python model2_spacenet_v1_qat.py
```

**特点:**
- 参数量小 (268K)，硬件对齐率高 (~96.3%)
- 20 epochs QAT 微调
- stem 层 (3→8) 是唯一未对齐的层

### 6.4 Model 3: SpaceNet V2 QAT

```bash
# Mode A: 标准 QAT
python model3_spacenet_v2_qat.py

# Mode B: KD + QAT 联合 (推荐!)
python model3_spacenet_v2_qat.py --use_kd
```

**两种模式对比:**

| 特性 | Mode A (Standard QAT) | Mode B (KD + QAT) |
|------|----------------------|-------------------|
| 损失函数 | CrossEntropy | α·KL + (1-α)·CE |
| 需要教师模型 | 否 | 是 (teacher_resnet18.pth) |
| 训练速度 | 较快 | 较慢 (教师前向) |
| 预期精度 | ~87-89% | ~89-91% |
| 适用场景 | 教师模型不可用 | 追求最佳精度 |

**Mode B 原理:**
KD + QAT 联合微调将蒸馏损失和量化适应结合在一个阶段:
1. 教师提供软标签引导 (保持蒸馏的精度优势)
2. 伪量化让模型同时适应 int4 精度
3. 两个目标联合优化，避免分阶段训练的次优解

---

## 7. Docker 部署

### 7.1 Dockerfile 示例

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# 安装依赖
RUN pip install torchvision numpy matplotlib tqdm

# 复制项目文件
COPY model1_baseline_qat.py ./
COPY model2_spacenet_v1_qat.py ./
COPY model3_spacenet_v2_qat.py ./
COPY optic_qat.py ./
COPY optic_layers.py ./
COPY optic_inference.py ./
COPY noise_robustness.py ./

# 复制预训练 FP32 权重
COPY baseline_vgg.pth ./
COPY spacenet_v1.pth ./
COPY spacenet_v2_distilled.pth ./
COPY teacher_resnet18.pth ./

# 复制数据
COPY data/ ./data/

# 默认命令
CMD ["python", "model1_baseline_qat.py"]
```

### 7.2 构建与运行

```bash
# 构建镜像
docker build -t optic-qat:latest .

# 运行 Model 1 QAT
docker run --gpus all -v $(pwd)/output:/workspace/output \
    optic-qat:latest python model1_baseline_qat.py

# 运行 Model 2 QAT
docker run --gpus all -v $(pwd)/output:/workspace/output \
    optic-qat:latest python model2_spacenet_v1_qat.py

# 运行 Model 3 QAT (KD mode)
docker run --gpus all -v $(pwd)/output:/workspace/output \
    optic-qat:latest python model3_spacenet_v2_qat.py --use_kd

# 交互式调试
docker run --gpus all -it -v $(pwd)/output:/workspace/output \
    optic-qat:latest /bin/bash
```

### 7.3 Docker Compose (多模型并行)

```yaml
version: '3.8'
services:
  qat-model1:
    build: .
    command: python model1_baseline_qat.py
    volumes:
      - ./output:/workspace/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  qat-model2:
    build: .
    command: python model2_spacenet_v1_qat.py
    volumes:
      - ./output:/workspace/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  qat-model3:
    build: .
    command: python model3_spacenet_v2_qat.py --use_kd
    volumes:
      - ./output:/workspace/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

```bash
# 三模型并行训练
docker-compose up -d

# 查看日志
docker-compose logs -f
```

---

## 8. 超参数调优指南

### 8.1 QAT 学习率

QAT 微调的学习率选择至关重要:

| 情况 | 推荐 lr | 说明 |
|------|---------|------|
| 大模型 (VGG, 2.4M) | 1e-5 ~ 5e-5 | 权重已接近最优，小步微调 |
| 小模型 (SpaceNet, 268K) | 5e-5 ~ 1e-4 | 需要稍大的调整空间 |
| 收敛不理想 | 提高至 2e-4 | 但注意不要破坏原有权重结构 |
| 训练不稳定 | 降低至 5e-5 | 量化噪声可能导致梯度波动 |

### 8.2 QAT 训练轮数

```python
# 经验法则: 大模型少训，小模型多训
if model_params > 1_000_000:
    QAT_EPOCHS = 10-15   # 大模型快速适应
else:
    QAT_EPOCHS = 15-25   # 小模型需要更多迭代
```

**监控指标:**
- QAT loss 应该在 2-3 个 epoch 内稳定
- 如果 QAT accuracy 持续低于 float accuracy 超过 5%，考虑:
  - 降低学习率
  - 增加训练轮数
  - 检查 BN 融合是否正确

### 8.3 BN 融合注意事项

```python
# ✅ 正确: 融合前确保 BN 有 running stats
model.eval()
with torch.no_grad():
    for images, _ in train_loader:
        model(images)  # 至少跑一个 batch
        break
prepare_qat_model(model, fuse_bn=True)

# ❌ 错误: BN 未初始化就融合
model = MyModel()
model.load_state_dict(weights)  # BN running stats 可能是随机的
prepare_qat_model(model, fuse_bn=True)  # 融合结果错误!
```

### 8.4 诊断工具

```python
# 1. 检查 QAT vs Float 精度差
comparison = compare_qat_vs_float(model, val_loader, device)
# 目标: accuracy_gap < 2%

# 2. 逐层检查量化误差
for name, module in model.named_modules():
    if isinstance(module, QATConv2d):
        # 查看该层量化后的权重分布
        w_orig = module.weight.data
        w_q = fake_int4_quantize(w_orig, per_channel=True, ch_dim=0)
        mse = (w_orig - w_q).pow(2).mean()
        print(f"{name}: MSE={mse:.6f}, max_err={(w_orig-w_q).abs().max():.4f}")

# 3. 检查梯度流
# 如果 QAT 层梯度过小，说明 STE 可能有问题
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

---

## 9. 预期结果与基准

### 9.1 Model 1: Baseline VGG

| 阶段 | 精度 | 说明 |
|------|------|------|
| FP32 训练 | 97.17% | 标准 60 epochs 训练 |
| PTQ (int4 直接量化) | ~80-87% | **大幅下降 10-17%** |
| **QAT (微调后 int4)** | **~95-97%** | **接近 FP32 水平** [推荐] |

### 9.2 Model 2: SpaceNet V1

| 阶段 | 精度 | 说明 |
|------|------|------|
| FP32 训练 | 90.15% | 独立训练 80 epochs |
| PTQ (int4 直接量化) | ~75-80% | **大幅下降 10-15%** |
| **QAT (微调后 int4)** | **~87-89%** | **接近 FP32 水平** [推荐] |

### 9.3 Model 3: SpaceNet V2 (KD)

| 阶段 | 精度 | 说明 |
|------|------|------|
| 教师 ResNet-18 | 97.83% | 仅用于蒸馏，不部署 |
| FP32 KD 学生 | 91.44% | 100 epochs 蒸馏 |
| PTQ (int4 直接量化) | ~77-82% | **大幅下降 9-14%** |
| **QAT Standard Mode** | **~87-89%** | 标准 QAT 微调 |
| **QAT KD Joint Mode** | **~89-91%** | **KD+QAT 联合，最佳** [推荐] |

### 9.4 精度-效率权衡总结

```
                    精度 ↑
                      │
    FP32  ● (最高精度)
                      │  ← QAT 成功: 精度几乎不降
    QAT   ◉ (int4 推理)
                      │
                      │  ← PTQ 失败区域 (10-20% 精度损失)
    PTQ   ○ (int4 直接量化)
                      │
                      └──────────────────→ 推理能耗 (越低越好)
                        光计算 >> GPU >> CPU
```

**关键结论: QAT 是光计算迁移的必要步骤。** PTQ 无法接受精度损失，
QAT 通过让模型在训练时感知量化，实现了精度和效率的兼得。

---

## 10. FAQ

### Q1: QAT 训练后的权重还是 FP32，为什么推理时变成 int4 就不会掉精度了？

A: QAT 训练的本质是让 FP32 权重"学会"在 int4 量化后仍能正确工作。可以类比为: 一个人先在安静环境中学会演讲 (FP32 训练)，然后戴着耳塞练习 (QAT 微调)，最终即使在嘈杂环境中也能讲得很好 (int4 推理)。而 PTQ 相当于直接让安静环境训练的人去嘈杂环境演讲，自然发挥不佳。

### Q2: QAT 训练需要多久？

A: QAT 是"微调"而非从头训练。一般 10-25 epochs，在 CPU 上每个模型约 10-30 分钟，在 GPU 上仅需 3-10 分钟。对比原始 FP32 训练的 60-100 epochs，QAT 非常快。

### Q3: QAT 训练后可以恢复为标准 FP32 推理吗？

A: 可以。QAT 权重仍是 FP32，只是更"量化友好"。你可以:
- 用 `disable_qat()` 关闭伪量化，进行标准 FP32 推理
- 用 `optic_layers.py` 进行 int4 光计算推理
- 导出为 ONNX 等格式部署

### Q4: 为什么 Model 2/3 的对齐率是 96.3% 而不是 100%？

A: stem 层的 Conv(3→8, 1×1) 的 patch_len = 3×1×1 = 3，需要补零到 8。但这一层的 ops 极少（仅占总 ops 的 ~2%），影响几乎可忽略。

### Q5: QAT 训练时需要光计算硬件/模拟器吗？

A: 不需要。QAT 在标准 PyTorch 上运行，使用 F.conv2d / F.linear 进行模拟。只需在推理时使用光计算硬件或 optic_layers.py 模拟器。

### Q6: 如何知道 QAT 是否成功？

A: 使用 `compare_qat_vs_float()` 工具。如果 QAT 模式精度与 Float 模式精度差距 < 2%，说明 QAT 成功。

### Q7: QAT 可以和知识蒸馏一起用吗？

A: 可以，而且推荐! Model 3 的 `--use_kd` 模式就是 KD + QAT 联合训练，能同时获得蒸馏的精度优势和量化的鲁棒性。

---

## 附录 A: 光计算硬件背景

### A.1 8×2 光学矩阵乘法器

- **8-wide vector unit**: 每次处理 8 个元素的向量点积
- **2 parallel channels**: 2 路并行计算
- **int4 precision**: 权重和输入均为 4-bit 整数
- **操作**: X(m×k) @ W(k×n), 其中 k 维度需要对齐到 8

### A.2 对齐率计算

```
对于 Conv2d(C_in, C_out, k×k):
  patch_len = C_in × k_h × k_w
  如果 patch_len % 8 ≠ 0:
    padded_len = ceil(patch_len / 8) × 8
    对齐率 = patch_len / padded_len
  否则:
    对齐率 = 100%
```

### A.3 噪声类型

光计算硬件存在多种物理噪声，已在 `optic_layers.py` 和 `noise_robustness.py` 中建模:
- **Gaussian Readout**: 探测器热噪声
- **Phase Noise**: MZI 相位误差
- **Shot Noise**: 光子计数统计噪声
- **Crosstalk**: 波导间串扰

QAT 训练主要解决量化噪声，其他噪声可以通过 `noise_robustness.py` 进行鲁棒性评估。

---

## 附录 B: 与原有文件的兼容性

| 操作 | 使用文件 | 权重来源 |
|------|----------|----------|
| FP32 标准训练 | model{1,2,3}_*.py | 随机初始化 |
| **QAT 微调** | **model{1,2,3}_*_qat.py** | **FP32 预训练权重** |
| 光计算推理模拟 | optic_inference.py + optic_layers.py | QAT 或 FP32 权重 |
| 噪声鲁棒性测试 | noise_robustness.py + optic_layers.py | 任意权重 |

**推荐工作流:**
```
FP32 训练 → QAT 微调 → 光计算推理评估 → 噪声鲁棒性测试
```

---

*文档版本: v1.0 | 最后更新: 2026-07-05*
