# MNIST 模型训练与光子芯片部署文档

## 1. 项目概述

本项目旨在训练一个用于 MNIST 手写数字识别的神经网络模型，并将其部署到光子芯片模拟器上进行推理。项目包含完整的数据获取、模型训练、权重量化和光子芯片模拟推理流程。

## 2. 项目结构

```
LT-Simulator/train/
├── data/                    # 原始 MNIST 数据集
│   └── MNIST/
│       └── raw/
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-labels-idx1-ubyte
│           ├── train-images-idx3-ubyte
│           └── train-labels-idx1-ubyte
├── mnist_data/              # 处理后的数据
│   ├── test_images.npy
│   ├── test_labels.npy
│   ├── train_images.npy
│   └── train_labels.npy
├── src/                     # 源代码
│   ├── download_mnist_api.py    # 从 Hugging Face API 下载数据
│   ├── load_dataset.py          # 加载 MNIST 数据集
│   ├── train_and_quantize.py    # 训练并量化模型
│   ├── train_with_api.py        # 使用 API 数据训练
│   ├── train_with_local_data.py # 使用本地数据训练
│   ├── run_simulator.py         # 运行光子芯片模拟器
│   ├── w1_int4.npy              # 量化后的第一层权重
│   ├── w2_int4.npy              # 量化后的第二层权重
│   ├── w1_int4_local.npy        # 本地数据训练的第一层权重
│   └── w2_int4_local.npy        # 本地数据训练的第二层权重
└── .gitignore
```

## 3. 数据获取

项目提供了两种数据获取方式：

### 3.1 从 Hugging Face API 下载数据

使用 `download_mnist_api.py` 脚本从 Hugging Face API 下载 MNIST 数据：

```bash
python src/download_mnist_api.py
```

该脚本会：
- 从 Hugging Face API 分批下载指定数量的 MNIST 图像和标签
- 将图像转换为灰度图并保存为 numpy 数组
- 将数据保存到 `mnist_data` 目录

### 3.2 从本地加载数据

使用 `load_dataset.py` 脚本从本地加载 MNIST 数据：

```bash
python src/load_dataset.py
```

该脚本会：
- 使用 `datasets` 库加载 MNIST 数据集
- 将数据处理为 numpy 数组
- 保存到 `mnist_data` 目录

## 4. 模型结构

项目使用了一个极其精简的 MLP（多层感知机）模型，专为光子芯片设计：

```python
class PhotonicMLP(nn.Module):
    def __init__(self):
        super(PhotonicMLP, self).__init__()
        # 展平后 28x28 = 784。隐藏层设为 64。
        # 必须设置 bias=False，因为光子芯片示例中只有单纯的矩阵乘法
        self.fc1 = nn.Linear(784, 64, bias=False)
        self.fc2 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

### 模型特点：
- 无偏置设计：符合光子芯片的硬件限制
- 精简结构：仅包含两个全连接层
- 输入尺寸：28x28 像素的灰度图像（展平为 784 维向量）
- 输出尺寸：10 个类别（0-9 的数字）

## 5. 训练流程

项目提供了三种训练方式：

### 5.1 标准训练与量化 (train_and_quantize.py)

```bash
python src/train_and_quantize.py
```

训练步骤：
1. 下载并加载 MNIST 数据集
2. 训练浮点模型（3 个 epochs）
3. 提取并量化权重为 4-bit 有符号整数
4. 模拟光子芯片推理并验证准确率

### 5.2 使用 API 数据训练 (train_with_api.py)

```bash
python src/train_with_api.py
```

训练步骤：
1. 从 Hugging Face API 获取 MNIST 数据
2. 创建自定义数据集
3. 训练浮点模型（10 个 epochs）
4. 提取并量化权重
5. 模拟光子芯片推理并验证准确率

### 5.3 使用本地数据训练 (train_with_local_data.py)

```bash
python src/train_with_local_data.py
```

训练步骤：
1. 加载本地 MNIST 数据
2. 创建自定义数据集
3. 训练浮点模型（3 个 epochs）
4. 提取并量化权重
5. 模拟光子芯片推理并验证准确率

## 6. 权重量化

### 6.1 权重量化函数

```python
def quantize_weight_int4(weight_tensor):
    """将浮点权重映射到 4-bit 有符号整数 [-8, 7]"""
    w_numpy = weight_tensor.detach().cpu().numpy()
    max_val = np.max(np.abs(w_numpy))
    # 缩放因子：最大绝对值映射到 7
    scale = 7.0 / max_val
    # 缩放、四舍五入、截断
    w_q = np.clip(np.round(w_numpy * scale), -8, 7).astype(np.int32)
    return w_q
```

### 6.2 输入量化函数

```python
def quantize_input_uint4(input_numpy):
    """将 [0.0, 1.0] 的输入图像映射到 4-bit 无符号整数 [0, 15]"""
    return np.clip(np.round(input_numpy * 15.0), 0, 15).astype(np.int32)
```

### 6.3 权重转置

**重要**：PyTorch 的权重形状是 `[out, in]`，而光子芯片需要 `[in, out]`，因此在量化前需要转置权重：

```python
w1_float = model.fc1.weight.T  # 转置权重
w2_float = model.fc2.weight.T
```

### 6.4 激活值量化

光子芯片要求第二层的输入也必须是 0-15 的整数，因此需要对激活值进行量化：

```python
# 模拟激活层 (CPU计算: ReLU)
h1_sim = np.maximum(0, y1_sim)

# 隐藏层到第二层的量化
max_h1 = np.max(h1_sim)
scale_h1 = 15.0 / max_h1  # 把最大的激活值映射到 15
h1_uint4 = np.clip(np.round(h1_sim * scale_h1), 0, 15).astype(np.int32)
```

## 7. 光子芯片模拟

使用 `run_simulator.py` 脚本运行光子芯片模拟器：

```bash
python src/run_simulator.py
```

模拟步骤：
1. 加载量化后的权重
2. 加载测试数据并量化
3. 分批处理测试数据：
   - 第一层光子计算
   - 上报光子计算量
   - CPU 激活与再量化
   - 第二层光子计算
   - 上报光子计算量
   - 统计准确率
4. 输出最终测试结果

## 8. 关键参数

### 8.1 训练参数
- **批大小**：256
- **学习率**：0.005
- **优化器**：Adam
- **损失函数**：CrossEntropyLoss
- **训练轮数**：
  - 标准训练：3 epochs
  - API 数据训练：10 epochs

### 8.2 量化参数
- **权重精度**：4-bit 有符号整数 [-8, 7]
- **输入精度**：4-bit 无符号整数 [0, 15]
- **输出精度**：12-bit

### 8.3 缩放因子

训练过程中会计算并输出第一层激活值的缩放因子 `scale_h1`，这是光子模拟器必需的参数：

```
【重要参数】第一层激活值的缩放因子 scale_h1 = 0.008460
```

## 9. 性能指标

### 9.1 准确率目标
- 量化后准确率需达到 **90% 以上**（训练验证）
- 光子模拟器准确率需达到 **85% 以上**（最终目标）

### 9.2 预期结果

使用标准训练流程，模型在量化后应能达到：
- 训练集准确率：> 95%
- 测试集准确率：> 90%
- 光子模拟器准确率：> 85%

## 10. 部署步骤

1. **数据准备**：
   - 运行 `download_mnist_api.py` 或 `load_dataset.py` 获取数据

2. **模型训练**：
   - 运行 `train_and_quantize.py`、`train_with_api.py` 或 `train_with_local_data.py` 训练模型
   - 记录训练过程中输出的 `scale_h1` 值

3. **权重部署**：
   - 训练完成后，量化权重会保存为 `.npy` 文件
   - 在 `run_simulator.py` 中更新权重文件路径和 `scale_h1` 值

4. **模拟器运行**：
   - 运行 `run_simulator.py` 启动光子芯片模拟
   - 查看终端输出的准确率和性能指标

## 11. 故障排除

### 11.1 准确率不足
- **解决方案**：增加训练轮数，调整学习率，或尝试不同的训练数据量

### 11.2 API 下载失败
- **解决方案**：检查网络连接，或使用本地数据训练方式

### 11.3 模拟器错误
- **解决方案**：确保权重文件路径正确，`scale_h1` 值与训练时一致，输入数据格式正确

## 12. 总结

本项目实现了一个完整的 MNIST 模型训练与光子芯片部署流程，包括：

- 多种数据获取方式
- 专为光子芯片设计的精简模型
- 完整的权重量化流程
- 光子芯片模拟推理
- 性能评估与验证

通过本项目，您可以了解如何将传统神经网络模型适配到光子芯片硬件上，并通过量化技术确保模型在硬件限制下仍能保持良好的性能。