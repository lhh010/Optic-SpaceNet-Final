# CODE_INDEX / 代码索引

## Table of Contents / 目录

- [Overview / 概述](#overview--概述)
- [Documentation / 文档](#documentation--文档)
- [Directory Structure / 目录结构](#directory-structure--目录结构)
- [Quantization Methods / 量化方法](#quantization-methods--量化方法)
- [Models / 模型](#models--模型)
- [Data & Training / 数据与训练](#data--training--数据与训练)
- [Inference / 推理](#inference--推理)
- [Runnable Scripts / 可执行脚本](#runnable-scripts--可执行脚本)
- [Artifacts & Data / 产物与数据](#artifacts--数据)

---

## Overview / 概述

This index maps every significant file in the repository to its purpose and public API. The project implements Quantization-Aware Training (QAT) for a photonic MNIST classifier with three methods: baseline STE, LSQ+, and DSQ. The codebase has been refactored into a clean `src/` package with separation of concerns.

本索引将仓库中每个重要文件映射到其用途和公共 API。本项目为光计算 MNIST 分类器实现了量化感知训练（QAT），包含基础 STE 以及两种改进方案（LSQ+ 和 DSQ）。代码已重构为结构清晰的 `src/` 包，职责分离明确。

---

## Documentation / 文档

| File | Purpose / 用途 |
|------|----------------|
| [`README.md`](README.md) | Project overview and quick start / 项目概述与快速开始 |
| [`ARCHITECTURE/README.md`](ARCHITECTURE/README.md) | System design and data flow / 系统设计与数据流 |
| [`CODE_INDEX.md`](CODE_INDEX.md) | This file / 本文件 |

---

## Directory Structure / 目录结构

```
MNIST-train/
├── data/                      # Raw and processed datasets / 原始与预处理数据
│   ├── raw/MNIST/raw/         # Original MNIST ubyte files / 原始 ubyte 文件
│   └── processed/             # Preprocessed .npy arrays / 预处理后的 numpy 数据
├── docs/                      # Documentation / 文档
│   ├── CODE_INDEX.md          # This file / 本文件
│   └── legacy/                # Legacy docs from old src_dsqlsq / 旧版文档
├── src/                       # Unified source package / 统一源码包
│   ├── quantization/          # QAT method implementations / QAT 方法实现
│   ├── models/                # Neural architectures / 神经网络架构
│   ├── data/                  # Data loaders / 数据加载器
│   ├── training/              # Shared training loops / 共享训练循环
│   ├── inference/             # NumPy engines, simulator, robustness / 推理引擎、模拟器、鲁棒性测试
│   ├── utils/                 # I/O helpers / I/O 工具
│   └── scripts/               # Runnable entry points / 可执行入口脚本
└── artifacts/                 # Model weights, results, and plots / 模型权重、结果与图表
    ├── ste/                   # STE baseline artifacts / STE 基线产物
    ├── lsqplus/               # LSQ+ artifacts / LSQ+ 产物
    ├── dsq/                   # DSQ artifacts / DSQ 产物
    ├── robustness/            # Robustness test results / 鲁棒性测试结果
    └── plots/                 # Visualization outputs / 可视化输出
```

---

## Quantization Methods / 量化方法

| File | Purpose / 用途 | Key API / 关键 API |
|------|----------------|-------------------|
| `src/quantization/ste.py` | STE fake-quantization core / STE 伪量化核心 | `FakeQuantizeSTE` |
| `src/quantization/lsqplus.py` | LSQ+ fake-quantization core / LSQ+ 伪量化核心 | `LSQPlusFakeQuantize`, `LSQPlusQuantizer` |
| `src/quantization/dsq.py` | DSQ fake-quantization core / DSQ 伪量化核心 | `DSQFakeQuantize`, `DSQQuantizer` |

---

## Models / 模型

| File | Purpose / 用途 | Key API / 关键 API |
|------|----------------|-------------------|
| `src/models/base.py` | Full-precision 2-layer MLP / 全精度两层 MLP | `PhotonicMLP(hidden_dim=64)` |
| `src/models/qat_steq.py` | STE QAT model with noise injection / 带噪声注入的 STE QAT 模型 | `PhotonicMLP_STEQ(hidden_dim=64, noise_std=0.05)` |
| `src/models/qat_lsqplus.py` | LSQ+ QAT model / LSQ+ QAT 模型 | `PhotonicMLP_LSQPlus(hidden_dim=128)` |
| `src/models/qat_dsq.py` | DSQ QAT model with temperature annealing / 带温度退火的 DSQ QAT 模型 | `PhotonicMLP_DSQ(hidden_dim=128)` |

---

## Data & Training / 数据与训练

| File | Purpose / 用途 | Key API / 关键 API |
|------|----------------|-------------------|
| `src/data/loaders.py` | MNIST data loaders (local & torchvision) / MNIST 数据加载器 | `get_mnist_loaders_local()`, `get_mnist_loaders_torchvision()`, `MNISTDataset` |
| `src/data/download.py` | Hugging Face API download helpers / HF API 下载工具 | `download_mnist_split()`, `load_mnist_to_numpy()` |
| `src/training/common.py` | Shared training & evaluation loops / 共享训练与评估循环 | `train_epoch()`, `evaluate()` |
| `src/utils/io.py` | Weight export/import utilities / 权重导出导入工具 | `export_weights_steq()`, `export_weights_lsqplus()`, `export_weights_dsq()`, `load_weights_and_params()` |

---

## Inference / 推理

| File | Purpose / 用途 | Key API / 关键 API |
|------|----------------|-------------------|
| `src/inference/numpy_steq.py` | NumPy inference for STE/FP / STE/FP 的 NumPy 推理 | `run_inference(images, labels, w1, w2, scale_h1)` |
| `src/inference/numpy_lsqplus.py` | NumPy inference for LSQ+ (with ZP compensation) / LSQ+ 的 NumPy 推理（含 ZP 补偿） | `run_inference(images, labels, w1, w2, quant_params)` |
| `src/inference/numpy_dsq.py` | NumPy inference for DSQ / DSQ 的 NumPy 推理 | `run_inference(images, labels, w1, w2, quant_params)` |
| `src/inference/simulator.py` | Photonic 8x2 MAC tiling simulator / 光子 8x2 MAC 分块模拟器 | `optical_mac_tiling()`, `compute_optical_ratio()` |
| `src/inference/robustness.py` | Shared robustness test framework / 共享鲁棒性测试框架 | `run_robustness_test()`, `add_noise_to_weights()` |

---

## Runnable Scripts / 可执行脚本

All scripts can be run directly with `python src/scripts/<name>.py`.
所有脚本均可直接通过 `python src/scripts/<name>.py` 运行。

| File | Purpose / 用途 |
|------|----------------|
| `src/scripts/train_fp.py` | Full-precision training / 全精度训练 |
| `src/scripts/train_qat_steq.py` | STE QAT training / STE QAT 训练 |
| `src/scripts/train_qat_lsqplus.py` | LSQ+ QAT training / LSQ+ QAT 训练 |
| `src/scripts/train_qat_dsq.py` | DSQ QAT training / DSQ QAT 训练 |
| `src/scripts/run_simulator_steq.py` | STE photonic simulator / STE 光子模拟器 |
| `src/scripts/run_simulator_lsqplus.py` | LSQ+ photonic simulator / LSQ+ 光子模拟器 |
| `src/scripts/run_simulator_dsq.py` | DSQ photonic simulator / DSQ 光子模拟器 |
| `src/scripts/test_robustness_steq.py` | STE robustness test / STE 鲁棒性测试 |
| `src/scripts/test_robustness_lsqplus.py` | LSQ+ robustness test / LSQ+ 鲁棒性测试 |
| `src/scripts/test_robustness_dsq.py` | DSQ robustness test / DSQ 鲁棒性测试 |
| `src/scripts/compare_methods.py` | Visualization comparing quantization curves / 量化方法可视化对比 |

---

## Artifacts & Data / 产物与数据

### Datasets / 数据集

| Path | Description / 描述 |
|------|-------------------|
| `data/raw/MNIST/raw/` | Original MNIST ubyte files (`train-images-idx3-ubyte`, etc.) / 原始 MNIST ubyte 文件 |
| `data/processed/train_images.npy` | 60,000 training images (float32, normalized) / 训练图像 |
| `data/processed/train_labels.npy` | 60,000 training labels / 训练标签 |
| `data/processed/test_images.npy` | 10,000 test images (float32, normalized) / 测试图像 |
| `data/processed/test_labels.npy` | 10,000 test labels / 测试标签 |

### Model Artifacts / 模型产物

| Path | Description / 描述 |
|------|-------------------|
| `artifacts/ste/w1_int4.npy`, `w2_int4.npy` | STE quantized weights / STE 量化权重 |
| `artifacts/lsqplus/w1_int4_lsq_plus.npy`, `w2_int4_lsq_plus.npy` | LSQ+ quantized weights / LSQ+ 量化权重 |
| `artifacts/dsq/w1_int4_dsq.npy`, `w2_int4_dsq.npy` | DSQ quantized weights / DSQ 量化权重 |
| `artifacts/lsqplus/lsq_plus_quant_params.npy` | LSQ+ learned scales and zero-points / LSQ+ 学习到的 scale 和 zero-point |
| `artifacts/dsq/dsq_quant_params.npy` | DSQ learned scales / DSQ 学习到的 scale |

### Results & Plots / 结果与图表

| Path | Description / 描述 |
|------|-------------------|
| `artifacts/robustness/robustness_test_steq.npy` | STE robustness results / STE 鲁棒性测试结果 |
| `artifacts/robustness/robustness_test_lsq_plus.npy` | LSQ+ robustness results / LSQ+ 鲁棒性测试结果 |
| `artifacts/robustness/robustness_test_dsq.npy` | DSQ robustness results / DSQ 鲁棒性测试结果 |
| `artifacts/plots/quantization_comparison.png` | Visualization from `compare_methods.py` / 方法对比可视化图 |
