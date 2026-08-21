# MNIST-train / 光计算 MNIST 分类

## Table of Contents / 目录

- [Overview / 概述](#overview--概述)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Directory Structure / 目录结构](#directory-structure--目录结构)
- [Documentation / 文档](#documentation--文档)

---

## Overview / 概述

MNIST-train implements Quantization-Aware Training (QAT) for a photonic MNIST classifier. It supports three quantization methods:

- **STE** (Straight-Through Estimator) — baseline static-scale QAT
- **LSQ+** (Learned Step Size Quantization Plus) — learned scale and zero-point
- **DSQ** (Differentiable Soft Quantization) — soft tanh quantization with temperature annealing

MNIST-train 为光计算 MNIST 分类器实现量化感知训练（QAT），支持三种量化方法：基础 STE、LSQ+ 和 DSQ。

---

## Quick Start / 快速开始

```bash
# 1. Install dependencies / 安装依赖
pip install torch torchvision numpy matplotlib

# 2. Train models / 训练模型
python src/scripts/train_fp.py
python src/scripts/train_qat_steq.py
python src/scripts/train_qat_lsqplus.py
python src/scripts/train_qat_dsq.py

# 3. Run photonic simulator / 运行光子模拟器
python src/scripts/run_simulator_steq.py
python src/scripts/run_simulator_lsqplus.py
python src/scripts/run_simulator_dsq.py

# 4. Robustness test / 鲁棒性测试
python src/scripts/test_robustness_steq.py
python src/scripts/test_robustness_lsqplus.py
python src/scripts/test_robustness_dsq.py
```

---

## Directory Structure / 目录结构

```
MNIST-train/
├── data/              # MNIST datasets (raw + processed)
├── artifacts/         # Model weights, quant params, plots, robustness results
├── docs/              # Documentation
│   ├── README.md
│   ├── CODE_INDEX.md
│   ├── ARCHITECTURE/
│   └── BUGS/
└── src/               # Unified source package
    ├── models/
    ├── quantization/
    ├── data/
    ├── training/
    ├── inference/
    ├── utils/
    └── scripts/
```

---

## Documentation / 文档

- [`CODE_INDEX.md`](CODE_INDEX.md) — File-to-purpose mapping / 文件用途索引
- [`ARCHITECTURE/README.md`](ARCHITECTURE/README.md) — System design / 系统设计
- [`BUGS/README.md`](BUGS/README.md) — Bug records / 错误记录
