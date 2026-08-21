# Architecture / 架构设计

## Table of Contents / 目录

- [Overview / 概述](#overview--概述)
- [Model Architecture / 模型架构](#model-architecture--模型架构)
- [Training Flow / 训练流程](#training-flow--训练流程)
- [Inference Pipeline / 推理流程](#inference-pipeline--推理流程)
- [Key Design Decisions / 关键设计决策](#key-design-decisions--关键设计决策)

---

## Overview / 概述

This project trains a tiny photonic MLP for MNIST using Quantization-Aware Training (QAT). The architecture separates modeling, training, quantization methods, quantized numpy inference, photonic simulator, and robustness testing into distinct modules under `src/`.

本项目为光计算 MNIST 图像分类任务实现量化感知训练（QAT）。架构将建模、训练、量化方法、NumPy 量化推理、光子模拟器与鲁棒性测试分离为 `src/` 下的独立模块。

---

## Model Architecture / 模型架构

All models are 2-layer MLPs (`bias=False`) with ReLU activation after the hidden layer:

```
Input (784) -> FC1 -> ReLU -> FC2 -> Output (10)
```

| Variant | Hidden Dim | Quantization |
|---------|-----------|--------------|
| `PhotonicMLP` | 64 | None (FP32) |
| `PhotonicMLP_STEQ` | 64 | STE fake-quant (static scale) |
| `PhotonicMLP_LSQPlus` | 128 | LSQ+ (learned scale + zero-point) |
| `PhotonicMLP_DSQ` | 128 | DSQ (soft tanh quant + temperature annealing) |

**Why `bias=False`?** Photonic MAC arrays compute matrix-vector products without additive bias terms.

---

## Training Flow / 训练流程

```
MNIST data
    |
    v
DataLoader (local .npy or torchvision)
    |
    v
Model (base / STE / LSQ+ / DSQ)
    |
    v
Common training loop (train_epoch + evaluate)
    |
    v
Export quantized weights & params -> artifacts/{ste,lsqplus,dsq}/
```

- **STE**: static scale computed from max abs value; noise injected during training.
- **LSQ+**: 4 independent `LSQPlusQuantizer`s (input, w1, h1, w2) with per-parameter learning rates.
- **DSQ**: 4 independent `DSQQuantizer`s with temperature annealing (soft -> hard across epochs).

---

## Inference Pipeline / 推理流程

Each method has a dedicated NumPy inference engine (`src/inference/numpy_*.py`) that replicates the quantized forward pass without PyTorch:

```
images (numpy) -> quantize input -> int4 MAC -> ReLU -> quantize activation -> int4 MAC -> argmax
```

The photonic simulator (`src/inference/simulator.py`) replaces MAC operations with `optical_mac_tiling`, which models an 8×2 photonic array tiling behavior using INT32 accumulation.

鲁棒性测试 (`src/inference/robustness.py`) 向权重注入高斯噪声并测量准确率随噪声水平的衰减曲线。

---

## Key Design Decisions / 关键设计决策

1. **Unified `src/` package** / 统一源码包
   - Eliminates duplication between `src_raw/` and `src_dsqlsq/`.
   - Bottom-up extraction: quantization → models → data → training → inference → utils → scripts.

2. **Separate hidden dims** / 分离隐藏层维度
   - Baseline STE and FP use `hidden_dim=64` (legacy behavior).
   - LSQ+ and DSQ use `hidden_dim=128` (improved accuracy for advanced QAT).

3. **Thin script wrappers** / 薄脚本封装
   - All entry points live in `src/scripts/` and are runnable directly with `python src/scripts/xxx.py`.
   - They handle `sys.path` insertion so the package imports work without installation.

4. **Artifact directory per method** / 每种方法独立产物目录
   - `artifacts/ste/`, `artifacts/lsqplus/`, `artifacts/dsq/` keep weights and params isolated.
