# LSQ+ INT8 光计算容器推理日志

> 日期: 2026-07-10
> 模型: Model 2 SpaceNet V1 LSQ+ (INT8, 可学习 scale/zp, 92.80%)
> 权重: spacenet_v1_lsq_int8.pth
> 文件: optic_inference_lsq.py

---

## 配置

| 参数 | 值 |
|---|---|
| QAT 模块 | optic_qat_lsq |
| 权重位宽 | int8 |
| 激活位宽 | int8 |
| Stem | 电计算 (LSQConv2d, `_use_lsq=False`) |
| 其余层 | LSQ 量化 → engine.matmul → real osimulator |
| Patch 层数 | 5 (3 LSQConv2d + 2 LSQLinear, stem 跳过) |

## 开发过程

### 第一版: 通用 optical 路径 → 60.00%

使用 `build_optical_model` 将 LSQ 层转为 OpticConv2d/OpticLinear,
`quantize_to_int` 重新计算 scale, 完全破坏了 LSQ 学到的量化网格。

### 根因分析

LSQ+ 的 scale/zp 是训练学出来的, 不能重新计算:
- `in_scale` 跨通道差异高达 6722x (classifier.1: 0.000007 ~ 0.047909)
- `in_zp` 非零 (范围 -0.19 ~ +0.20)
- `weight_zp` 非零 (~0.02-0.04)
- 用 per-tensor 均值近似 → 15% (随机)

LSQ vs STE 的本质区别:
| | STE | LSQ+ |
|---|---|---|
| 量化参数 | 推理时计算 | 训练时学习, 不能改 |
| 权重 | 对噪声鲁棒 | 与特定 scale/zp 绑定 |
| 部署 | 通用光学路径 | 专用路径 (保留 scale/zp) |

### 第二版: LSQ 专用路径 → 96.00%

**方案**: Monkey-patch LSQ 层, 保留 LSQ 量化, matmul 送 osimulator:
1. 加载 LSQ 模型 + `prepare_model_lsq` + `enable_qat`
2. 对每个 LSQConv2d/LSQLinear (跳过 stem):
   - 用 `lsq_quantize` + 学到的 scale/zp 量化输入和权重
   - im2col (Conv) 或 reshape (Linear)
   - `engine.matmul(quantize_inputs=False)` → osimulator
   - col2im (Conv)
3. LSQ 量化后的值是粗粒度网格, `_matmul_real` 的再量化基本无损

```
python optic_inference_lsq.py --quick 50
```

| 指标 | 值 |
|---|---|
| 准确率 | **96.00%** (50 张) |
| 训练参考 | 92.80% int8 LSQ+ |
| 单张耗时 | ~3.2s |
| 总耗时 | 159s |
| 引擎调用 | 250 次 |
| 总 MACs | 4.76e+07 |

路径: `LSQ quant (learned scales) → _matmul_real requant → real osimulator → dequant`

---

## 待办

- [ ] 全量 5400 张验证 (~4h)

## 备注

LSQ+ 模型的优势: 可学习 scale/zp 可直接导出为 Gazelle 硬件配置寄存器, 无需软件量化。
LSQ+ 的 FP32 模式精度极低 (~62%) 是正常的 — 权重过度特化适配 int8 量化。

容器文件 `optic_inference_lsq.py` 中 `_patch_lsq_conv2d` / `_patch_lsq_linear`
实现了 LSQ 专用 monkey-patch 逻辑。
