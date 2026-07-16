# INT8 光计算容器推理日志

> 日期: 2026-07-10
> 模型: Model 2 SpaceNet V1 Phase4 v3 (INT8)
> 权重: spacenet_v1_phase4_v3_int8.pth (训练精度 93.11%)
> 容器: Gazelle osimulator (8×2 tile, 8a8w12o)

---

## Run 1: 初版 (有双重量化 Bug) — 84.43%

问题: OpticConv2d/OpticLinear 内先 `quantize_symmetric` (signed int8) 量化一次, 然后
`_matmul_real` 又 `quantize_to_int` (unsigned uint8) 再量化一次, 双重量化叠加噪声。
且 stem 层被错误转换为 OpticConv2d (训练时是 FP32)。

```
python optic_inference_int8.py
```

| 指标 | 值 |
|---|---|
| 准确率 | **84.43%** |
| 耗时 | 16949s (4.7h) |
| 引擎调用 | 32400 次 |
| 总 MACs | 6.56e+09 |

---

## Run 2: 修复版 (消除双重量化 + stem 保留电计算) — 93.28%

修复内容:
1. OpticConv2d/OpticLinear: 真实引擎路径跳过预量化, 传 raw float 给 `_matmul_real` 一次性量化
2. `build_optical_model(keep_first_conv_electronic=True)`: stem 保留 Conv2d (匹配训练时 first_conv_fp32=True)
3. `_matmul_fake` 使用 `quantize_symmetric` 正确位宽 (之前硬编码 int4)

```
python optic_inference_int8.py
```

```
Device: cpu
============================================================
  Optic-SpaceNet INT8: In-Container Optical Inference
  Model:  SpaceNet V1 Phase4 v3 (INT8, Gazelle-optimized)
  Weight: spacenet_v1_phase4_v3_int8.pth
  Mode:   Optic (osimulator, in-container)
  Batch:  1, full test set
============================================================

--- Loading Independent Test Set ---
Full dataset: 27000 imgs
Train (used): 21600 imgs  |  Val (used): 5400 imgs  |  Test (now): 5400 imgs
Test/Val overlap: 0 (zero = truly independent)
Classes: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Mode: Optic] Initializing Optical Engine (osimulator) in container...
[optic_layers] [OK] Real optical simulator loaded (osimulator)
wght_mapping_factor:  64
input_mapping_factor:  85
wght_mapping_factor:  64
input_mapping_factor:  64
wght_mapping_factor:  16
input_mapping_factor:  17
wght_mapping_factor:  16
input_mapping_factor:  16
wght_mapping_factor:  1
input_mapping_factor:  1
wght_mapping_factor:  1
input_mapping_factor:  1
[OpticalEngine] Using real optical simulator

============================================================
  Model 2 Phase4 v3 INT8  [Optic mode: osimulator]
============================================================

  [1/3] Creating standard model & loading weights...
  Weights loaded from: spacenet_v1_phase4_v3_int8.pth
  Params: 267,944

  Model 2 Phase4 v3 INT8 (Original FP32) 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [Conv2d]      stem.0                       3   1×1          3          8   37.5%
  [Conv2d]      stage1.0                     8   2×2         32         32   100.0%
  [Conv2d]      stage2.0                    16   2×2         64         64   100.0%
  [Conv2d]      stage3.0                    32   1×1         32         32   100.0%
  [Linear]      classifier.1               —     —           1024       1024   100.0%
  [Linear]      classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  [2/3] Converting to optical (OpticConv2d + OpticLinear, int8, stem=electronic)...

  Model 2 Phase4 v3 INT8 (Optical) 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [Conv2d]      stem.0                       3   1×1          3          8   37.5%
  [OpticConv2d] stage1.0                     8   2×2         32         32   100.0%
  [OpticConv2d] stage2.0                    16   2×2         64         64   100.0%
  [OpticConv2d] stage3.0                    32   1×1         32         32   100.0%
  [OpticLinear] classifier.1               —     —           1024       1024   100.0%
  [OpticLinear] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  [3/3] Evaluating via osimulator...
  [Model 2 Phase4 v3 INT8 optic] 5400 batches, report every 540 batch(es)
  [Model 2 Phase4 v3 INT8 optic]  540/5400 ( 10.0%) acc=93.89%  elapsed=1467s  ETA=13203s
  [Model 2 Phase4 v3 INT8 optic] 1080/5400 ( 20.0%) acc=93.89%  elapsed=2972s  ETA=11887s
  [Model 2 Phase4 v3 INT8 optic] 1620/5400 ( 30.0%) acc=93.83%  elapsed=4495s  ETA=10488s
  [Model 2 Phase4 v3 INT8 optic] 2160/5400 ( 40.0%) acc=93.80%  elapsed=6057s  ETA=9086s
  [Model 2 Phase4 v3 INT8 optic] 2700/5400 ( 50.0%) acc=93.59%  elapsed=7544s  ETA=7544s
  [Model 2 Phase4 v3 INT8 optic] 3240/5400 ( 60.0%) acc=93.55%  elapsed=9009s  ETA=6006s
  [Model 2 Phase4 v3 INT8 optic] 3780/5400 ( 70.0%) acc=93.47%  elapsed=10505s  ETA=4502s
  [Model 2 Phase4 v3 INT8 optic] 4320/5400 ( 80.0%) acc=93.43%  elapsed=11996s  ETA=2999s
  [Model 2 Phase4 v3 INT8 optic] 4860/5400 ( 90.0%) acc=93.29%  elapsed=13480s  ETA=1498s
  [Model 2 Phase4 v3 INT8 optic] 5400/5400 (100.0%) acc=93.28%  elapsed=14841s  ETA=0s
  [Model 2 Phase4 v3 INT8 optic] DONE — 5400 batches, acc=93.28%, total=14841s
  Optical Accuracy: 93.28%
  Optical Time:     14840.61s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 14534.282s, 总运算量: 5.15e+09 MACs
```

### 精度演进

| 阶段 | 进度 | 累计准确率 |
|---|---|---|
| 10% (540/5400) | elapsed=1467s | 93.89% |
| 20% (1080/5400) | elapsed=2972s | 93.89% |
| 30% (1620/5400) | elapsed=4495s | 93.83% |
| 40% (2160/5400) | elapsed=6057s | 93.80% |
| 50% (2700/5400) | elapsed=7544s | 93.59% |
| 60% (3240/5400) | elapsed=9009s | 93.55% |
| 70% (3780/5400) | elapsed=10505s | 93.47% |
| 80% (4320/5400) | elapsed=11996s | 93.43% |
| 90% (4860/5400) | elapsed=13480s | 93.29% |
| **100% (5400/5400)** | elapsed=14841s | **93.28%** |

### MOPs 光计算占比统计

| 层 | 类型 | 原始 MOPs | 计算位置 |
|---|---|---|---|
| stem.conv | Conv 3→8, 1×1 | 0.098M | ○ 电计算 (FP32) |
| stage1.conv | Conv 8→16, 2×2 | 0.524M | ◉ 光计算 (INT8) |
| stage2.conv | Conv 16→32, 2×2 | 0.131M | ◉ 光计算 (INT8) |
| stage3.conv | Conv 32→16, 1×1 | 0.033M | ◉ 光计算 (INT8) |
| fc1 | Linear 1024→256 | 0.262M | ◉ 光计算 (INT8) |
| fc2 | Linear 256→10 | 0.003M | ◉ 光计算 (INT8) |
| **合计** | | **1.051M** | |

- **光计算占比: 90.65%** | 电子计算: 9.35%
- 补零浪费: 0 (所有光计算层展平长度均为 8 的倍数)
- 硬件对齐率: 99.6%
- 总运算量: 5.15e+09 MACs (5400 images)

### 最终结果

| 指标 | 值 |
|---|---|
| 模型 | Model 2 SpaceNet V1 Phase4 v3 |
| 参数量 | 267,944 |
| 量化方案 | INT8 激活 + INT8 权重 (Gazelle 8a8w12o) |
| 训练精度 (QAT) | 93.11% (训练 val set, seed=42) |
| **光计算容器精度** | **93.28%** (独立测试集 5400 张, seed=42) |
| 光计算占比 | 90.65% |
| 总耗时 | 14841s (~4.1h) |
| 测试集规模 | 5400 张 (与训练 val 零重叠) |

### Bug 修复记录

| Bug | 影响 | 修复 |
|---|---|---|
| 双重量化: OpticConv2d 预量化 + `_matmul_real` 再量化 | 84.43% | `engine.use_real` 时跳过预量化, 传 raw float |
| `_matmul_fake` 硬编码 int4 | 模拟偏差 | 改用 `quantize_symmetric(x, bits=input_bit)` |
| stem 被转为 OpticConv2d (训练时是 FP32) | 首层噪声 | `keep_first_conv_electronic=True` |
