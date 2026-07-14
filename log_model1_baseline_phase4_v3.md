# Model 1 (Baseline VGG) Phase4 v3 int8 重训日志 (2026-07-14)

> Bug #11 split 修复后重训 (eurosat_split): train 16200 / val 5400 / 留出 test 5400 (三段严格不相交)
> 变体 A (conv1_1 FP32): Int8 val 97.87% / test 97.89% | 光计算占比 97.74% | 280.0min
> 变体 B (conv1_1 + conv3_2 FP32): Int8 val 98.02% / test 97.96% | 光计算占比 73.64% | 276.1min
> **test∩train=0 (干净), test≈val → 无泄漏、泛化良好** (旧 leaky-split 版 test 虚高 99.96% 已作废)

| 变体 | 电计算层 | Int8 (val) | FP32 (val) | Int8 (test) | FP32 (test) | 量化损失 (test) | test vs val (Int8) |
|---|---|---|---|---|---|---|---|
| A | conv1_1 | 97.87% | 97.93% | 97.89% | 97.91% | +0.02% | +0.02% |
| B | conv1_1 + conv3_2 | 98.02% | 97.96% | 97.96% | 98.04% | +0.07% | −0.06% |

---

## 训练 — 变体 A

`python -u model1_baseline_phase4_v3.py --variant A 2>&1 | tee train_A_v2.log`

```
设备: cpu

============================================================
  Model 1 Phase4 v3: Baseline VGG int8 + Gazelle 硬件噪声
  变体 A: 首层 conv1_1 FP32, 其余 Conv+Linear int8
============================================================
训练: 16200, 验证: 5400, 留出测试: 5400 (见 eurosat_split)
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 2,387,168

[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声, 首层 conv1_1 FP32)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 5 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 6
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [Baseline VGG (v4, 变体 A)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] conv1_1                      3   3×3        27        32   84.4%  w8
  [QATConv2d_v4 QAT  ] conv1_2                     32   3×3       288       288  100.0%  w8
  [QATConv2d_v4 QAT  ] conv2_1                     32   3×3       288       288  100.0%  w8
  [QATConv2d_v4 QAT  ] conv2_2                     64   3×3       576       576  100.0%  w8
  [QATConv2d_v4 QAT  ] conv3_1                     64   3×3       576       576  100.0%  w8
  [QATConv2d_v4 QAT  ] conv3_2                    128   3×3      1152      1152  100.0%  w8
  综合硬件对齐率: 100.0% (展平总长度 11355 → 补零后 11360)

[Step 2] 训练 (100 epochs, lr=0.001, wd=0.0005, label_smoothing=0.05)
  int8 权重 (硬件原生精度) | GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4)
  光计算层 (int8 QAT): conv1_2/conv2_1/conv2_2/conv3_1/conv3_2 + fc1/fc2
  电计算层 (FP32): conv1_1
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.5454   50.36% |    1.2450  65.20% |  65.20% | 0.00020 | 1054.6s
      5  |     0.9629   75.93% |    0.7960  80.81% |  80.81% | 0.00100 |  148.6s
     10  |     0.7288   86.25% |    0.6130  88.85% |  88.85% | 0.00099 |  144.7s
     15  |     0.6054   91.11% |    0.5316  91.89% |  92.93% | 0.00097 |  136.2s
     20  |     0.5576   92.75% |    0.5027  92.57% |  95.33% | 0.00094 |  143.0s
     25  |     0.5035   94.64% |    0.4214  95.59% |  95.59% | 0.00090 |  167.6s
     30  |     0.4737   95.78% |    0.4112  96.22% |  96.22% | 0.00084 |  182.8s
     35  |     0.4482   96.58% |    0.4080  96.04% |  96.31% | 0.00078 |  158.4s
     40  |     0.4425   96.74% |    0.4079  96.22% |  96.87% | 0.00070 |  164.6s
     45  |     0.4244   97.40% |    0.4044  96.13% |  96.94% | 0.00063 |  154.1s
     50  |     0.4095   98.01% |    0.3869  97.02% |  97.02% | 0.00055 |  142.1s
     55  |     0.3988   98.42% |    0.3800  97.13% |  97.13% | 0.00046 |  171.5s
     60  |     0.3909   98.62% |    0.3714  97.39% |  97.46% | 0.00038 |  142.7s
     65  |     0.3828   98.88% |    0.3705  97.31% |  97.63% | 0.00031 |  142.3s
     70  |     0.3767   99.06% |    0.3699  97.67% |  97.67% | 0.00023 |  167.7s
     75  |     0.3704   99.32% |    0.3596  97.74% |  97.78% | 0.00017 |  153.5s
     80  |     0.3665   99.54% |    0.3610  97.78% |  97.78% | 0.00011 |  171.3s
     85  |     0.3638   99.52% |    0.3592  97.61% |  97.80% | 0.00007 |  145.8s
     90  |     0.3636   99.51% |    0.3569  97.81% |  97.87% | 0.00004 |  153.8s
     95  |     0.3609   99.59% |    0.3569  97.85% |  97.87% | 0.00002 |  167.6s
    100  |     0.3603   99.65% |    0.3589  97.83% |  97.87% | 0.00001 |  114.0s

[Step 3] 最终评估 (val set)
[enable_qat] Enabled QAT on 8 layers
  Int8 模式 (光计算模拟) 准确率: 97.87%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:          97.93%
  Int8 量化损失:               +0.06%

  模型已保存: baseline_vgg_phase4_v3_int8.pth

============================================================
  训练完成 — 结果汇总 (变体 A)
============================================================
  模型:              Baseline VGG (flat+BN, bias=False)
  参数量:            2,387,168
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  电计算层 (FP32):   conv1_1
  训练总耗时:        16800.2s (280.0min)
  硬件对齐率:        100.0%
  Int8 最佳准确率:   97.87%
  Float32 准确率:    97.93%
  量化损失:          +0.06%
  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%
  推理脚本:          python optic_inference_int8_model1.py --variant A
```

## 训练 — 变体 B

`python -u model1_baseline_phase4_v3.py --variant B 2>&1 | tee train_B_v2.log`

```
设备: cpu

============================================================
  Model 1 Phase4 v3: Baseline VGG int8 + Gazelle 硬件噪声
  变体 B: 首层 conv1_1 FP32 + conv3_2 FP32, 其余 Conv+Linear int8
============================================================
训练: 16200, 验证: 5400, 留出测试: 5400 (见 eurosat_split)
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 2,387,168

[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声, 首层 conv1_1 FP32)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 5 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 6
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

[Step 1b] 变体 B: conv3_2 也保持 FP32 (电计算)
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128

  [Baseline VGG (v4, 变体 B)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] conv1_1                      3   3×3        27        32   84.4%  w8
  [QATConv2d_v4 QAT  ] conv1_2                     32   3×3       288       288  100.0%  w8
  [QATConv2d_v4 QAT  ] conv2_1                     32   3×3       288       288  100.0%  w8
  [QATConv2d_v4 QAT  ] conv2_2                     64   3×3       576       576  100.0%  w8
  [QATConv2d_v4 QAT  ] conv3_1                     64   3×3       576       576  100.0%  w8
  [QATConv2d_v4 FP32 ] conv3_2                    128   3×3      1152      1152  100.0%  w8
  综合硬件对齐率: 100.0% (展平总长度 11355 → 补零后 11360)

[Step 2] 训练 (100 epochs, lr=0.001, wd=0.0005, label_smoothing=0.05)
  int8 权重 (硬件原生精度) | GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4)
  光计算层 (int8 QAT): conv1_2/conv2_1/conv2_2/conv3_1 + fc1/fc2
  电计算层 (FP32): conv1_1 + conv3_2
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.5290   52.48% |    1.1943  65.31% |  65.31% | 0.00020 | 1038.3s
      5  |     0.9158   77.60% |    0.7175  83.56% |  83.56% | 0.00100 |  145.9s
     10  |     0.6687   88.10% |    0.5822  89.57% |  90.63% | 0.00099 |  143.3s
     15  |     0.5671   92.02% |    0.4747  93.61% |  93.72% | 0.00097 |  134.1s
     20  |     0.5225   93.48% |    0.4328  95.19% |  95.19% | 0.00094 |  140.2s
     25  |     0.4731   95.27% |    0.4225  95.65% |  95.65% | 0.00090 |  177.4s
     30  |     0.4594   95.68% |    0.4062  96.11% |  96.13% | 0.00084 |  178.8s
     35  |     0.4313   96.78% |    0.3971  96.37% |  96.37% | 0.00078 |  152.4s
     40  |     0.4165   97.36% |    0.4135  96.04% |  96.52% | 0.00070 |  167.5s
     45  |     0.4030   97.78% |    0.3881  96.78% |  97.13% | 0.00063 |  150.1s
     50  |     0.3915   98.34% |    0.3869  96.94% |  97.24% | 0.00055 |  170.6s
     55  |     0.3827   98.48% |    0.3674  97.70% |  97.70% | 0.00046 |  169.6s
     60  |     0.3714   98.88% |    0.3721  97.46% |  97.70% | 0.00038 |  146.6s
     65  |     0.3644   99.11% |    0.3620  97.70% |  97.70% | 0.00031 |  139.2s
     70  |     0.3589   99.27% |    0.3600  97.76% |  97.76% | 0.00023 |  169.7s
     75  |     0.3508   99.58% |    0.3639  97.52% |  97.76% | 0.00017 |  167.0s
     80  |     0.3490   99.61% |    0.3568  97.94% |  97.94% | 0.00011 |  164.8s
     85  |     0.3476   99.67% |    0.3593  97.76% |  97.94% | 0.00007 |  143.5s
     90  |     0.3457   99.75% |    0.3576  97.89% |  97.96% | 0.00004 |  147.4s
     95  |     0.3438   99.78% |    0.3588  97.81% |  97.96% | 0.00002 |  160.5s
    100  |     0.3434   99.75% |    0.3569  97.83% |  98.02% | 0.00001 |  172.1s

[Step 3] 最终评估 (val set)
[enable_qat] Enabled QAT on 8 layers
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128
  Int8 模式 (光计算模拟) 准确率: 98.02%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:          97.96%
  Int8 量化损失:               -0.06%

  模型已保存: baseline_vgg_phase4_v3_int8_vB.pth

============================================================
  训练完成 — 结果汇总 (变体 B)
============================================================
  模型:              Baseline VGG (flat+BN, bias=False)
  参数量:            2,387,168
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  电计算层 (FP32):   conv1_1 + conv3_2
  训练总耗时:        16566.7s (276.1min)
  硬件对齐率:        100.0%
  Int8 最佳准确率:   98.02%
  Float32 准确率:    97.96%
  量化损失:          -0.06%
  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%
  推理脚本:          python optic_inference_int8_model1.py --variant B
```

## QAT test 交叉验证 — 干净独立 test 集 (2026-07-14)

`python -u optic_inference_int8_model1.py --variant {A,B} --qat --batch 256 2>&1 | tee qat_{A,B}_v2.log`

独立 test 集 5400 张 (split=eurosat_split, **test∩train=0**); 量化损失 (test) = Float32(test) − Int8(test)。

| 变体 | Int8 QAT (val) | Int8 QAT (test) | Float32 (test) | 量化损失 (test) | test vs val (Int8) | 光计算占比 |
|---|---|---|---|---|---|---|
| A | 97.87% | **97.89%** | 97.91% | +0.02% | +0.02% | 97.74% (153.09M / 156.63M) |
| B | 98.02% | **97.96%** | 98.04% | +0.07% | −0.06% | 73.64% (115.35M / 156.63M) |

**test≈val** (Δ 在 ±0.06% 内) → Bug #11 修复后无泄漏, 泛化良好; 对比旧 leaky-split 版 test 虚高 99.96% (作废)。

### 变体 A

```
--- Loading Independent Test Set ---
Full: 27000 | Test(now): 5400 | split=eurosat_split (test∩train=0)

[Mode: QAT] PyTorch pseudo-quantization cross-validation...
  [1/3] Creating model...  Params: 2,387,168
  [2/3] Converting to QAT v4 (int8, 首层 conv1_1 FP32)...
  [3/3] Loading INT8 QAT weights: baseline_vgg_phase4_v3_int8.pth

  --- Native float32 (QAT disabled) ---
  [Model 1 Phase4 v3 INT8 (变体 A) fp32] 22 batches — acc=97.91%   (62.7s)
  --- int8 QAT (光计算模拟) ---
  [Model 1 Phase4 v3 INT8 (变体 A) int8] 22 batches — acc=97.89%   (21.3s)
  Quant Loss: +0.02%

  Model 1 INT8 (变体 A) — Container Verification Report
  QAT float32: 97.91%  |  QAT int8: 97.89%  |  Quant Loss: +0.02%

  [MOPs] 光计算占比 (变体 A): 97.74%  (光计算 153.0947M / 总 156.6336M, 电计算 conv1_1 3.5389M) [OK ≥50%]
```

### 变体 B

```
--- Loading Independent Test Set ---
Full: 27000 | Test(now): 5400 | split=eurosat_split (test∩train=0)

[Mode: QAT] PyTorch pseudo-quantization cross-validation...
  [1/3] Creating model...  Params: 2,387,168
  [2/3] Converting to QAT v4 (int8, 首层 conv1_1 FP32 + conv3_2 FP32)...
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128
  [3/3] Loading INT8 QAT weights: baseline_vgg_phase4_v3_int8_vB.pth

  --- Native float32 (QAT disabled) ---
  [Model 1 Phase4 v3 INT8 (变体 B) fp32] 22 batches — acc=98.04%   (51.5s)
  --- int8 QAT (光计算模拟) ---
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128
  [Model 1 Phase4 v3 INT8 (变体 B) int8] 22 batches — acc=97.96%   (20.9s)
  Quant Loss: +0.07%

  Model 1 INT8 (变体 B) — Container Verification Report
  QAT float32: 98.04%  |  QAT int8: 97.96%  |  Quant Loss: +0.07%

  [MOPs] 光计算占比 (变体 B): 73.64%  (光计算 115.3459M / 总 156.6336M,
        电计算 conv1_1 + conv3_2 41.2877M) [OK ≥50%]
```

## osimulator 真机抽样 (quick 50, 干净 test, 2026-07-15)

`python -u optic_inference_int8_model1.py --variant {A,B} --quick 50 2>&1 | tee osim_{A,B}.log`

干净 test 集前 50 张 (split=eurosat_split, test∩train=0); 默认 OPTIC 模式 = 真硬件 osimulator 8a8w 路径。

| 变体 | osim int8 (quick 50) | 正确/总数 | 耗时 | engine calls | 光计算占比 |
|---|---|---|---|---|---|
| A | **98.00%** | 49/50 | 11614s (~3.2h) | 350 | 97.74% |
| B | **100.00%** | 50/50 | 9853s (~2.7h) | 300 | 73.64% |

**结论**: 真硬件 8a8w 路径与 QAT/val 一致 (A osim 98.00% vs QAT test 97.89% / val 97.87%; B 50/50), **无损 ✓**。
- 变体 B engine calls 少 (300 vs 350): conv3_2 回退电计算, 光计算层 6 个 (A 是 7 个)。
- quick 50 样本小、波动大 (B 100% ≠ 总体 100%); 全量 5400 ~9 天不可行 (Model 1 MACs 156.6M, 是 Model 2 的 ~150x), 仅抽样。

### 变体 A — 报告

```
Optical Accuracy: 98.00%  Time: 11613.8s
[OpticalEngine 统计] 调用: 350, 总耗时: 11611.233s, 总运算量: 7.65e+09 MACs
Model 1 INT8 (变体 A) — Container Verification Report
Optic osimulator: 98.00%  |  Time: 11614s
光计算占比: 97.74% (光计算 153.0947M / 总 156.6336M, 电计算 conv1_1 3.5389M) [OK ≥50%]
```

### 变体 B — 报告

```
Optical Accuracy: 100.00%  Time: 9852.6s
[OpticalEngine 统计] 调用: 300, 总耗时: 9849.502s, 总运算量: 5.77e+09 MACs
Model 1 INT8 (变体 B) — Container Verification Report
Optic osimulator: 100.00%  |  Time: 9853s
光计算占比: 73.64% (光计算 115.3459M / 总 156.6336M, 电计算 conv1_1 + conv3_2 41.2877M) [OK ≥50%]
```
