# Model 3 (SpaceNet V2 +KD) Phase4 v3 int8 重训日志 (2026-07-13)

> Bug #11 split 修复后重训 (eurosat_split): train 16200 / val 5400 / 留出 test 5400
> 结果: Int8 val 91.83% / Float32 91.65% / 量化损失 -0.19% (vs FP32 KD 基准 91.44%, +0.39%)

```
PS E:\LT-Simulator\train-test>  python model3_spacenet_v2_phase4_v3.py
设备: cpu

============================================================
  Model 3 Phase4 v3: KD + int8 + Gazelle 噪声
  stem FP32 (匹配 osimulator) + Conv/Linear int8 QAT
============================================================
训练: 16200, 验证: 5400, 留出测试: 5400 (见 eurosat_split)
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 0] 加载教师 (ResNet-18)
  教师权重加载成功 (97.83%)
  学生参数量: 267,944
  架构: 4×Conv + 2×Linear, bias=False, BN 保留

[Step 1] 转换学生: stem FP32, 其余 Conv+Linear→int8 QAT
  噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 3 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 4
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)
  int8 QAT Conv: 3, int8 QAT Linear: 2, BN: 4 (float32), stem: FP32

  [Student (stem FP32 + int8)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   1×1         3         8   37.5%  w8
  [QATConv2d_v4 QAT  ] stage1.0                     8   2×2        32        32  100.0%  w8
  [QATConv2d_v4 QAT  ] stage2.0                    16   2×2        64        64  100.0%  w8
  [QATConv2d_v4 QAT  ] stage3.0                    32   1×1        32        32  100.0%  w8
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] KD+Phase4 v3 训练 (100 epochs, T=4.0, α=0.7)
  教师: ResNet-18 (fp32, 97.83%)
  学生: stem FP32 + Conv/Linear int8 + Gazelle 噪声
  int8 权重 (硬件原生精度, 匹配 osimulator)
---------------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  -------------------------------------------------------------------------
      1  |    10.2355   58.89% |    1.3540  72.30% |  72.30% | 0.00020 |   49.0s
      5  |     6.5715   75.07% |    1.2812  78.54% |  78.54% | 0.00100 |   45.4s
     10  |     5.7345   79.09% |    1.6170  73.20% |  80.61% | 0.00099 |   58.5s
     15  |     5.0921   82.35% |    1.1057  84.89% |  85.06% | 0.00097 |   84.5s
     20  |     4.8141   83.64% |    1.0951  84.81% |  86.46% | 0.00094 |   53.3s
     25  |     4.5001   85.03% |    1.1210  85.37% |  86.98% | 0.00090 |   54.4s
     30  |     4.2496   86.25% |    1.0028  88.69% |  88.69% | 0.00084 |   52.0s
     35  |     4.1040   87.22% |    0.9365  89.63% |  89.63% | 0.00078 |   57.2s
     40  |     3.9164   87.69% |    1.0005  88.39% |  89.63% | 0.00070 |   53.9s
     45  |     3.7901   88.46% |    0.9740  89.57% |  90.09% | 0.00063 |   54.9s
     50  |     3.6839   89.13% |    0.9735  89.26% |  90.19% | 0.00055 |   53.6s
     55  |     3.5490   89.46% |    0.9406  90.50% |  90.50% | 0.00046 |   60.6s
     60  |     3.4368   90.24% |    0.9352  90.94% |  90.94% | 0.00038 |   56.7s
     65  |     3.3646   90.35% |    0.9043  91.31% |  91.31% | 0.00031 |   57.0s
     70  |     3.3262   90.44% |    0.8859  91.30% |  91.31% | 0.00023 |   51.6s
     75  |     3.2521   90.74% |    0.9151  91.24% |  91.31% | 0.00017 |   57.1s
     80  |     3.2113   90.90% |    0.8931  91.28% |  91.31% | 0.00011 |   47.3s
     85  |     3.2086   90.69% |    0.8934  91.44% |  91.54% | 0.00007 |   46.8s
     90  |     3.1497   91.30% |    0.8855  91.39% |  91.54% | 0.00004 |   46.7s
     95  |     3.1330   91.20% |    0.8841  91.59% |  91.83% | 0.00002 |   46.3s
    100  |     3.1629   91.23% |    0.8925  91.61% |  91.83% | 0.00001 |   46.2s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int8 模式 (光计算模拟) 准确率: 91.83%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              91.65%
  Int8 量化损失:             -0.19%

  模型已保存: spacenet_v2_phase4_v3_int8.pth

============================================================
  训练完成 — 结果汇总
============================================================
  学生模型:          OpticSpaceNet (bias=False)
  教师模型:          ResNet-18 (fp32, 97.83%)
  参数量:            267,944
  权重量化:          int8 (匹配 osimulator 原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  首层:              FP32 (对齐率 37.5%, 匹配 osimulator)
  蒸馏:              T=4.0, α=0.7
  训练总耗时:        5426.2s (90.4min)
  硬件对齐率:        99.6%
  Int8 最佳准确率: 91.83%
  Float32 准确率:    91.65%
  FP32 KD 基准:      91.44% (全 fp32 KD)
  osimulator 预期:   ~91.8%% (训练推理配置对齐, 应接近训练精度)
```

## QAT test 交叉验证 — 干净独立 test 集 (2026-07-14)

`python -u optic_inference_kd.py --qat --batch 256 2>&1 | tee qat_model3_v2.log`

独立 test 集 5400 张 (split=eurosat_split, **test∩train=0**)。

> ⚠️ **此 `--qat` 路径是 int4 (optic_qat_v3), 不是 int8!** 84.59% 是把 int8 训练权重再降到 int4 的退化
> (与 §16 int4 困境一致), **非 Model 3 的 int8 test 数**。int8 test 数须走 osimulator (OPTIC 模式)。

| 指标 | 值 | 说明 |
|---|---|---|
| Int4 QAT (test) | 84.59% | int4 退化 (optic_qat_v3, w4/a8), 非 int8 |
| Float32 (test) | 92.13% | QAT 关闭, ≈ val fp32 91.65% → fp32 无泄漏 |
| 量化损失 | +7.54% | int4 vs fp32 |
| Int8 (val, 干净重训) | 91.83% | int8 训练精度 |
| 光计算占比 | 90.65% | — |

**结论**: float32 test 92.13% ≈ val fp32 91.65% → Bug #11 修复后 fp32 层无泄漏 ✓。
int4 84.59% 符合 §16 预期 (int8 权重→int4 网格不对齐)。
**int8 test 数待 osimulator** (`optic_inference_kd.py` 默认 OPTIC 模式, "v3 int8 → 8a8w, 应接近训练精度 91.83%")。
