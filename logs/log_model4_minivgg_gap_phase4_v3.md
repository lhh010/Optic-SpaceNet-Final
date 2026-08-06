设备: cpu | torch 2.13.0+cpu | threads 16
==================================================================
  Model 4 MiniVGG-GAP Phase4 v3: int8 QAT + Gazelle 噪声 (本地 CPU)
  首层 stem FP32 (对齐率低), 其余 Conv+Linear int8
  策略: 从 FP32 基线 minivgg_gap.pth 微调 5 epochs
==================================================================
训练: 16200, 验证: 5400 (eurosat_split, test 留出)

参数量: 260,234
[load] FP32 基线 weights/minivgg_gap.pth: missing=0 unexpected=0

[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声, stem FP32)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 6 enabled + 1 fp32 (first layer)
  QAT Linear: 1, BN: 7
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [MiniVGG-GAP (v4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   3×3        27        32   84.4%  w8
  [QATConv2d_v4 QAT  ] stage1.0                    32   3×3       288       288  100.0%  w8
  [QATConv2d_v4 QAT  ] stage1.3                    48   3×3       432       432  100.0%  w8
  [QATConv2d_v4 QAT  ] stage2.0                    48   3×3       432       432  100.0%  w8
  [QATConv2d_v4 QAT  ] stage2.3                    72   3×3       648       648  100.0%  w8
  [QATConv2d_v4 QAT  ] stage3.0                    72   3×3       648       648  100.0%  w8
  [QATConv2d_v4 QAT  ] stage3.3                    96   3×3       864       864  100.0%  w8
  综合硬件对齐率: 99.9% (展平总长度 3435 → 补零后 3440)

[Step 2] 微调 (5 epochs, lr=0.001, wd=0.0005, label_smoothing=0.05)
  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声 (训练时)
------------------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
----------------------------------------------------------------------------
      1  |     0.4719   97.13% |    0.4590  94.96% |  94.96% | 0.00020 |  111.8s
      2  |     0.3727   98.74% |    0.4472  95.22% |  95.22% | 0.00040 |  111.7s
      3  |     0.3732   98.58% |    0.4301  95.59% |  95.59% | 0.00060 |  110.7s
      4  |     0.3816   97.93% |    0.4443  95.35% |  95.59% | 0.00080 |  109.3s
      5  |     0.3946   97.23% |    0.7585  82.26% |  95.59% | 0.00100 |  109.1s

[Step 3] 最终评估 (最佳 val 95.59%)
[enable_qat] Enabled QAT on 8 layers
  Int8 模式 (光计算模拟) val 准确率: 95.59%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式 val 准确率:        95.43%
  Int8 量化损失:                 -0.17%
[enable_qat] Enabled QAT on 8 layers
  Int8 模式 test 准确率 (独立 5400 张): 95.50%

  模型已保存: weights/minivgg_gap_phase4_v3_int8.pth

==================================================================
  训练完成 — 结果汇总
==================================================================
  模型:           MiniVGG-GAP (Phase4 v3, int8 + Gazelle 噪声)
  参数量:         260,234
  首层:           FP32 (stem 3→32, 对齐率低)
  其余:           Conv×7 + Linear×1 → int8 QAT (含 GAP 前输入)
  硬件对齐率:     99.9%
  训练总耗时:     552.6s (9.2min, 5 epochs)
  Int8 最佳 val:  95.59%
  Int8 最终 val:  95.59%
  Float32 val:    95.43%
  FP32 基线参考:  96.65% (minivgg_gap.pth, 原生 FP32)
  权重:           weights/minivgg_gap_phase4_v3_int8.pth
==================================================================
