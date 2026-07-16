Device: cuda
============================================================
  Optic-SpaceNet KD: In-Container Optical Inference
  Model 3 KD Phase4 v3 (int8+KD, TBD)  |  Weight: spacenet_v2_phase4_v3_int8.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | split=eurosat_split (test∩train=0)
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
  Model 3 KD (int8+KD)  [Optic mode: osimulator]
============================================================

  [1/3] Loading weights...
  Params: 267,944

  Original FP32 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [Conv2d]      stem.0                       3   1×1          3          8   37.5%
  [Conv2d]      stage1.0                     8   2×2         32         32   100.0%
  [Conv2d]      stage2.0                    16   2×2         64         64   100.0%
  [Conv2d]      stage3.0                    32   1×1         32         32   100.0%
  [Linear]      classifier.1               —     —           1024       1024   100.0%
  [Linear]      classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  [2/3] Converting to optical (int8 act + int8 weight, stem=electronic)...
  [Note] v3 int8 weights → osimulator native 8a8w (训练推理配置对齐, 应接近训练精度)

  Optical 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [Conv2d]      stem.0                       3   1×1          3          8   37.5%
  [OpticConv2d] stage1.0                     8   2×2         32         32   100.0%
  [OpticConv2d] stage2.0                    16   2×2         64         64   100.0%
  [OpticConv2d] stage3.0                    32   1×1         32         32   100.0%
  [OpticLinear] classifier.1               —     —           1024       1024   100.0%
  [OpticLinear] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  [3/3] Evaluating via osimulator...
  [optic] 5400 batches, report every 540 batch(es)
  [optic]  540/5400 ( 10.0%) acc=90.19%  elapsed=1392s  ETA=12526s
  [optic] 1080/5400 ( 20.0%) acc=89.07%  elapsed=2701s  ETA=10804s
  [optic] 1620/5400 ( 30.0%) acc=89.75%  elapsed=4011s  ETA=9358s
  [optic] 2160/5400 ( 40.0%) acc=89.77%  elapsed=5309s  ETA=7963s
  [optic] 2700/5400 ( 50.0%) acc=90.19%  elapsed=6722s  ETA=6722s
  [optic] 3240/5400 ( 60.0%) acc=90.22%  elapsed=8050s  ETA=5367s
  [optic] 3780/5400 ( 70.0%) acc=90.48%  elapsed=9389s  ETA=4024s
  [optic] 4320/5400 ( 80.0%) acc=90.35%  elapsed=10724s  ETA=2681s
  [optic] 4860/5400 ( 90.0%) acc=90.23%  elapsed=12041s  ETA=1338s
  [optic] 5400/5400 (100.0%) acc=90.28%  elapsed=13370s  ETA=0s
  [optic] DONE — 5400 batches, acc=90.28%, total=13370s
  Optical Accuracy: 90.28%  Time: 13370.2s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 13358.249s, 总运算量: 5.15e+09 MACs

====================================================================================================
  Model 3 KD — Container Verification Report
====================================================================================================
  Optic osimulator: 90.28%  |  Time: 13370s
  Training ref: v3 int8+KD (stem FP32, Gazelle noise, first_conv_fp32=True)

==============================================================================================================
  KD+INT4 模型光计算 MOPs 统计 — Model 3 SpaceNet V2 KD Phase4 v2
  Gazelle 硬件: 8x2 tile, act=int8, weight=int4, stem 电计算
==============================================================================================================
  stem.conv        Conv       3     8    1x1      64x64      64x64   None      3      8  37.5%    0.0983M    0.0000M    0.0983M [Electronic]
  stage1.conv      Conv       8    16    2x2      64x64      32x32 Max2x2     32     32 100.0%    0.5243M    0.5243M    0.0000M [Optical]   
  stage2.conv      Conv      16    32    2x2      16x16        8x8   None     64     64 100.0%    0.1311M    0.1311M    0.0000M [Optical]   
  stage3.conv      Conv      32    16    1x1        8x8        8x8   None     32     32 100.0%    0.0328M    0.0328M    0.0000M [Optical]   
  fc1              Linear  1024   256      -          -          -   None   1024   1024 100.0%    0.2621M    0.2621M    0.0000M [Optical]   
  fc2              Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0026M    0.0000M [Optical]   
  [MOPs] 光计算占比: 90.65%  |  总 MOPs: 1.0511 M
  [Note] KD 训练用 ResNet-18 (97.83%) 做教师, 推理时不需要教师模型
==============================================================================================================
====================================================================================================
