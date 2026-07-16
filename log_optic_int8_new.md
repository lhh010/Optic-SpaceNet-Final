Device: cuda
============================================================
  Optic-SpaceNet INT8: In-Container Optical Inference
  Model:  SpaceNet V1 Phase4 v3 (INT8, Gazelle-optimized)
  Weight: spacenet_v1_phase4_v3_int8.pth
  Mode:   Optic (osimulator, in-container)
  Batch:  1, full test set
============================================================

--- Loading Independent Test Set ---
Full dataset: 27000 imgs
Test (now): 5400 imgs | split=eurosat_split (test∩train=0)
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
  [Model 2 Phase4 v3 INT8 optic]  540/5400 ( 10.0%) acc=89.07%  elapsed=1365s  ETA=12289s
  [Model 2 Phase4 v3 INT8 optic] 1080/5400 ( 20.0%) acc=89.81%  elapsed=2680s  ETA=10722s
  [Model 2 Phase4 v3 INT8 optic] 1620/5400 ( 30.0%) acc=90.25%  elapsed=4001s  ETA=9336s
  [Model 2 Phase4 v3 INT8 optic] 2160/5400 ( 40.0%) acc=90.14%  elapsed=5282s  ETA=7924s
  [Model 2 Phase4 v3 INT8 optic] 2700/5400 ( 50.0%) acc=90.30%  elapsed=6705s  ETA=6705s
  [Model 2 Phase4 v3 INT8 optic] 3240/5400 ( 60.0%) acc=90.43%  elapsed=8044s  ETA=5363s
  [Model 2 Phase4 v3 INT8 optic] 3780/5400 ( 70.0%) acc=90.69%  elapsed=9373s  ETA=4017s
  [Model 2 Phase4 v3 INT8 optic] 4320/5400 ( 80.0%) acc=90.56%  elapsed=10691s  ETA=2673s
  [Model 2 Phase4 v3 INT8 optic] 4860/5400 ( 90.0%) acc=90.31%  elapsed=12016s  ETA=1335s
  [Model 2 Phase4 v3 INT8 optic] 5400/5400 (100.0%) acc=90.43%  elapsed=13357s  ETA=0s
  [Model 2 Phase4 v3 INT8 optic] DONE — 5400 batches, acc=90.43%, total=13357s
  Optical Accuracy: 90.43%
  Optical Time:     13357.49s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 13345.952s, 总运算量: 5.15e+09 MACs



==============================================================================================================
  OPTIC-SPACENET INT8: Optical Computing Inference & MOPs Report
  Model 2 SpaceNet V1 Phase4 v3 — 当前最佳 INT8 模型
==============================================================================================================

  ------------------------------------------------------------
  [Accuracy] Optic osimulator Hardware Simulation (独立测试集)
  ------------------------------------------------------------
  模型:               Model 2 Phase4 v3 INT8
  光计算准确率:       90.43%
  osimulator 耗时:    13357.5s


==============================================================================================================
  INT8 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v3
  Gazelle 硬件: 8×2 tile, 8a8w12o, 首层 stem FP32 (电计算)
==============================================================================================================

  Layer            Type    C_in C_out Kernel      Input    ConvOut   Pool  Patch Padded   Align    RawMOPs    OptMOPs   ElecMOPs      Compute
  ------------------------------------------------------------------------------------------------------------------------
  stem.conv        Conv       3     8    1x1      64x64      64x64   None      3      8  37.5%    0.0983M    0.0000M    0.0983M [Electronic]
  stage1.conv      Conv       8    16    2x2      64x64      32x32 Max2x2     32     32 100.0%    0.5243M    0.5243M    0.0000M [Optical]   
  stage2.conv      Conv      16    32    2x2      16x16        8x8   None     64     64 100.0%    0.1311M    0.1311M    0.0000M [Optical]   
  stage3.conv      Conv      32    16    1x1        8x8        8x8   None     32     32 100.0%    0.0328M    0.0328M    0.0000M [Optical]   
  fc1              Linear  1024   256      -          -          -   None   1024   1024 100.0%    0.2621M    0.2621M    0.0000M [Optical]   
  fc2              Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0026M    0.0000M [Optical]   
  ------------------------------------------------------------------------------------------------------------------------
  Total                                                                                            1.0511M    0.9528M    0.0983M

  ------------------------------------------------------------
  [MOPs] 光计算占比汇总
  ------------------------------------------------------------
  总原始 MOPs:           1.0511 M
  光计算 MOPs (有效):    0.9528 M
  电子计算 MOPs:         0.0983 M
  总有效 MOPs:           1.0511 M
  -------------------------------------
  ** 光计算占比:         90.65%  (**)
  光计算补零浪费:        0 (所有光计算层完美对齐 8 的倍数) [OK]
  ------------------------------------------------------------

  [Note] 说明:
    - stem.conv (3->8, 1x1): 展平长度=3, 对齐率仅 37.5%, 保留电计算 (FP32)
    - 其余 Conv/Linear 展平长度均为 8 的倍数, 完美对齐 Gazelle 8×2 tile
    - 光计算占比 = 光计算有效MOPs / (光计算MOPs + 电计算MOPs)
    - 光计算有效MOPs 已包含补零对齐的硬件开销
==============================================================================================================

  ------------------------------------------------------------
  [Verdict] 部署评估结论
  ------------------------------------------------------------
  光计算占比:         90.65%
  判定:               [OK] 高度适合光计算部署
  硬件对齐率:         99.6% (除 stem 外所有层完美对齐 8×2 tile)
  首层 stem:          FP32 电计算 (展平=3, 对齐率 37.5%, 电计算更高效)
  推荐部署策略:       stem 在 CPU/GPU, 其余 5 层在 Gazelle 光计算
==============================================================================================================
