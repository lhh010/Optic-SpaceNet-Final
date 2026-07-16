```powershell
PS E:\LT-Simulator> docker start LT-Simulator-container
LT-Simulator-container
PS E:\LT-Simulator> docker exec -it -w /workspace LT-Simulator-container /bin/bash
(moca_llm) root@a39a38d1a33b:/workspace# cd share/
(moca_llm) root@a39a38d1a33b:/workspace/share# ls
LT-Simulator_docker_v1.4.6-CCIC.tar  docs  scratch  train-firstround  train-test
(moca_llm) root@a39a38d1a33b:/workspace/share# cd train-test
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python model1_baseline_qat.py
设备: cpu
数据目录: data/EuroSAT_RGB
============================================================
  模型一 QAT (Baseline VGG): int4 QAT 微调
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Phase 1] 加载预训练 float32 权重: baseline_vgg.pth
  参数量: 2,386,986
  权重加载成功!

[Phase 1b] 评估 float32 基准准确率 (eval mode)...
  Float32 验证准确率: 97.07%
  Float32 验证损失:   0.1629

[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...
[prepare_qat_model] Model converted to QAT-ready (fuse_bn=True)
  QAT 层: {'QATConv2d': 6, 'QATLinear': 2}

  [BaselineVGG (QAT)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] block1.0                     3   3×3         27         32   84.4%
  [QATConv2d  ] block1.2                    32   3×3        288        288   100.0%
  [QATConv2d  ] block2.0                    32   3×3        288        288   100.0%
  [QATConv2d  ] block2.2                    64   3×3        576        576   100.0%
  [QATConv2d  ] block3.0                    64   3×3        576        576   100.0%
  [QATConv2d  ] block3.2                   128   3×3       1152       1152   100.0%
  [QATLinear  ] classifier.1               —     —           8192       8192   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 100.0% (总展平 11355 → 补零后 11360)

[Phase 2b] 校准 QAT 模型...
[calibrate] Running calibration on 3 batches...
  Batch 1/3 — input range: [-2.118, 2.640]
  Batch 2/3 — input range: [-2.118, 2.640]
  Batch 3/3 — input range: [-2.118, 2.640]
[calibrate] Calibration complete.

[Phase 3] QAT 微调 (15 epochs, lr=0.0001)
  说明: 使用低学习率 (1/10 of FP32 training)
        伪 int4 量化在每层前向时自动施加
        STE 梯度让模型学会对量化噪声具有鲁棒性
----------------------------------------------------------------------
   Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.3078   76.74% |    0.7437  85.17% |  85.17% |  855.3s
      2  |     0.6759   82.39% |    1.1293  81.37% |  85.17% |  589.6s
      3  |     0.6018   83.67% |    1.1647  80.93% |  85.17% |  421.9s
      4  |     0.5534   85.14% |    0.9487  81.70% |  85.17% |  421.8s
      5  |     0.5592   85.09% |    0.6326  86.65% |  86.65% |  414.5s
      6  |     0.5344   85.05% |    0.4774  89.72% |  89.72% |  425.5s
      7  |     0.4845   86.11% |    0.4398  90.87% |  90.87% |  425.3s
      8  |     0.4783   86.31% |    0.4448  90.83% |  90.87% |  468.0s
      9  |     0.4782   86.45% |    0.4293  91.35% |  91.35% |  464.5s
     10  |     0.4564   87.05% |    0.4329  91.61% |  91.61% |  417.1s
     11  |     0.4682   87.53% |    0.4687  90.93% |  91.61% |  417.3s
     12  |     0.4526   87.66% |    0.4727  90.19% |  91.61% |  418.6s
     13  |     0.4381   87.92% |    0.4632  90.44% |  91.61% |  415.6s
     14  |     0.4534   87.56% |    0.4670  90.22% |  91.61% |  420.3s
     15  |     0.4335   87.59% |    0.4696  90.17% |  91.61% |  422.7s

[Phase 4] QAT 训练完成 — 评估与保存

============================================================
  QAT vs Float32 Comparison
============================================================
[enable_qat] Enabled QAT on 8 layers
  QAT mode (int4):     Accuracy = 91.61%
[disable_qat] Disabled QAT on 8 layers
  Float mode (float32): Accuracy = 91.61%
  Accuracy gap:         0.00% (✓ QAT successful)

  保存 QAT-trained 权重至: baseline_vgg_qat.pth
  文件大小: 9330.0 KB

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:          Mini-VGG (全 3×3 卷积)
  参数量:            2,386,986
  Float32 基准准确率: 97.07%
  QAT 最佳准确率:     91.61%
  QAT vs Float gap:  0.00%
  QAT 训练耗时:      6998.0 秒 (116.6 分钟)
  8×2 硬件对齐率:    100.0%

  预期光计算推理:
    原生 float32:     97.1%
    PTQ (直接 int4):  ~87.1% (大幅下降)
    QAT (微调后 int4): 91.6% (接近 float32)
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python model2_spacenet_v1_qat.py
设备: cpu
============================================================
  模型二 QAT (Optic-SpaceNet V1): int4 QAT 微调
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Phase 1] 加载预训练 float32 权重: spacenet_v1.pth
  参数量: 268,210
  权重加载成功!

[Phase 1b] 评估 float32 基准准确率...
  Float32 验证准确率: 89.98%
  Float32 验证损失:   0.2955

[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...
[prepare_qat_model] Model converted to QAT-ready (fuse_bn=True)
  QAT 层: {'QATConv2d': 4, 'QATLinear': 2}

  [OpticSpaceNetV1 (QAT)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] stem.0                       3   1×1          3          8   37.5%
  [QATConv2d  ] stage1.0                     8   2×2         32         32   100.0%
  [QATConv2d  ] stage2.0                    16   2×2         64         64   100.0%
  [QATConv2d  ] stage3.0                    32   1×1         32         32   100.0%
  [QATLinear  ] classifier.1               —     —           1024       1024   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  ⚠ 注意: stem 层 patch_len=3, padded=8, 对齐率仅 37.5%
    这是唯一未对齐的层，但 ops 极少 (3×1×1=3)

[Phase 2b] 校准 QAT 模型...
[calibrate] Running calibration on 3 batches...
  Batch 1/3 — input range: [-2.118, 2.640]
  Batch 2/3 — input range: [-2.118, 2.640]
  Batch 3/3 — input range: [-2.118, 2.640]
[calibrate] Calibration complete.

[Phase 3] QAT 微调 (20 epochs, lr=0.0001)
----------------------------------------------------------------------
   Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.9802   55.37% |    1.3855  61.72% |  61.72% |  220.9s
      2  |     1.2684   62.18% |    1.4557  60.80% |  61.72% |  211.1s
      3  |     1.1640   63.95% |    1.5892  59.85% |  61.72% |  215.7s
      4  |     1.1490   64.78% |    1.3241  63.24% |  63.24% |  217.0s
      5  |     1.1128   64.88% |    1.4693  59.54% |  63.24% |  219.8s
      6  |     1.0344   66.16% |    1.4789  59.33% |  63.24% |  220.7s
      7  |     1.0047   67.03% |    1.5876  58.57% |  63.24% |  212.9s
      8  |     0.9873   67.47% |    1.5828  60.22% |  63.24% |  257.9s
      9  |     0.9789   67.81% |    1.5182  60.33% |  63.24% |  259.1s
     10  |     0.9703   68.21% |    1.4797  60.93% |  63.24% |  278.3s
     11  |     0.9916   67.79% |    1.5929  60.26% |  63.24% |  241.7s
     12  |     0.9524   68.62% |    1.6528  60.00% |  63.24% |  260.2s
     13  |     0.9613   68.59% |    1.6722  59.52% |  63.24% |  262.5s
     14  |     0.9616   68.85% |    1.7647  58.70% |  63.24% |  264.3s
     15  |     0.9686   69.12% |    1.7116  59.46% |  63.24% |  263.2s
     16  |     0.9445   69.17% |    1.7209  59.39% |  63.24% |  263.9s
     17  |     0.9409   69.17% |    1.7231  59.59% |  63.24% |  260.5s
     18  |     0.9458   68.77% |    1.7206  59.56% |  63.24% |  277.2s
     19  |     0.9388   69.43% |    1.7164  59.63% |  63.24% |  248.6s
     20  |     0.9445   69.94% |    1.7158  59.59% |  63.24% |  293.6s

[Phase 4] QAT 训练完成 — 评估与保存

============================================================
  QAT vs Float32 Comparison
============================================================
[enable_qat] Enabled QAT on 6 layers
  QAT mode (int4):     Accuracy = 63.24%
[disable_qat] Disabled QAT on 6 layers
  Float mode (float32): Accuracy = 63.24%
  Accuracy gap:         0.00% (✓ QAT successful)

  保存 QAT-trained 权重至: spacenet_v1_qat.pth
  文件大小: 1052.0 KB

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:          Optic-SpaceNet V1 (硬件对齐)
  参数量:            268,210
  Float32 基准准确率: 89.98%
  QAT 最佳准确率:     63.24%
  QAT vs Float gap:  0.00%
  QAT 训练耗时:      4949.2 秒 (82.5 分钟)
  8×2 硬件对齐率:    99.6%

  预期光计算推理:
    原生 float32:     90.0%
    PTQ (直接 int4):  ~78.0% (大幅下降)
    QAT (微调后 int4): 63.2% 

```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python model3_spacenet_v2_qat.py --use_kd
设备: cpu
============================================================
  模型三 QAT (Optic-SpaceNet V2): Mode B: KD + QAT 联合微调
============================================================
Traceback (most recent call last):
  File "/workspace/share/train-test/model3_spacenet_v2_qat.py", line 442, in <module>
    main()
  File "/workspace/share/train-test/model3_spacenet_v2_qat.py", line 283, in main
    train_loader, val_loader = load_data()
  File "/workspace/share/train-test/model3_spacenet_v2_qat.py", line 178, in load_data
    rng = np.random.RandomState(SEED)
NameError: name 'np' is not defined
```



---

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python model1_baseline_qat.py
设备: cpu
数据目录: data/EuroSAT_RGB
============================================================
  模型一 QAT (Baseline VGG): int4 QAT 微调
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Phase 1] 加载预训练 float32 权重: baseline_vgg.pth
  参数量: 2,386,986
  权重加载成功!

[Phase 1b] 评估 float32 基准准确率 (eval mode)...
  Float32 验证准确率: 97.07%
  Float32 验证损失:   0.1629

[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...
[prepare_qat_model] Model converted to QAT-ready (fuse_bn=True)
  QAT 层: {'QATConv2d': 6, 'QATLinear': 2}

  [BaselineVGG (QAT)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] block1.0                     3   3×3         27         32   84.4%
  [QATConv2d  ] block1.2                    32   3×3        288        288   100.0%
  [QATConv2d  ] block2.0                    32   3×3        288        288   100.0%
  [QATConv2d  ] block2.2                    64   3×3        576        576   100.0%
  [QATConv2d  ] block3.0                    64   3×3        576        576   100.0%
  [QATConv2d  ] block3.2                   128   3×3       1152       1152   100.0%
  [QATLinear  ] classifier.1               —     —           8192       8192   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 100.0% (总展平 11355 → 补零后 11360)

[Phase 2b] 校准 QAT 模型...
[calibrate] Running calibration on 3 batches...
  Batch 1/3 — input range: [-2.118, 2.640]
  Batch 2/3 — input range: [-2.118, 2.640]
  Batch 3/3 — input range: [-2.118, 2.640]
[calibrate] Calibration complete.

[Phase 3] QAT 微调 (15 epochs, lr=0.0001)
  说明: 使用低学习率 (1/10 of FP32 training)
        伪 int4 量化在每层前向时自动施加
        STE 梯度让模型学会对量化噪声具有鲁棒性
----------------------------------------------------------------------
   Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.2969   75.85% |    1.0835  80.87% |  80.87% |  415.6s
      2  |     0.6486   82.40% |    1.0868  81.20% |  81.20% |  439.9s
      3  |     0.6334   83.40% |    0.9506  81.63% |  81.63% |  431.3s
      4  |     0.5532   84.75% |    0.7793  84.37% |  84.37% |  442.2s
      5  |     0.5443   85.23% |    0.6996  85.91% |  85.91% |  444.1s
      6  |     0.5307   85.85% |    0.8495  83.00% |  85.91% |  443.9s
      7  |     0.5113   86.07% |    0.7879  84.78% |  85.91% |  440.7s
      8  |     0.5054   86.47% |    0.7692  84.74% |  85.91% |  425.7s
      9  |     0.4880   86.50% |    0.7526  84.67% |  85.91% |  470.8s
     10  |     0.4845   86.47% |    0.7026  85.19% |  85.91% |  479.8s
     11  |     0.5127   85.93% |    0.6865  83.98% |  85.91% |  482.0s
     12  |     0.4862   86.33% |    0.6458  85.61% |  85.91% |  485.3s
     13  |     0.4852   86.51% |    0.6796  84.78% |  85.91% |  493.7s
     14  |     0.4846   86.27% |    0.6810  85.54% |  85.91% |  495.9s
     15  |     0.4645   86.79% |    0.6536  85.85% |  85.91% |  538.6s

[Phase 4] QAT 训练完成 — 评估与保存

============================================================
  QAT vs Float32 Comparison
============================================================
[enable_qat] Enabled QAT on 8 layers
  QAT mode (int4):     Accuracy = 85.91%
[disable_qat] Disabled QAT on 8 layers
  Float mode (float32): Accuracy = 88.85%
  Accuracy gap:         2.94% (⚠ Needs more QAT training)

  保存 QAT-trained 权重至: baseline_vgg_qat.pth
  文件大小: 9330.0 KB

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:          Mini-VGG (全 3×3 卷积)
  参数量:            2,386,986
  Float32 基准准确率: 97.07%
  QAT 最佳准确率:     85.91%
  QAT vs Float gap:  2.94%
  QAT 训练耗时:      6929.6 秒 (115.5 分钟)
  8×2 硬件对齐率:    100.0%

  预期光计算推理:
    原生 float32:     97.1%
    PTQ (直接 int4):  ~87.1% (大幅下降)
    QAT (微调后 int4): 85.9% 
```


```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python model2_spacenet_v1_qat.py
设备: cpu
============================================================
  模型二 QAT (Optic-SpaceNet V1): int4 QAT 微调
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Phase 1] 加载预训练 float32 权重: spacenet_v1.pth
  参数量: 268,210
  权重加载成功!

[Phase 1b] 评估 float32 基准准确率...
  Float32 验证准确率: 89.98%
  Float32 验证损失:   0.2955

[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...
[prepare_qat_model] Model converted to QAT-ready (fuse_bn=True)
  QAT 层: {'QATConv2d': 4, 'QATLinear': 2}

  [OpticSpaceNetV1 (QAT)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] stem.0                       3   1×1          3          8   37.5%
  [QATConv2d  ] stage1.0                     8   2×2         32         32   100.0%
  [QATConv2d  ] stage2.0                    16   2×2         64         64   100.0%
  [QATConv2d  ] stage3.0                    32   1×1         32         32   100.0%
  [QATLinear  ] classifier.1               —     —           1024       1024   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  ⚠ 注意: stem 层 patch_len=3, padded=8, 对齐率仅 37.5%
    这是唯一未对齐的层，但 ops 极少 (3×1×1=3)

[Phase 2b] 校准 QAT 模型...
[calibrate] Running calibration on 3 batches...
  Batch 1/3 — input range: [-2.118, 2.640]
  Batch 2/3 — input range: [-2.118, 2.640]
  Batch 3/3 — input range: [-2.118, 2.640]
[calibrate] Calibration complete.

[Phase 3] QAT 微调 (20 epochs, lr=0.0001)
----------------------------------------------------------------------
   Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.9924   55.22% |    1.2091  65.65% |  65.65% |  306.1s
      2  |     1.2406   62.46% |    1.1694  68.54% |  68.54% |  324.3s
      3  |     1.1712   63.31% |    1.2168  67.37% |  68.54% |  313.9s
      4  |     1.1102   65.54% |    0.9740  70.11% |  70.11% |  262.2s
      5  |     1.1288   65.35% |    1.0042  68.67% |  70.11% |  250.9s
      6  |     1.0631   66.02% |    1.2410  66.78% |  70.11% |  238.9s
      7  |     1.0569   66.42% |    0.8954  71.54% |  71.54% |  257.8s
      8  |     1.0008   67.75% |    1.2317  66.11% |  71.54% |  227.8s
      9  |     0.9859   67.58% |    1.2110  67.15% |  71.54% |  236.8s
     10  |     0.9692   68.26% |    0.9092  70.54% |  71.54% |  237.7s
     11  |     0.9505   68.75% |    0.8713  72.19% |  72.19% |  243.0s
     12  |     0.9394   69.49% |    0.8295  73.28% |  73.28% |  258.4s
     13  |     0.9487   69.29% |    0.8475  72.63% |  73.28% |  263.4s
     14  |     0.9218   70.10% |    0.8650  72.52% |  73.28% |  267.4s
     15  |     0.9136   69.71% |    0.8322  73.44% |  73.44% |  280.6s
     16  |     0.8996   70.38% |    0.8707  71.76% |  73.44% |  255.4s
     17  |     0.9075   70.08% |    0.8093  73.59% |  73.59% |  222.0s
     18  |     0.9077   69.95% |    0.8576  72.50% |  73.59% |  260.0s
     19  |     0.9269   70.06% |    0.8286  73.63% |  73.63% |  228.2s
     20  |     0.8896   70.82% |    0.8691  73.37% |  73.63% |  212.3s

[Phase 4] QAT 训练完成 — 评估与保存

============================================================
  QAT vs Float32 Comparison
============================================================
[enable_qat] Enabled QAT on 6 layers
  QAT mode (int4):     Accuracy = 73.63%
[disable_qat] Disabled QAT on 6 layers
  Float mode (float32): Accuracy = 57.22%
  Accuracy gap:         -16.41% (✓ QAT successful)

  保存 QAT-trained 权重至: spacenet_v1_qat.pth
  文件大小: 1052.0 KB

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:          Optic-SpaceNet V1 (硬件对齐)
  参数量:            268,210
  Float32 基准准确率: 89.98%
  QAT 最佳准确率:     73.63%
  QAT vs Float gap:  -16.41%
  QAT 训练耗时:      5146.9 秒 (85.8 分钟)
  8×2 硬件对齐率:    99.6%

  预期光计算推理:
    原生 float32:     90.0%
    PTQ (直接 int4):  ~78.0% (大幅下降)
    QAT (微调后 int4): 73.6% (接近 float32)
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python model3_spacenet_v2_qat.py --use_kd
设备: cpu
============================================================
  模型三 QAT (Optic-SpaceNet V2): Mode B: KD + QAT 联合微调
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Phase 1] 加载预训练权重
  学生权重: spacenet_v2_distilled.pth
  学生参数量: 268,210
  学生权重加载成功!
  教师权重: teacher_resnet18.pth
  教师参数量: 11,181,642
  教师权重加载成功!

[Phase 1b] 评估 float32 基准准确率...
  Float32 验证准确率: 91.44%
  Float32 验证损失:   0.3328

[Phase 2] 准备 QAT 模型 (Conv+BN 融合 + QAT 层替换)...
[prepare_qat_model] Model converted to QAT-ready (fuse_bn=True)
  QAT 层: {'QATConv2d': 4, 'QATLinear': 2}

  [OpticSpaceNetStudent (QAT)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] stem.0                       3   1×1          3          8   37.5%
  [QATConv2d  ] stage1.0                     8   2×2         32         32   100.0%
  [QATConv2d  ] stage2.0                    16   2×2         64         64   100.0%
  [QATConv2d  ] stage3.0                    32   1×1         32         32   100.0%
  [QATLinear  ] classifier.1               —     —           1024       1024   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

[Phase 2b] 校准 QAT 模型...
[calibrate] Running calibration on 3 batches...
  Batch 1/3 — input range: [-2.118, 2.640]
  Batch 2/3 — input range: [-2.118, 2.640]
  Batch 3/3 — input range: [-2.118, 2.640]
[calibrate] Calibration complete.

[Phase 3] QAT 微调 (20 epochs, lr=0.0001)
  模式: KD + QAT (T=4.0, α=0.5)
  教师模型固定, 学生通过蒸馏损失 + 伪量化进行微调
----------------------------------------------------------------------
   Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     6.5734   60.31% |    1.5117  66.33% |  66.33% |  324.8s
      2  |     4.9508   68.69% |    1.4937  68.04% |  68.04% |  356.6s
      3  |     4.6165   70.94% |    1.4548  67.56% |  68.04% |  363.5s
      4  |     4.4945   71.78% |    1.4969  68.61% |  68.61% |  345.1s
      5  |     4.4239   72.06% |    1.3596  69.87% |  69.87% |  327.3s
      6  |     4.4179   72.09% |    1.4646  68.76% |  69.87% |  397.0s
      7  |     4.3518   72.50% |    1.5537  66.98% |  69.87% |  414.5s
      8  |     4.3038   72.82% |    1.5451  67.96% |  69.87% |  375.6s
      9  |     4.3194   72.56% |    1.2234  71.48% |  71.48% |  365.3s
     10  |     4.2672   73.00% |    1.4156  70.41% |  71.48% |  337.6s
     11  |     4.2250   73.07% |    1.3287  70.57% |  71.48% |  335.4s
     12  |     4.2194   73.01% |    1.2248  72.17% |  72.17% |  332.2s
     13  |     4.1562   73.84% |    1.1573  71.98% |  72.17% |  363.7s
     14  |     4.1643   73.82% |    1.2062  72.46% |  72.46% | 1663.7s
     15  |     4.1347   73.82% |    1.1640  72.91% |  72.91% |  533.5s
     16  |     4.1173   74.26% |    1.1494  72.98% |  72.98% |  271.0s
     17  |     4.1546   73.68% |    1.3284  70.44% |  72.98% |  271.6s
     18  |     4.0818   74.08% |    1.2179  72.28% |  72.98% |  288.2s
     19  |     4.1242   73.98% |    1.1530  73.22% |  73.22% |  252.3s
     20  |     4.0718   74.23% |    1.2332  71.41% |  73.22% |  257.6s

[Phase 4] QAT 训练完成 — 评估与保存

============================================================
  QAT vs Float32 Comparison
============================================================
[enable_qat] Enabled QAT on 6 layers
  QAT mode (int4):     Accuracy = 73.22%
[disable_qat] Disabled QAT on 6 layers
  Float mode (float32): Accuracy = 64.13%
  Accuracy gap:         -9.09% (✓ QAT successful)

  保存 QAT-trained 权重至: spacenet_v2_qat.pth
  文件大小: 1052.0 KB

============================================================
  训练完成 — 结果汇总
============================================================
  学生模型:          OpticSpaceNet (硬件完美对齐)
  学生参数量:        268,210
  教师模型:          ResNet-18
  Float32 基准准确率: 91.44%
  QAT 最佳准确率:     73.22%
  QAT vs Float gap:  -9.09%
  QAT 训练耗时:      8176.3 秒 (136.3 分钟)
  8×2 硬件对齐率:    99.6%
  QAT 模式:          KD + QAT

  预期光计算推理:
    原生 float32:     91.4%
    PTQ (直接 int4):  ~0.8 (大幅下降)
    QAT (微调后 int4): 0.7 (接近 float32)

  提示: 如需更好的精度，使用 KD+QAT 联合微调:
     python model3_spacenet_v2_qat.py --use_kd
```
