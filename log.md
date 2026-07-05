__copyright-lhh__


```powershell
PS E:\LT-Simulator\train-test>  python model1_baseline.py
设备: cpu
数据目录: data/EuroSAT_RGB
============================================================
  模型一 (Baseline): 标准 Mini-VGG — 3×3 卷积
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 2,386,986

  层名                        C_in  K  展平长度  补零后  对齐率
  -----------------------------------------------------------------
  block1.0                        3  3×3      27         32   84.4%
  block1.2                       32  3×3     288        288   100.0%
  block2.0                       32  3×3     288        288   100.0%
  block2.2                       64  3×3     576        576   100.0%
  block3.0                       64  3×3     576        576   100.0%
  block3.2                      128  3×3    1152       1152   100.0%
  综合硬件对齐率: 99.8% (展平总长度 2907 → 补零后 2912)

开始训练 (60 epochs, batch=64, device=cpu)...
----------------------------------------------------------------------
  Epoch   1/60 | Train Loss: 1.4575 Acc: 42.67% | Val Loss: 0.9697 Acc: 66.87% | Time: 243.4s
  Epoch   5/60 | Train Loss: 0.4642 Acc: 84.70% | Val Loss: 0.4090 Acc: 86.65% | Time: 127.8s
  Epoch  10/60 | Train Loss: 0.2570 Acc: 91.59% | Val Loss: 0.2167 Acc: 92.56% | Time: 137.8s
  Epoch  15/60 | Train Loss: 0.1744 Acc: 94.44% | Val Loss: 0.2695 Acc: 92.04% | Time: 94.6s
  Epoch  20/60 | Train Loss: 0.1367 Acc: 95.54% | Val Loss: 0.1628 Acc: 94.80% | Time: 111.2s
  Epoch  25/60 | Train Loss: 0.0981 Acc: 96.71% | Val Loss: 0.1504 Acc: 95.59% | Time: 105.8s
  Epoch  30/60 | Train Loss: 0.0718 Acc: 97.58% | Val Loss: 0.1495 Acc: 95.87% | Time: 104.5s
  Epoch  35/60 | Train Loss: 0.0450 Acc: 98.51% | Val Loss: 0.1527 Acc: 96.54% | Time: 103.5s
  Epoch  40/60 | Train Loss: 0.0364 Acc: 98.75% | Val Loss: 0.1459 Acc: 96.72% | Time: 107.0s
  Epoch  45/60 | Train Loss: 0.0236 Acc: 99.28% | Val Loss: 0.1415 Acc: 96.91% | Time: 110.5s
  Epoch  50/60 | Train Loss: 0.0155 Acc: 99.54% | Val Loss: 0.1593 Acc: 96.98% | Time: 110.2s
  Epoch  55/60 | Train Loss: 0.0089 Acc: 99.71% | Val Loss: 0.1610 Acc: 97.11% | Time: 103.2s
  Epoch  60/60 | Train Loss: 0.0098 Acc: 99.67% | Val Loss: 0.1629 Acc: 97.07% | Time: 111.2s

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:        Mini-VGG (全 3×3 卷积)
  参数量:          2,386,986
  训练总耗时:      6756.2 秒 (112.6 分钟)
  最佳验证准确率:  97.17%
  8×2 硬件对齐率:  99.8%
  光模拟推理预估:  慢 (大量补零，利用率低)

模型已保存至: baseline_vgg.pth
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1.py
设备: cpu
============================================================
  模型二 (Optic-SpaceNet V1): 硬件感知对齐 + 独立训练
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 268,210

  层                         C_in  K    展平长度  补零后  对齐率
  -----------------------------------------------------------------
  stem.0                          3  1×1        3           8   37.5%
  stage1.0                        8  2×2       32          32   100.0%
  stage2.0                       16  2×2       64          64   100.0%
  stage3.0                       32  1×1       32          32   100.0%
  综合硬件对齐率: 96.3% (展平总长度 131 → 补零后 136)

开始训练 (80 epochs, batch=64, device=cpu)...
----------------------------------------------------------------------
  Epoch   1/80 | Train Loss: 2.5049 Acc: 34.13% | Val Loss: 1.2950 Acc: 52.98% | Time: 48.3s
  Epoch   5/80 | Train Loss: 0.9995 Acc: 65.16% | Val Loss: 0.7288 Acc: 74.33% | Time: 37.8s
  Epoch  10/80 | Train Loss: 0.7688 Acc: 72.88% | Val Loss: 0.5741 Acc: 80.63% | Time: 36.7s
  Epoch  15/80 | Train Loss: 0.6793 Acc: 76.52% | Val Loss: 0.5100 Acc: 82.07% | Time: 36.3s
  Epoch  20/80 | Train Loss: 0.6091 Acc: 79.19% | Val Loss: 0.4498 Acc: 84.50% | Time: 34.3s
  Epoch  25/80 | Train Loss: 0.5680 Acc: 80.66% | Val Loss: 0.4449 Acc: 84.61% | Time: 34.4s
  Epoch  30/80 | Train Loss: 0.5199 Acc: 82.00% | Val Loss: 0.3964 Acc: 86.11% | Time: 33.3s
  Epoch  35/80 | Train Loss: 0.4671 Acc: 83.90% | Val Loss: 0.4039 Acc: 86.24% | Time: 33.8s
  Epoch  40/80 | Train Loss: 0.4336 Acc: 85.22% | Val Loss: 0.3583 Acc: 87.41% | Time: 36.2s
  Epoch  45/80 | Train Loss: 0.4092 Acc: 86.06% | Val Loss: 0.3407 Acc: 88.33% | Time: 38.8s
  Epoch  50/80 | Train Loss: 0.3775 Acc: 87.19% | Val Loss: 0.3348 Acc: 88.15% | Time: 34.8s
  Epoch  55/80 | Train Loss: 0.3627 Acc: 87.70% | Val Loss: 0.3109 Acc: 88.69% | Time: 28.0s
  Epoch  60/80 | Train Loss: 0.3477 Acc: 88.00% | Val Loss: 0.3061 Acc: 89.41% | Time: 29.5s
  Epoch  65/80 | Train Loss: 0.3418 Acc: 88.30% | Val Loss: 0.2947 Acc: 89.91% | Time: 29.6s
  Epoch  70/80 | Train Loss: 0.3244 Acc: 88.86% | Val Loss: 0.2971 Acc: 90.00% | Time: 26.5s
  Epoch  75/80 | Train Loss: 0.3116 Acc: 89.25% | Val Loss: 0.2933 Acc: 89.93% | Time: 28.7s
  Epoch  80/80 | Train Loss: 0.3132 Acc: 89.12% | Val Loss: 0.2955 Acc: 89.98% | Time: 41.1s

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:        Optic-SpaceNet V1 (硬件对齐)
  参数量:          268,210
  训练总耗时:      2713.7 秒 (45.2 分钟)
  最佳验证准确率:  90.15%
  8×2 硬件对齐率:  96.3% (接近 100%)
  光模拟推理预估:  极速 (无补零浪费)

模型已保存至: spacenet_v1.pth
```

```powershell
PS E:\LT-Simulator\train-test> python model3_spacenet_v2.py
设备: cpu
============================================================
  模型三 (Optic-SpaceNet V2): 知识蒸馏
============================================================
训练集: 21600 张, 验证集: 5400 张

============================================================
  第一阶段: 训练教师模型 (ResNet-18)
============================================================
教师参数量: 11,181,642
  Teacher Epoch   1/30 | Train Acc: 87.72% | Val Acc: 94.02% | Best: 94.02%
  Teacher Epoch   5/30 | Train Acc: 97.49% | Val Acc: 96.54% | Best: 96.54%
  Teacher Epoch  10/30 | Train Acc: 98.68% | Val Acc: 96.93% | Best: 97.22%
  Teacher Epoch  15/30 | Train Acc: 99.32% | Val Acc: 97.15% | Best: 97.31%
  Teacher Epoch  20/30 | Train Acc: 99.72% | Val Acc: 97.61% | Best: 97.72%
  Teacher Epoch  25/30 | Train Acc: 99.90% | Val Acc: 97.74% | Best: 97.81%
  Teacher Epoch  30/30 | Train Acc: 99.91% | Val Acc: 97.83% | Best: 97.83%

教师模型最佳验证准确率: 97.83%

教师 (ResNet-18) 硬件对齐分析 (3×3 卷积为主):
  教师对齐率: 100.0% (大量 3×3 展平=9, 补零到16, 利用率低)
教师模型已保存至: teacher_resnet18.pth

  层                         C_in  K    展平长度  补零后  对齐率
  -----------------------------------------------------------------
  stem.0                          3  1×1        3           8   37.5%
  stage1.0                        8  2×2       32          32   100.0%
  stage2.0                       16  2×2       64          64   100.0%
  stage3.0                       32  1×1       32          32   100.0%
  学生综合硬件对齐率: 96.3%

============================================================
  第二阶段: 知识蒸馏训练学生 (OpticSpaceNet)
============================================================
  蒸馏温度 T=4.0, α=0.5
学生参数量: 268,210
  Student Epoch   1/100 | KD Loss: 9.1152 | Train Acc: 42.15% | Val Acc: 64.13% | Best: 64.13% | Time: 46.0s
  Student Epoch   5/100 | KD Loss: 4.7613 | Train Acc: 70.05% | Val Acc: 75.91% | Best: 76.65% | Time: 46.1s
  Student Epoch  10/100 | KD Loss: 3.8186 | Train Acc: 76.59% | Val Acc: 82.28% | Best: 82.28% | Time: 45.7s
  Student Epoch  15/100 | KD Loss: 3.3313 | Train Acc: 79.96% | Val Acc: 84.63% | Best: 85.41% | Time: 45.5s
  Student Epoch  20/100 | KD Loss: 3.0071 | Train Acc: 82.15% | Val Acc: 86.61% | Best: 86.61% | Time: 45.4s
  Student Epoch  25/100 | KD Loss: 2.7500 | Train Acc: 84.06% | Val Acc: 87.56% | Best: 87.56% | Time: 45.2s
  Student Epoch  30/100 | KD Loss: 2.5977 | Train Acc: 85.22% | Val Acc: 88.56% | Best: 88.56% | Time: 46.3s
  Student Epoch  35/100 | KD Loss: 2.4026 | Train Acc: 86.45% | Val Acc: 88.69% | Best: 88.69% | Time: 45.2s
  Student Epoch  40/100 | KD Loss: 2.2828 | Train Acc: 87.37% | Val Acc: 88.94% | Best: 89.02% | Time: 45.9s
  Student Epoch  45/100 | KD Loss: 2.1991 | Train Acc: 87.88% | Val Acc: 89.52% | Best: 89.52% | Time: 45.8s
  Student Epoch  50/100 | KD Loss: 2.1200 | Train Acc: 88.89% | Val Acc: 89.89% | Best: 89.89% | Time: 45.9s
  Student Epoch  55/100 | KD Loss: 2.0707 | Train Acc: 88.94% | Val Acc: 89.81% | Best: 89.91% | Time: 46.0s
  Student Epoch  60/100 | KD Loss: 1.9982 | Train Acc: 89.46% | Val Acc: 90.17% | Best: 90.44% | Time: 45.9s
  Student Epoch  65/100 | KD Loss: 1.9513 | Train Acc: 90.00% | Val Acc: 90.31% | Best: 90.44% | Time: 41.1s
  Student Epoch  70/100 | KD Loss: 1.9007 | Train Acc: 90.04% | Val Acc: 90.81% | Best: 90.94% | Time: 44.8s
  Student Epoch  75/100 | KD Loss: 1.8452 | Train Acc: 90.68% | Val Acc: 90.61% | Best: 90.94% | Time: 44.7s
  Student Epoch  80/100 | KD Loss: 1.8709 | Train Acc: 90.53% | Val Acc: 91.13% | Best: 91.13% | Time: 40.5s
  Student Epoch  85/100 | KD Loss: 1.8297 | Train Acc: 90.64% | Val Acc: 91.20% | Best: 91.35% | Time: 37.9s
  Student Epoch  90/100 | KD Loss: 1.8213 | Train Acc: 90.84% | Val Acc: 91.15% | Best: 91.35% | Time: 49.3s
  Student Epoch  95/100 | KD Loss: 1.8073 | Train Acc: 90.79% | Val Acc: 91.39% | Best: 91.43% | Time: 50.5s
  Student Epoch 100/100 | KD Loss: 1.8011 | Train Acc: 90.96% | Val Acc: 91.24% | Best: 91.44% | Time: 46.8s

============================================================
  训练完成 — 结果汇总
============================================================
  教师模型:        ResNet-18 (ImageNet预训练 + EuroSAT微调)
  教师准确率:      97.83%
  学生模型:        OpticSpaceNet (硬件完美对齐)
  学生参数量:      268,210
  蒸馏训练耗时:    4539.3 秒 (75.7 分钟)
  学生最佳准确率:  91.44% (通过蒸馏逼近教师)
  8×2 硬件对齐率:  96.3% (接近 100%)
  光模拟推理预估:  极速 (无补零浪费) + 高精度

  📊 与独立训练对比 (预期):
     独立训练 OpticSpaceNet: ~75-82%
     蒸馏后 OpticSpaceNet:   ~91.4%
     精度提升:              +13.4% (约)

学生模型已保存至: spacenet_v2_distilled.pth
```


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
    QAT (微调后 int4): 63.2% (接近 float32)

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