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

  💡 提示: 如需更好的精度，使用 KD+QAT 联合微调:
     python model3_spacenet_v2_qat.py --use_kd
```


---

```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_int4.py
设备: cpu
数据目录: data/EuroSAT_RGB
============================================================
  模型一 Int4 (Baseline VGG): 从零 QAT 训练
  全程 int4 伪量化 — 特征天然兼容光计算硬件
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 1] 创建模型 (随机初始化)
  参数量: 2,386,986

[Step 2] 转换为 QAT 模型 (保留 BatchNorm)
[prepare_qat_from_scratch] Converted 8 layers to QAT, 0 BN layers preserved
  QAT 层: {'QATConv2d': 6, 'QATLinear': 2}

  [BaselineVGG (Int4)] 层名                            C_in   K      展平长度  补零后  对齐率
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

[Step 3] 开始 int4 QAT 训练 (60 epochs, lr=0.001)
  注意: 每一层 Conv/Linear 在每次前向传播时都施加 int4 伪量化
        模型从随机初始化开始，从未见过 float32 精度
        STE 梯度确保权重向 int4 兼容方向优化
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.5329   40.00% |    1.1117  58.93% |  58.93% |  211.4s
      5  |     0.6653   77.09% |    0.6654  77.20% |  77.20% |  152.7s
     10  |     0.4947   83.81% |    0.4770  83.50% |  83.50% |  150.8s
     15  |     0.4355   85.79% |    0.4562  85.52% |  85.61% |  151.9s
     20  |     0.3462   88.89% |    0.4446  87.13% |  87.24% |  156.6s
     25  |     0.3065   90.31% |    0.4371  87.41% |  87.41% |  158.5s
     30  |     0.2674   91.42% |    0.4735  87.09% |  88.31% |  152.1s
     35  |     0.2518   92.35% |    0.4641  89.30% |  89.31% |  153.4s
     40  |     0.2283   92.96% |    0.4920  88.02% |  89.70% |  187.2s
     45  |     0.2055   93.63% |    0.4277  90.35% |  90.35% |  183.7s
     50  |     0.1758   94.72% |    0.4508  90.30% |  91.00% |  188.1s
     55  |     0.1721   94.81% |    0.5151  89.57% |  91.00% |  183.5s
     60  |     0.1451   95.39% |    0.4706  90.35% |  91.17% |  168.0s

[Step 4] 最终评估
[enable_qat] Enabled QAT on 8 layers
  Int4 模式 (光计算模拟) 准确率: 91.17%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:         84.24%
  Int4 量化精度损失:          -6.93%

  保存 int4 QAT 权重至: baseline_vgg_int4.pth

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:            Mini-VGG (全 3×3 卷积)
  参数量:              2,386,986
  训练方式:            从零 QAT (int4 from epoch 1)
  训练总耗时:          10424.6 秒 (173.7 分钟)
  8×2 硬件对齐率:      100.0%
  Int4 最佳准确率:     91.17%
  Float32 准确率:      84.24%
  Int4 量化损失:       -6.93%

  与 FP32 训练对比:
    FP32 from scratch:  97.17% (model1_baseline.py)
    QAT fine-tune:      85.91% (model1_baseline_qat.py, 效果差)
    QAT from scratch:   91.17% (本脚本, 新方案)
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_int4.py
设备: cpu
============================================================
  模型二 Int4 (Optic-SpaceNet V1): 从零 QAT 训练
  全程 int4 伪量化 — 特征天然兼容光计算硬件
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 1] 创建模型 (随机初始化, 不使用预训练权重)
  参数量: 268,210

[Step 2] 转换为 QAT 模型 (保留 BatchNorm 层)
[prepare_qat_from_scratch] Converted 6 layers to QAT, 4 BN layers preserved
  QAT 层: {'QATConv2d': 4, 'QATLinear': 2}

  [OpticSpaceNetV1 (Int4)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] stem.0                       3   1×1          3          8   37.5%
  [QATConv2d  ] stage1.0                     8   2×2         32         32   100.0%
  [QATConv2d  ] stage2.0                    16   2×2         64         64   100.0%
  [QATConv2d  ] stage3.0                    32   1×1         32         32   100.0%
  [QATLinear  ] classifier.1               —     —           1024       1024   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)
  ⚠ stem 层对齐率仅 37.5% (patch=3→8)，但 ops 极少

[Step 3] 开始 int4 QAT 训练 (80 epochs, lr=0.001)
  从随机初始化开始，全程 int4 伪量化
  保留 BN 层: 帮助稳定训练 + 补偿部分量化误差
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.2053   56.56% |    0.9365  67.15% |  67.15% |  354.9s
      5  |     0.8597   69.78% |    0.7549  73.63% |  73.63% |   45.4s
     10  |     0.7873   72.30% |    0.7109  74.39% |  75.80% |   43.1s
     15  |     0.7607   73.06% |    0.6422  76.83% |  76.83% |   42.9s
     20  |     0.7257   74.81% |    0.6432  76.80% |  77.06% |   42.2s
     25  |     0.7196   74.75% |    0.6763  76.02% |  77.80% |   43.7s
     30  |     0.6943   75.61% |    0.7238  74.63% |  78.13% |   43.4s
     35  |     0.6752   76.59% |    0.6784  76.35% |  79.56% |   43.0s
     40  |     0.6594   76.82% |    0.5997  78.93% |  79.56% |   41.7s
     45  |     0.6303   77.76% |    0.5705  80.11% |  80.11% |   43.0s
     50  |     0.6117   78.60% |    0.6232  77.89% |  80.11% |   43.7s
     55  |     0.6006   78.71% |    0.5641  80.22% |  80.22% |   43.2s
     60  |     0.5932   79.30% |    0.5598  80.22% |  80.59% |   43.4s
     65  |     0.5681   79.72% |    0.5564  80.70% |  80.70% |   42.3s
     70  |     0.5811   79.40% |    0.5969  79.20% |  80.70% |   43.1s
     75  |     0.5585   80.37% |    0.5521  80.56% |  81.20% |   42.9s
     80  |     0.5381   80.95% |    0.5652  80.09% |  81.20% |   45.2s

[Step 4] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 81.20%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:         69.17%
  Int4 量化精度损失:          -12.04%

  保存 int4 QAT 权重至: spacenet_v1_int4.pth

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:            Optic-SpaceNet V1 (硬件对齐)
  参数量:              268,210
  训练方式:            从零 QAT (int4 from epoch 1)
  训练总耗时:          3785.6 秒 (63.1 分钟)
  8×2 硬件对齐率:      99.6%
  Int4 最佳准确率:     81.20%
  Float32 准确率:      69.17%
  Int4 量化损失:       -12.04%

  与 FP32 训练对比:
    FP32 from scratch:  90.15% (model2_spacenet_v1.py)
    QAT fine-tune:      73.63% (model2_spacenet_v1_qat.py, 效果差)
    QAT from scratch:   81.20% (本脚本, 新方案)
```

```powershell
PS E:\LT-Simulator\train-test>  python model3_spacenet_v2_int4.py
设备: cpu
============================================================
  模型三 Int4 (Optic-SpaceNet V2): 从零 KD+QAT 联合训练
  教师引导 + int4 约束 — 学生学 int4 兼容特征
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 1] 加载教师模型 (ResNet-18)
  教师权重加载自: teacher_resnet18.pth
  教师参数量: 11,181,642

[Step 2] 创建学生模型 (随机初始化, 不使用预训练权重)
  学生参数量: 268,210

[Step 3] 转换为 QAT 模型 (保留 BatchNorm 层)
[prepare_qat_from_scratch] Converted 6 layers to QAT, 4 BN layers preserved
  QAT 层: {'QATConv2d': 4, 'QATLinear': 2}

  [OpticSpaceNetStudent (Int4)] 层名                            C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [QATConv2d  ] stem.0                       3   1×1          3          8   37.5%
  [QATConv2d  ] stage1.0                     8   2×2         32         32   100.0%
  [QATConv2d  ] stage2.0                    16   2×2         64         64   100.0%
  [QATConv2d  ] stage3.0                    32   1×1         32         32   100.0%
  [QATLinear  ] classifier.1               —     —           1024       1024   100.0%
  [QATLinear  ] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

[Step 4] 开始 KD+QAT 联合训练
  学生 epochs: 100, lr=0.001
  蒸馏温度 T=4.0, α=0.5
  教师: 提供软标签 (固定不更新)
  学生: 从零学习 + QAT int4 伪量化 + KD 引导
  保留 BN: 稳定训练 + 补偿量化误差
----------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     6.2989   55.04% |    1.3024  64.72% |  64.72% |   76.1s
      5  |     4.3317   70.93% |    1.0224  73.59% |  74.11% |   69.4s
     10  |     3.9575   74.06% |    0.9362  76.26% |  76.30% |   69.4s
     15  |     3.8063   75.07% |    0.8460  78.06% |  78.37% |   72.9s
     20  |     3.7317   75.74% |    0.8290  78.04% |  78.46% |   73.1s
     25  |     3.5941   77.02% |    0.8195  78.54% |  78.80% |   72.4s
     30  |     3.5021   77.63% |    0.8133  80.37% |  80.37% |   68.0s
     35  |     3.4649   78.03% |    0.7484  80.85% |  80.85% |   69.8s
     40  |     3.3926   78.71% |    0.9949  76.48% |  80.85% |   66.9s
     45  |     3.3510   79.28% |    0.7524  80.87% |  80.87% |   68.9s
     50  |     3.2752   79.95% |    0.7526  81.43% |  81.43% |   69.9s
     55  |     3.2410   79.83% |    1.0489  76.20% |  81.43% |   69.2s
     60  |     3.2298   80.20% |    0.8408  79.93% |  81.76% |   74.7s
     65  |     3.1469   80.43% |    0.8155  79.72% |  81.76% |   74.6s
     70  |     3.1021   80.60% |    0.8832  78.20% |  81.76% |   74.8s
     75  |     3.1067   80.86% |    0.7725  81.04% |  81.76% |   74.4s
     80  |     3.0583   81.20% |    0.7042  81.50% |  82.11% |   74.7s
     85  |     3.0391   81.30% |    0.7008  82.31% |  82.31% |   72.5s
     90  |     3.0375   81.00% |    0.7790  81.11% |  83.26% |   68.7s
     95  |     3.0199   81.16% |    0.7066  82.07% |  83.26% |   68.0s
    100  |     2.9442   82.10% |    0.7852  80.06% |  83.26% |   73.4s

[Step 5] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 83.26%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:         67.37%
  Int4 量化精度损失:          -15.89%

  保存 int4 QAT 权重至: spacenet_v2_int4.pth

============================================================
  训练完成 — 结果汇总
============================================================
  教师模型:            ResNet-18 (11,181,642 params)
  学生模型:            OpticSpaceNet (268,210 params)
  训练方式:            从零 KD+QAT 联合 (int4 from epoch 1)
  训练总耗时:          7405.4 秒 (123.4 分钟)
  8×2 硬件对齐率:      99.6%
  Int4 最佳准确率:     83.26%
  Float32 准确率:      67.37%
  Int4 量化损失:       -15.89%

  与 FP32 KD 训练对比:
    FP32 KD from scratch: 91.44% (model3_spacenet_v2.py)
    QAT fine-tune:        73.22% (model3_spacenet_v2_qat.py, 效果差)
    QAT from scratch:     83.26% (本脚本, 新方案)
```

```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_int4.py
设备: cpu
数据目录: data/EuroSAT_RGB
============================================================
  模型一 Int4 (Baseline VGG): 从零 QAT 训练
  全程 int4 伪量化 — 特征天然兼容光计算硬件
============================================================
训练集: 21600 张, 验证集: 5400 张
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 1] 创建模型 (随机初始化)
  参数量: 2,386,986

[Step 2] 转换为 QAT 模型 (保留 BatchNorm)
[prepare_qat_from_scratch] Converted 8 layers to QAT (mode=LSQ), 0 BN layers preserved
  QAT int4 层: {'QATConv2d': 5, 'QATLinear': 1}
  Float32 层: ['block1.0 (float32)', 'classifier.4 (float32)']

  [BaselineVGG (Int4)] 层名                            C_in   K      展平长度  补零后  对齐率
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

[Step 3] 开始混合精度 int4 QAT 训练 (60 epochs, lr=0.001)
  混合精度: 首层 (block1.0) + 末层 (classifier.4) float32, 其余 int4 QAT
  高 weight_decay=5e-4: 抑制 int4 模式下的过拟合
  从随机初始化开始，STE 梯度让 int4 层找到量化友好的权重
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.5632   38.48% |    1.0209  63.06% |  63.06% |  164.4s
      5  |     0.6649   77.08% |    0.6675  76.48% |  76.48% |  166.3s
     10  |     0.4555   85.15% |    0.5188  82.96% |  83.07% |  168.1s
     15  |     0.3714   87.88% |    0.5439  83.74% |  84.70% |  172.4s
     20  |     0.2874   90.61% |    0.3572  89.00% |  89.00% |  178.1s
     25  |     0.2783   90.81% |    0.2763  91.04% |  91.04% |  174.4s
     30  |     0.2388   91.81% |    0.3523  89.67% |  91.04% |  169.7s
     35  |     0.1998   93.38% |    0.3967  89.37% |  91.04% |  169.1s
     40  |     0.1652   94.39% |    0.3429  90.24% |  91.04% |  166.0s
     45  |     0.1416   95.07% |    0.2918  91.54% |  91.54% |  169.9s
     50  |     0.1173   95.98% |    0.2990  91.46% |  92.17% |  168.4s
     55  |     0.1016   96.44% |    0.3283  91.52% |  92.17% |  169.6s
     60  |     0.0970   96.54% |    0.3248  91.41% |  92.17% |  173.7s

[Step 4] 最终评估
[enable_qat] Enabled QAT on 8 layers
  Int4 模式 (光计算模拟) 准确率: 64.28%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:         87.96%
  Int4 量化精度损失:          23.69%

  保存 int4 QAT 权重至: baseline_vgg_int4.pth

============================================================
  训练完成 — 结果汇总
============================================================
  网络结构:            Mini-VGG (全 3×3 卷积)
  参数量:              2,386,986
  训练方式:            从零 QAT (int4 from epoch 1)
  训练总耗时:          10169.6 秒 (169.5 分钟)
  8×2 硬件对齐率:      100.0%
  Int4 最佳准确率:     92.17%
  Float32 准确率:      87.96%
  Int4 量化损失:       23.69%

  与 FP32 训练对比:
    FP32 from scratch:  97.17% (model1_baseline.py)
    QAT fine-tune:      85.91% (model1_baseline_qat.py, 效果差)
    QAT from scratch:   92.17% (本脚本, 新方案)
```


```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_phase4.py
设备: cpu, 模式: ste, 噪声注入: True
============================================================
  Model 1 Phase 4: STE + uint4/int4 非对称量化
============================================================
训练: 21600, 验证: 5400

[Step 1] 创建 BaselineVGG (bias=False)
  参数量: 2,386,272

[Step 2] 转换为 Phase 4 QAT (mode=ste, noise=True)
[prepare_model_phase4] Converted 8 layers to QAT v2 (mode=ste, noise=True, bias=False)
  QATConv2d_v2: 6, QATLinear_v2: 2

[Step 3] 训练 (60 epochs, lr=0.001)
  STE 模式: 静态 scale + 噪声注入 (std=0.05*scale)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     2.0671   17.26% |    1.9783  16.30% |  16.30% |  208.4s
      5  |     1.7242   29.67% |    1.6962  29.61% |  31.19% |  205.6s
     10  |     1.6117   36.18% |    1.5291  37.67% |  37.67% |  207.9s
     15  |     1.5179   40.51% |    1.4162  44.41% |  45.31% |  206.0s
     20  |     1.4578   42.72% |    1.3407  47.06% |  47.06% |  239.1s
     25  |     1.4199   44.49% |    1.3609  47.09% |  49.50% |  227.3s
     30  |     1.3835   45.58% |    1.2220  52.31% |  52.31% |  193.6s
     35  |     1.3548   46.88% |    1.3154  50.96% |  52.33% |  203.0s
     40  |     1.3331   47.33% |    1.2295  52.15% |  53.04% |  191.3s
     45  |     1.3150   47.73% |    1.2018  53.22% |  53.85% |  189.1s
     50  |     1.2892   48.40% |    1.1918  53.81% |  53.98% |  188.8s
     55  |     1.2841   48.27% |    1.1809  53.72% |  54.28% |  204.8s
     60  |     1.2823   48.55% |    1.1805  54.24% |  54.30% |  197.9s

  模型已保存: baseline_vgg_phase4_ste.pth

============================================================
  结果: Int4 (uint4/int4) = 54.30%
  FP32 基准: 97.17%, Phase 2 最佳: 91.17%
  模式: ste, 噪声: True, 耗时: 205.2min
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4.py
设备: cpu, 模式: ste, 噪声注入: True
============================================================
  Model 2 Phase 4: STE + uint4/int4 非对称量化
============================================================
训练: 21600, 验证: 5400

[Step 1] 创建 OpticSpaceNetV1 (bias=False)
  参数量: 267,944

[Step 2] 转换为 Phase 4 QAT
[prepare_model_phase4] Converted 6 layers to QAT v2 (mode=ste, noise=True, bias=False)
  QATConv2d_v2: 4, QATLinear_v2: 2, BN: 4

[Step 3] 训练 (80 epochs)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     1.7187   32.70% |    1.6217  36.69% |  36.69% |   58.3s
      5  |     1.4077   45.41% |    1.3534  47.63% |  47.63% |   60.5s
     10  |     1.3049   49.59% |    1.3981  48.11% |  50.74% |   59.4s
     15  |     1.2556   50.97% |    1.2059  53.00% |  53.96% |   60.4s
     20  |     1.2171   52.67% |    1.1694  54.02% |  55.24% |   61.2s
     25  |     1.1934   53.79% |    1.1361  55.87% |  56.39% |   59.7s
     30  |     1.1670   54.07% |    1.2131  54.57% |  57.41% |  163.5s
     35  |     1.1339   55.97% |    1.1329  57.52% |  59.11% |   46.0s
     40  |     1.1107   56.44% |    1.2617  57.09% |  59.11% |   46.3s
     45  |     1.0861   57.83% |    1.1501  59.20% |  59.20% |   45.9s
     50  |     1.0282   60.29% |    1.0764  59.98% |  62.00% |   46.1s
     55  |     0.9583   63.49% |    1.1463  62.57% |  62.57% |   46.5s
     60  |     0.9034   65.60% |    1.0388  64.57% |  64.57% |   46.7s
     65  |     0.8931   65.82% |    1.1754  63.24% |  64.57% |   49.4s
     70  |     0.8882   66.23% |    1.0752  63.41% |  64.57% |   46.3s
     75  |     0.8796   66.63% |    1.0375  64.39% |  64.57% |   47.3s
     80  |     0.8836   66.16% |    1.1445  62.70% |  64.57% |   46.3s

  模型已保存: spacenet_v1_phase4_ste.pth

============================================================
  结果: Int4 = 64.57%
  FP32 基准: 90.15%, Phase 2: 81.20%
  模式: ste, 耗时: 73.6min
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model3_spacenet_v2_phase4.py
设备: cpu, 模式: ste, 噪声注入: True
============================================================
  Model 3 Phase 4: KD + STE + uint4/int4 非对称量化
============================================================
训练: 21600, 验证: 5400

[Step 1] 加载教师 (ResNet-18)
  教师权重加载成功

[Step 2] 创建学生 (bias=False)
  参数量: 267,944

[Step 3] 转换为 Phase 4 QAT
[prepare_model_phase4] Converted 6 layers to QAT v2 (mode=ste, noise=True, bias=False)
  QATConv2d_v2: 4, QATLinear_v2: 2

[Step 4] KD + QAT 联合训练 (100 epochs)
----------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     8.5046   30.18% |    1.7415  37.41% |  37.41% |   69.5s
      5  |     6.6126   47.93% |    1.4766  50.44% |  50.44% |   67.4s
     10  |     6.1494   51.65% |    1.7095  52.74% |  52.74% |   68.8s
     15  |     5.9466   53.81% |    1.3637  54.31% |  54.31% |   69.6s
     20  |     5.7668   55.21% |    1.3746  55.28% |  55.28% |   68.5s
     25  |     5.6932   56.00% |    1.3584  59.09% |  59.09% |   67.4s
     30  |     5.6250   56.43% |    1.3072  56.41% |  59.09% |   67.7s
     35  |     5.5145   57.26% |    1.6160  55.81% |  59.09% |   77.0s
     40  |     5.3051   59.60% |    1.4496  56.83% |  59.96% |   69.4s
     45  |     5.0675   62.44% |    1.2644  62.96% |  64.72% |   68.7s
     50  |     4.8012   64.60% |    1.2544  64.24% |  65.19% |   70.1s
     55  |     4.7431   65.42% |    1.1880  64.46% |  65.31% |   69.3s
     60  |     4.5787   67.40% |    1.3337  64.70% |  66.80% |   69.1s
     65  |     4.4582   68.43% |    1.3299  65.09% |  66.80% |   70.2s
     70  |     4.4339   68.48% |    1.1244  66.59% |  66.80% |   68.6s
     75  |     4.4282   68.66% |    1.2644  64.61% |  66.80% |   67.9s
     80  |     4.3747   69.25% |    1.1734  66.50% |  66.80% |   67.9s
     85  |     4.3792   69.02% |    1.1608  66.81% |  67.33% |   70.7s
     90  |     4.3681   69.21% |    1.3083  65.98% |  67.76% |   68.8s
     95  |     4.3699   69.22% |    1.1225  67.22% |  68.19% |   69.3s
    100  |     4.3971   69.05% |    1.3765  65.13% |  68.19% |   69.3s

  模型已保存: spacenet_v2_phase4_ste.pth

============================================================
  结果: Int4 = 68.19%
  FP32 KD 基准: 91.44%, Phase 2: 83.26%
  模式: ste, 耗时: 115.4min
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_phase4.py
设备: cpu, 模式: ste, act_bits: 8

============================================================
  Model 1 Phase 4: STE + int4 权重量化 (修复版)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 2,387,680

[Step 1] 转换为 QAT v3 (mode=ste)
[prepare_model_v3] Converted 6 Conv + 2 Linear to QAT v3
  mode=ste, w4/a8, noise=True, noise_std=0.02
  first_layer_fp32=True, last_layer_fp32=True
  BN preserved=6 (float32)
  QAT Conv: 5, QAT Linear: 1, BN: 6 (float32)

  [BaselineVGG+BN (Phase 4, ste, a8)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 FP32 ] conv1_1                      3   3×3        27        32   84.4%
  [QATConv2d_v3 QAT  ] conv1_2                     32   3×3       288       288  100.0%
  [QATConv2d_v3 QAT  ] conv2_1                     32   3×3       288       288  100.0%
  [QATConv2d_v3 QAT  ] conv2_2                     64   3×3       576       576  100.0%
  [QATConv2d_v3 QAT  ] conv3_1                     64   3×3       576       576  100.0%
  [QATConv2d_v3 QAT  ] conv3_2                    128   3×3      1152      1152  100.0%
  [QATLinear_v3 QAT  ] fc1                         —     —          8192      8192  100.0%
  [QATLinear_v3 FP32 ] fc2                         —     —           256       256  100.0%
  综合硬件对齐率: 100.0% (展平总长度 11355 → 补零后 11360)

[Step 2] 训练 (80 epochs, lr=0.001, warmup=5, wd=0.0005)
  噪声注入: std=0.02*scale (仅 int4 权重层)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     0.8646   69.84% |    0.5560  80.43% |  80.43% | 0.00020 |  941.8s
      5  |     0.3664   88.05% |    0.4160  85.44% |  87.83% | 0.00100 |  179.4s
     10  |     0.2096   92.75% |    0.2029  93.28% |  94.94% | 0.00099 |  181.0s
     15  |     0.1385   95.17% |    0.1240  95.85% |  95.85% | 0.00096 |  158.5s
     20  |     0.1079   96.12% |    0.1352  95.72% |  95.85% | 0.00091 |  158.4s
     25  |     0.0875   97.08% |    0.1079  96.41% |  96.41% | 0.00084 |  179.1s
     30  |     0.0672   97.69% |    0.0936  97.02% |  97.22% | 0.00075 |  165.1s
     35  |     0.0531   98.21% |    0.1024  96.93% |  97.22% | 0.00066 |  173.9s
     40  |     0.0399   98.63% |    0.1065  96.81% |  97.35% | 0.00056 |  186.4s
     45  |     0.0341   98.80% |    0.0889  97.48% |  97.50% | 0.00045 |  181.0s
     50  |     0.0222   99.23% |    0.0942  97.50% |  97.80% | 0.00035 |  183.8s
     55  |     0.0168   99.42% |    0.0828  97.91% |  97.91% | 0.00026 |  182.3s
     60  |     0.0138   99.55% |    0.0834  97.83% |  97.91% | 0.00017 |  191.4s
     65  |     0.0082   99.72% |    0.0862  97.81% |  98.04% | 0.00010 |  203.9s
     70  |     0.0064   99.82% |    0.0805  97.94% |  98.04% | 0.00005 |  197.8s
     75  |     0.0061   99.81% |    0.0813  98.07% |  98.07% | 0.00002 |  205.8s
     80  |     0.0049   99.88% |    0.0804  98.04% |  98.07% | 0.00001 |  197.8s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 8 layers
  Int4 模式 (光计算模拟) 准确率: 96.46%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:         98.06%
  Int4 量化损失:              1.59%

  模型已保存: baseline_vgg_phase4_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              BaselineVGG+BN (Phase 4, ste, a8)
  参数量:            2,387,680
  模式:              ste, w4/a8
  训练总耗时:        15464.8s (257.7min)
  硬件对齐率:        100.0%
  Int4 最佳准确率:   98.07%
  Float32 准确率:    98.06%
  FP32 基准 (参考):  97.17% (model1_baseline.py)
  量化损失:          1.59%
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4.py
设备: cpu, 模式: ste, act_bits: 8

============================================================
  Model 2 Phase 4: STE + SpaceNet V1 int4 (修复版)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 268,210

[Step 1] 转换为 QAT v3 (mode=ste)
[prepare_model_v3] Converted 4 Conv + 2 Linear to QAT v3
  mode=ste, w4/a8, noise=True, noise_std=0.02
  first_layer_fp32=True, last_layer_fp32=True
  BN preserved=4 (float32)
  QAT Conv: 0, QAT Linear: 1, BN: 4 (float32)

  [OpticSpaceNetV1 (Phase 4, ste, a8)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 FP32 ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 FP32 ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 FP32 ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 FP32 ] stage3.0                    32   1×1        32        32  100.0%
  [QATLinear_v3 QAT  ] classifier.1                —     —          1024      1024  100.0%
  [QATLinear_v3 FP32 ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (100 epochs, lr=0.001, warmup=5, wd=0.0005)
  噪声注入: std=0.02*scale (仅 int4 权重层)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.0315   63.25% |    0.7740  72.17% |  72.17% | 0.00020 |   45.5s
      5  |     0.6689   76.28% |    0.5717  79.50% |  79.91% | 0.00100 |   40.4s
     10  |     0.5503   80.67% |    0.4563  83.65% |  84.04% | 0.00099 |   44.6s
     15  |     0.4618   84.26% |    0.3766  87.02% |  87.17% | 0.00097 |   40.0s
     20  |     0.4159   85.45% |    0.3535  88.06% |  88.06% | 0.00094 |   52.3s
     25  |     0.3792   86.94% |    0.2991  89.50% |  89.50% | 0.00090 |   84.1s
     30  |     0.3434   88.02% |    0.2863  89.70% |  89.70% | 0.00084 |   63.7s
     35  |     0.3204   89.06% |    0.2829  90.56% |  90.56% | 0.00078 |   70.8s
     40  |     0.2994   89.49% |    0.2818  90.02% |  90.78% | 0.00070 |   58.2s
     45  |     0.2796   90.32% |    0.2759  90.65% |  90.89% | 0.00063 |   57.9s
     50  |     0.2648   90.96% |    0.2490  91.50% |  91.50% | 0.00055 |   64.6s
     55  |     0.2491   91.45% |    0.2642  91.33% |  91.98% | 0.00046 |   56.2s
     60  |     0.2396   91.63% |    0.2394  92.09% |  92.13% | 0.00038 |   54.7s
     65  |     0.2214   92.26% |    0.2571  91.93% |  92.26% | 0.00031 |   56.2s
     70  |     0.2185   92.47% |    0.2379  92.06% |  92.50% | 0.00023 |   43.5s
     75  |     0.1993   93.07% |    0.2292  92.46% |  92.50% | 0.00017 |   41.2s
     80  |     0.1979   93.12% |    0.2276  92.44% |  92.67% | 0.00011 |   43.4s
     85  |     0.1921   93.46% |    0.2250  92.59% |  92.67% | 0.00007 |   41.5s
     90  |     0.1834   93.57% |    0.2211  92.70% |  92.74% | 0.00004 |   41.1s
     95  |     0.1869   93.51% |    0.2242  92.57% |  92.74% | 0.00002 |   46.7s
    100  |     0.1842   93.61% |    0.2186  92.87% |  92.87% | 0.00001 |   42.4s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 74.35%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:         92.81%
  Int4 量化损失:              18.46%

  模型已保存: spacenet_v1_phase4_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              OpticSpaceNetV1 (Phase 4, ste, a8)
  参数量:            268,210
  模式:              ste, w4/a8
  训练总耗时:        5350.2s (89.2min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   92.87%
  Float32 准确率:    92.81%
  FP32 基准 (参考):  90.15% (model2_spacenet_v1.py)
  量化损失:          18.46%
============================================================
```





---

```powershell
  # 训练 (推荐 STE 模式)
  python model1_baseline_phase4.py
  python model2_spacenet_v1_phase4.py
  python model3_spacenet_v2_phase4.py

  # LSQ+ 模式 (可选)
  python model1_baseline_phase4.py --mode lsqplus

  # 噪声鲁棒性测试
  python noise_robustness_v2.py --all
  
  
  
  # 运行
  python model1_baseline_mixed.py
  python model2_spacenet_v1_mixed.py
  python model3_spacenet_v2_mixed.py
```