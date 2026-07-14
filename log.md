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

```powershell
PS E:\LT-Simulator\train-test>  python model3_spacenet_v2_phase4.py
设备: cpu, 模式: ste, act_bits: 8

============================================================
  Model 3 Phase 4: KD + STE + int4 (修复版)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 0] 加载教师模型
  教师权重加载成功: teacher_resnet18.pth
  学生参数量: 268,210

[Step 1] 转换学生模型为 QAT v3
[prepare_model_v3] Converted 4 Conv + 2 Linear to QAT v3
  mode=ste, w4/a8, noise=True, noise_std=0.02
  first_layer_fp32=True, last_layer_fp32=True
  BN preserved=4 (float32)
  QAT Conv: 0, QAT Linear: 1

  [Student (QAT)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 FP32 ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 FP32 ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 FP32 ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 FP32 ] stage3.0                    32   1×1        32        32  100.0%
  [QATLinear_v3 QAT  ] classifier.1                —     —          1024      1024  100.0%
  [QATLinear_v3 FP32 ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] KD+QAT 训练 (120 epochs, lr=0.001)
  蒸馏温度 T=4.0, α=0.7
  教师: ResNet-18 (固定), 学生: OpticSpaceNet (QAT)
----------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     9.5194   61.87% |    0.9068  74.89% |  74.89% |   77.1s
      5  |     6.1768   76.51% |    0.7012  80.11% |  80.11% |   64.1s
     10  |     5.1562   81.83% |    0.5250  86.20% |  86.20% |   67.2s
     15  |     4.8112   83.59% |    0.4865  87.59% |  87.59% |   66.8s
     20  |     4.3875   85.74% |    0.4417  88.59% |  88.59% |   73.0s
     25  |     4.1943   86.44% |    0.4272  88.78% |  89.39% |   73.3s
     30  |     3.9985   87.28% |    0.4146  89.56% |  89.87% |   71.1s
     35  |     3.8529   88.17% |    0.3836  90.39% |  90.44% |   72.4s
     40  |     3.6923   88.72% |    0.4185  89.39% |  91.04% |   73.5s
     45  |     3.5886   89.55% |    0.3605  90.78% |  91.04% |   70.9s
     50  |     3.4668   89.94% |    0.3573  90.87% |  91.44% |   72.8s
     55  |     3.3837   90.21% |    0.3214  91.87% |  91.87% |   76.6s
     60  |     3.3057   90.70% |    0.3067  91.98% |  92.20% |   76.7s
     65  |     3.2638   90.75% |    0.3339  91.54% |  92.24% |   72.2s
     70  |     3.1698   91.25% |    0.3199  92.22% |  92.35% |   71.9s
     75  |     3.1181   91.37% |    0.2996  92.15% |  92.65% |   72.1s
     80  |     3.0490   91.66% |    0.2897  92.65% |  92.65% |   71.6s
     85  |     3.0050   91.75% |    0.2839  92.72% |  92.80% |   74.7s
     90  |     3.0199   91.96% |    0.2855  92.89% |  92.94% |   69.9s
     95  |     2.9246   92.12% |    0.2801  92.83% |  93.15% |   71.0s
    100  |     2.9180   92.43% |    0.2801  92.67% |  93.15% |   71.8s
    105  |     2.9171   92.28% |    0.2799  92.98% |  93.15% |   75.4s
    110  |     2.8735   92.50% |    0.2759  93.04% |  93.22% |   73.3s
    115  |     2.8726   92.52% |    0.2709  92.81% |  93.22% |   76.7s
    120  |     2.8326   92.61% |    0.2838  92.37% |  93.22% |  137.0s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 78.26%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:         93.15%

  模型已保存: spacenet_v2_phase4_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  教师模型:          ResNet-18
  学生模型:          OpticSpaceNet (硬件对齐)
  学生参数量:        268,210
  蒸馏配置:          T=4.0, α=0.7
  训练总耗时:        9054.0s (150.9min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   93.22%
  Float32 准确率:    93.15%
  FP32 KD 基准:      91.44% (model3_spacenet_v2.py, FP32 KD)
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_phase4.py --mode lsqplus
设备: cpu, 模式: lsqplus, 噪声注入: False
============================================================
  Model 1 Phase 4: LSQPLUS + uint4/int4 非对称量化
============================================================
训练: 21600, 验证: 5400

[Step 1] 创建 BaselineVGG (bias=False)
  参数量: 2,386,272

[Step 2] 转换为 Phase 4 QAT (mode=lsqplus, noise=False)
[prepare_model_phase4] Converted 8 layers to QAT v2 (mode=lsqplus, noise=False, bias=False)
  QATConv2d_v2: 6, QATLinear_v2: 2

[Step 3] 训练 (60 epochs, lr=0.001)
  LSQ+ 模式: 可学习 scale/zero_point + 独立 lr (0.1x)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     2.0030   18.21% |    1.8096  25.52% |  25.52% |  197.2s
      5  |     1.4504   43.21% |    1.3215  48.87% |  48.87% |  195.1s
     10  |     1.2682   50.66% |    1.2491  52.35% |  52.46% |  207.8s
     15  |     1.2252   52.29% |    1.2236  53.69% |  55.35% |  203.2s
     20  |     1.1562   54.50% |    1.0818  58.17% |  58.26% |  376.6s
     25  |     1.1450   55.52% |    1.0483  59.22% |  59.67% |  183.6s
     30  |     1.1384   55.45% |    1.1088  57.74% |  59.67% |  180.0s
     35  |     1.1258   55.79% |    1.0460  59.94% |  60.43% |  181.8s
     40  |     1.0940   57.21% |    1.0218  60.87% |  60.91% |  182.0s
     45  |     1.0842   57.47% |    1.0111  61.19% |  61.19% |  188.4s
     50  |     1.0649   57.87% |    1.0043  61.72% |  61.72% |  196.1s
     55  |     1.0547   58.19% |    1.0200  61.43% |  61.72% |  206.7s
     60  |     1.0505   58.37% |    1.0148  61.28% |  61.72% |  244.6s

  模型已保存: baseline_vgg_phase4_lsqplus.pth

============================================================
  结果: Int4 (uint4/int4) = 61.72%
  FP32 基准: 97.17%, Phase 2 最佳: 91.17%
  模式: lsqplus, 噪声: False, 耗时: 198.4min
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python noise_robustness_v2.py --all
Device: cpu
============================================================
  Noise Robustness Testing v2 — int4 QAT Models
============================================================
验证集: 5400 张

============================================================
  测试: Model 1 (VGG int4)
============================================================
  权重加载: baseline_vgg_phase4_ste.pth
  基线准确率 (无噪声, int4): 98.06%

  ==================================================
  噪声: Weight Quantization (bits)
  模式: QAT int4
  ==================================================
    level=       8  |  Accuracy: 98.13%
    level=       6  |  Accuracy: 97.80%
    level=       5  |  Accuracy: 97.78%
    level=       4  |  Accuracy: 87.28%
    level=       3  |  Accuracy: 78.72%
    level=       2  |  Accuracy: 11.15%

  ==================================================
  噪声: Weight Gaussian Noise (σ)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 98.06%
    level=  0.0500  |  Accuracy: 98.00%
    level=  0.1000  |  Accuracy: 96.78%
    level=  0.2000  |  Accuracy: 88.37%
    level=  0.5000  |  Accuracy: 35.43%
    level=  1.0000  |  Accuracy: 24.59%

  ==================================================
  噪声: Activation Gaussian Noise (σ)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 98.06%
    level=  0.0200  |  Accuracy: 98.06%
    level=  0.0500  |  Accuracy: 98.06%
    level=  0.1000  |  Accuracy: 98.06%
    level=  0.2000  |  Accuracy: 98.06%
    level=  0.5000  |  Accuracy: 98.06%

  ==================================================
  噪声: Activation Quantization (bits)
  模式: QAT int4
  ==================================================
    level=       8  |  Accuracy: 98.06%
    level=       6  |  Accuracy: 98.06%
    level=       5  |  Accuracy: 98.06%
    level=       4  |  Accuracy: 98.06%
    level=       3  |  Accuracy: 98.06%
    level=       2  |  Accuracy: 98.06%

  ==================================================
  噪声: Weight Dropout (fraction)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 98.06%
    level=  0.0100  |  Accuracy: 97.35%
    level=  0.0500  |  Accuracy: 76.33%
    level=  0.1000  |  Accuracy: 48.43%
    level=  0.2000  |  Accuracy: 27.81%
    level=  0.5000  |  Accuracy: 11.15%

============================================================
  测试: Model 1 (VGG FP32 ref)
============================================================
  权重加载: baseline_vgg_phase4_ste.pth
  基线准确率 (无噪声, FP32): 98.06%

  ==================================================
  噪声: Weight Quantization (bits)
  模式: FP32
  ==================================================
    level=       8  |  Accuracy: 98.13%
    level=       6  |  Accuracy: 97.80%
    level=       5  |  Accuracy: 97.78%
    level=       4  |  Accuracy: 87.28%
    level=       3  |  Accuracy: 78.72%
    level=       2  |  Accuracy: 11.15%

  ==================================================
  噪声: Weight Gaussian Noise (σ)
  模式: FP32
  ==================================================
    level=  0.0000  |  Accuracy: 98.06%
    level=  0.0500  |  Accuracy: 97.80%
    level=  0.1000  |  Accuracy: 97.00%
    level=  0.2000  |  Accuracy: 76.85%
    level=  0.5000  |  Accuracy: 55.07%
    level=  1.0000  |  Accuracy: 16.80%

  ==================================================
  噪声: Activation Gaussian Noise (σ)
  模式: FP32
  ==================================================
    level=  0.0000  |  Accuracy: 98.06%
    level=  0.0200  |  Accuracy: 98.06%
    level=  0.0500  |  Accuracy: 98.06%
    level=  0.1000  |  Accuracy: 98.06%
    level=  0.2000  |  Accuracy: 98.06%
    level=  0.5000  |  Accuracy: 98.06%

  ==================================================
  噪声: Activation Quantization (bits)
  模式: FP32
  ==================================================
    level=       8  |  Accuracy: 98.06%
    level=       6  |  Accuracy: 98.06%
    level=       5  |  Accuracy: 98.06%
    level=       4  |  Accuracy: 98.06%
    level=       3  |  Accuracy: 98.06%
    level=       2  |  Accuracy: 98.06%

  ==================================================
  噪声: Weight Dropout (fraction)
  模式: FP32
  ==================================================
    level=  0.0000  |  Accuracy: 98.06%
    level=  0.0100  |  Accuracy: 97.26%
    level=  0.0500  |  Accuracy: 81.57%
    level=  0.1000  |  Accuracy: 42.56%
    level=  0.2000  |  Accuracy: 28.15%
    level=  0.5000  |  Accuracy: 11.15%

============================================================
  测试: Model 2 (SpaceNet V1 int4)
============================================================
  权重加载: spacenet_v1_phase4_ste.pth
  基线准确率 (无噪声, int4): 92.81%

  ==================================================
  噪声: Weight Quantization (bits)
  模式: QAT int4
  ==================================================
    level=       8  |  Accuracy: 92.61%
    level=       6  |  Accuracy: 90.85%
    level=       5  |  Accuracy: 86.69%
    level=       4  |  Accuracy: 63.52%
    level=       3  |  Accuracy: 21.85%
    level=       2  |  Accuracy: 7.83%

  ==================================================
  噪声: Weight Gaussian Noise (σ)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 92.81%
    level=  0.0500  |  Accuracy: 83.28%
    level=  0.1000  |  Accuracy: 66.76%
    level=  0.2000  |  Accuracy: 17.78%
    level=  0.5000  |  Accuracy: 17.50%
    level=  1.0000  |  Accuracy: 11.65%

  ==================================================
  噪声: Activation Gaussian Noise (σ)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 92.81%
    level=  0.0200  |  Accuracy: 92.33%
    level=  0.0500  |  Accuracy: 87.37%
    level=  0.1000  |  Accuracy: 76.67%
    level=  0.2000  |  Accuracy: 46.96%
    level=  0.5000  |  Accuracy: 27.52%

  ==================================================
  噪声: Activation Quantization (bits)
  模式: QAT int4
  ==================================================
    level=       8  |  Accuracy: 92.81%
    level=       6  |  Accuracy: 87.39%
    level=       5  |  Accuracy: 68.19%
    level=       4  |  Accuracy: 39.56%
    level=       3  |  Accuracy: 20.46%
    level=       2  |  Accuracy: 8.48%

  ==================================================
  噪声: Weight Dropout (fraction)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 92.81%
    level=  0.0100  |  Accuracy: 79.98%
    level=  0.0500  |  Accuracy: 21.91%
    level=  0.1000  |  Accuracy: 16.22%
    level=  0.2000  |  Accuracy: 22.61%
    level=  0.5000  |  Accuracy: 12.02%

============================================================
  测试: Model 3 (SpaceNet V2 KD int4)
============================================================
  权重加载: spacenet_v2_phase4_ste.pth
  基线准确率 (无噪声, int4): 93.15%

  ==================================================
  噪声: Weight Quantization (bits)
  模式: QAT int4
  ==================================================
    level=       8  |  Accuracy: 92.85%
    level=       6  |  Accuracy: 90.19%
    level=       5  |  Accuracy: 87.43%
    level=       4  |  Accuracy: 46.74%
    level=       3  |  Accuracy: 14.31%
    level=       2  |  Accuracy: 9.07%

  ==================================================
  噪声: Weight Gaussian Noise (σ)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 93.15%
    level=  0.0500  |  Accuracy: 88.96%
    level=  0.1000  |  Accuracy: 61.89%
    level=  0.2000  |  Accuracy: 58.31%
    level=  0.5000  |  Accuracy: 28.44%
    level=  1.0000  |  Accuracy: 7.78%

  ==================================================
  噪声: Activation Gaussian Noise (σ)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 93.15%
    level=  0.0200  |  Accuracy: 93.09%
    level=  0.0500  |  Accuracy: 91.96%
    level=  0.1000  |  Accuracy: 81.43%
    level=  0.2000  |  Accuracy: 50.02%
    level=  0.5000  |  Accuracy: 22.48%

  ==================================================
  噪声: Activation Quantization (bits)
  模式: QAT int4
  ==================================================
    level=       8  |  Accuracy: 93.00%
    level=       6  |  Accuracy: 90.67%
    level=       5  |  Accuracy: 74.96%
    level=       4  |  Accuracy: 41.63%
    level=       3  |  Accuracy: 20.94%
    level=       2  |  Accuracy: 9.02%

  ==================================================
  噪声: Weight Dropout (fraction)
  模式: QAT int4
  ==================================================
    level=  0.0000  |  Accuracy: 93.15%
    level=  0.0100  |  Accuracy: 91.65%
    level=  0.0500  |  Accuracy: 25.83%
    level=  0.1000  |  Accuracy: 28.80%
    level=  0.2000  |  Accuracy: 12.09%
    level=  0.5000  |  Accuracy: 11.13%



====================================================================================================
  NOISE ROBUSTNESS SUMMARY — int4 Optical Computing
====================================================================================================

  --- Weight Quantization (bits) ---
       Level | Model 1 (VGG int4)             | Model 1 (VGG FP32 ref)         | Model 2 (SpaceNet V1 int4)     | Model 3 (SpaceNet V2 KD int4)  |
  ------------------------------------------------------------------------------------------------------------------------------------------------
           8 |                        98.13% |                        98.13% |                        92.61% |                        92.85% |
           6 |                        97.80% |                        97.80% |                        90.85% |                        90.19% |
           5 |                        97.78% |                        97.78% |                        86.69% |                        87.43% |
           4 |                        87.28% |                        87.28% |                        63.52% |                        46.74% |
           3 |                        78.72% |                        78.72% |                        21.85% |                        14.31% |
           2 |                        11.15% |                        11.15% |                         7.83% |                         9.07% |

  --- Weight Gaussian Noise (σ) ---
       Level | Model 1 (VGG int4)             | Model 1 (VGG FP32 ref)         | Model 2 (SpaceNet V1 int4)     | Model 3 (SpaceNet V2 KD int4)  |
  ------------------------------------------------------------------------------------------------------------------------------------------------
      0.0000 |                        98.06% |                        98.06% |                        92.81% |                        93.15% |
      0.0500 |                        98.00% |                        97.80% |                        83.28% |                        88.96% |
      0.1000 |                        96.78% |                        97.00% |                        66.76% |                        61.89% |
      0.2000 |                        88.37% |                        76.85% |                        17.78% |                        58.31% |
      0.5000 |                        35.43% |                        55.07% |                        17.50% |                        28.44% |
      1.0000 |                        24.59% |                        16.80% |                        11.65% |                         7.78% |

  --- Activation Gaussian Noise (σ) ---
       Level | Model 1 (VGG int4)             | Model 1 (VGG FP32 ref)         | Model 2 (SpaceNet V1 int4)     | Model 3 (SpaceNet V2 KD int4)  |
  ------------------------------------------------------------------------------------------------------------------------------------------------
      0.0000 |                        98.06% |                        98.06% |                        92.81% |                        93.15% |
      0.0200 |                        98.06% |                        98.06% |                        92.33% |                        93.09% |
      0.0500 |                        98.06% |                        98.06% |                        87.37% |                        91.96% |
      0.1000 |                        98.06% |                        98.06% |                        76.67% |                        81.43% |
      0.2000 |                        98.06% |                        98.06% |                        46.96% |                        50.02% |
      0.5000 |                        98.06% |                        98.06% |                        27.52% |                        22.48% |

  --- Activation Quantization (bits) ---
       Level | Model 1 (VGG int4)             | Model 1 (VGG FP32 ref)         | Model 2 (SpaceNet V1 int4)     | Model 3 (SpaceNet V2 KD int4)  |
  ------------------------------------------------------------------------------------------------------------------------------------------------
           8 |                        98.06% |                        98.06% |                        92.81% |                        93.00% |
           6 |                        98.06% |                        98.06% |                        87.39% |                        90.67% |
           5 |                        98.06% |                        98.06% |                        68.19% |                        74.96% |
           4 |                        98.06% |                        98.06% |                        39.56% |                        41.63% |
           3 |                        98.06% |                        98.06% |                        20.46% |                        20.94% |
           2 |                        98.06% |                        98.06% |                         8.48% |                         9.02% |

  --- Weight Dropout (fraction) ---
       Level | Model 1 (VGG int4)             | Model 1 (VGG FP32 ref)         | Model 2 (SpaceNet V1 int4)     | Model 3 (SpaceNet V2 KD int4)  |
  ------------------------------------------------------------------------------------------------------------------------------------------------
      0.0000 |                        98.06% |                        98.06% |                        92.81% |                        93.15% |
      0.0100 |                        97.35% |                        97.26% |                        79.98% |                        91.65% |
      0.0500 |                        76.33% |                        81.57% |                        21.91% |                        25.83% |
      0.1000 |                        48.43% |                        42.56% |                        16.22% |                        28.80% |
      0.2000 |                        27.81% |                        28.15% |                        22.61% |                        12.09% |
      0.5000 |                        11.15% |                        11.15% |                        12.02% |                        11.13% |

  --- Noise Tolerance (accuracy >= 80%) ---
  Noise Type                     | Model 1 (VGG int4)             | Model 1 (VGG FP32 ref)         | Model 2 (SpaceNet V1 int4)     | Model 3 (SpaceNet V2 KD int4)  |
  ----------------------------------------------------------------------------------------------------
  Weight Quantization (bits)     | < 3                            | < 3                            | < 4                            | < 4                            |
  Weight Gaussian Noise (σ)      | < 0.5                          | < 0.2                          | < 0.1                          | < 0.1                          |
  Activation Gaussian Noise (σ)  | >max                           | >max                           | < 0.1                          | < 0.2                          |
  Activation Quantization (bits) | >max                           | >max                           | < 5                            | < 5                            |
  Weight Dropout (fraction)      | < 0.05                         | < 0.1                          | < 0.01                         | < 0.05                         |
====================================================================================================

  图表已保存: noise_robustness_v2.png

============================================================
  噪声鲁棒性测试完成!
============================================================
```


```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_mixed.py
设备: cpu, 模式: ste, act_bits: 8

============================================================
  Model 1 Mixed: Conv=int4 (光计算) + Linear=fp32 (电计算)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 2,387,946
  6×Conv → int4 QAT, 2×Linear → fp32

[Step 1] 转换为混合精度 QAT: Conv=int4, Linear=fp32
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 6 (6 enabled)  ← 光计算 int4
  nn.Linear:    2  ← 电计算 fp32 (不量化)
  BN (float32): 6, mode=ste, w4/a8
  训练噪声: std=0.02*scale (仅 int4 Conv 权重)
  int4 QAT Conv: 6, fp32 Linear: 2, BN(float32): 6

  [BaselineVGG Mixed (Conv=int4, Linear=fp32, ste)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 QAT  ] conv1_1                      3   3×3        27        32   84.4%
  [QATConv2d_v3 QAT  ] conv1_2                     32   3×3       288       288  100.0%
  [QATConv2d_v3 QAT  ] conv2_1                     32   3×3       288       288  100.0%
  [QATConv2d_v3 QAT  ] conv2_2                     64   3×3       576       576  100.0%
  [QATConv2d_v3 QAT  ] conv3_1                     64   3×3       576       576  100.0%
  [QATConv2d_v3 QAT  ] conv3_2                    128   3×3      1152      1152  100.0%
  [Linear       FP32 ] fc1                         —     —          8192      8192  100.0%
  [Linear       FP32 ] fc2                         —     —           256       256  100.0%
  综合硬件对齐率: 100.0% (展平总长度 11355 → 补零后 11360)

[Step 2] 训练 (80 epochs, lr=0.001, warmup=5, wd=0.0005)
  噪声注入: std=0.02*scale (仅 int4 Conv 权重)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     0.8564   70.11% |    0.6917  75.33% |  75.33% | 0.00020 |  271.5s
      5  |     0.3418   88.68% |    0.3999  86.39% |  88.89% | 0.00100 |  203.4s
     10  |     0.2018   93.09% |    0.1534  94.70% |  94.70% | 0.00099 |  176.7s
     15  |     0.1491   94.83% |    0.1244  95.72% |  95.72% | 0.00096 |  178.2s
     20  |     0.1068   96.30% |    0.1357  95.19% |  96.17% | 0.00091 |  145.6s
     25  |     0.0868   96.98% |    0.1328  95.59% |  96.17% | 0.00084 |  144.6s
     30  |     0.0672   97.62% |    0.1088  96.65% |  96.65% | 0.00075 |  141.7s
     35  |     0.0542   98.19% |    0.0895  97.33% |  97.33% | 0.00066 |  141.1s
     40  |     0.0387   98.59% |    0.1058  97.06% |  97.33% | 0.00056 |  203.4s
     45  |     0.0286   99.00% |    0.0922  97.24% |  97.74% | 0.00045 |  171.1s
     50  |     0.0236   99.20% |    0.1018  97.15% |  97.74% | 0.00035 |  139.2s
     55  |     0.0167   99.45% |    0.0852  97.67% |  97.80% | 0.00026 |  139.6s
     60  |     0.0132   99.57% |    0.0881  97.98% |  97.98% | 0.00017 |  178.8s
     65  |     0.0091   99.69% |    0.0828  98.09% |  98.09% | 0.00010 |  162.8s
     70  |     0.0079   99.76% |    0.0818  98.13% |  98.13% | 0.00005 |  183.8s
     75  |     0.0052   99.88% |    0.0828  98.11% |  98.19% | 0.00002 |  172.5s
     80  |     0.0061   99.83% |    0.0875  98.02% |  98.26% | 0.00001 |  184.9s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (Conv=int4 光计算) 准确率: 98.26%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:                97.91%
  Int4 量化损失:                     -0.35%

  模型已保存: baseline_vgg_mixed_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  策略:              Conv=int4 (光计算) + Linear=fp32 (电计算)
  模型:              BaselineVGG Mixed (Conv=int4, Linear=fp32, ste)
  参数量:            2,387,946
  训练总耗时:        13374.8s (222.9min)
  硬件对齐率:        100.0%
  Int4 最佳准确率:   98.26%
  Float32 准确率:    97.91%
  FP32 基准 (参考):  97.17% (全 fp32)
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_mixed.py
设备: cpu, 模式: ste, act_bits: 8

============================================================
  Model 2 Mixed: Conv=int4 (光计算) + Linear=fp32 (电计算)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 268,210
  4×Conv → int4 QAT, 2×Linear → fp32

[Step 1] 转换为混合精度 QAT: Conv=int4, Linear=fp32
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  nn.Linear:    2  ← 电计算 fp32 (不量化)
  BN (float32): 4, mode=ste, w4/a8
  训练噪声: std=0.02*scale (仅 int4 Conv 权重)
  int4 QAT Conv: 4, fp32 Linear: 2, BN(float32): 4

  [OpticSpaceNetV1 Mixed (Conv=int4, Linear=fp32, ste)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 QAT  ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 QAT  ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 QAT  ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 QAT  ] stage3.0                    32   1×1        32        32  100.0%
  [Linear       FP32 ] classifier.1                —     —          1024      1024  100.0%
  [Linear       FP32 ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (100 epochs, lr=0.001, warmup=5, wd=0.0005)
  噪声注入: std=0.02*scale (仅 int4 Conv 权重)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.0595   62.09% |    0.8087  70.24% |  70.24% | 0.00020 |   59.9s
      5  |     0.7177   74.38% |    0.6457  76.46% |  78.44% | 0.00100 |   51.8s
     10  |     0.6014   78.67% |    0.4921  82.83% |  82.83% | 0.00099 |   49.3s
     15  |     0.5261   81.33% |    0.4721  83.89% |  83.91% | 0.00097 |   48.8s
     20  |     0.4927   82.84% |    0.3930  86.74% |  86.74% | 0.00094 |   51.0s
     25  |     0.4502   84.32% |    0.4177  85.24% |  86.80% | 0.00090 |   51.3s
     30  |     0.4103   85.44% |    0.4369  85.17% |  87.17% | 0.00084 |   45.8s
     35  |     0.3788   86.95% |    0.3322  88.54% |  88.54% | 0.00078 |   46.3s
     40  |     0.3676   87.20% |    0.3338  88.48% |  88.54% | 0.00070 |   45.9s
     45  |     0.3496   87.79% |    0.4575  85.63% |  88.61% | 0.00063 |   45.5s
     50  |     0.3231   88.87% |    0.3273  89.04% |  89.15% | 0.00055 |   46.4s
     55  |     0.3193   88.84% |    0.3917  86.81% |  89.54% | 0.00046 |   45.6s
     60  |     0.3079   89.14% |    0.2935  90.17% |  90.17% | 0.00038 |   46.3s
     65  |     0.2873   90.04% |    0.3346  89.04% |  90.17% | 0.00031 |   45.8s
     70  |     0.2791   90.50% |    0.3063  89.65% |  90.44% | 0.00023 |   61.9s
     75  |     0.2589   91.15% |    0.3413  89.30% |  90.44% | 0.00017 |   61.8s
     80  |     0.2625   90.98% |    0.2751  91.26% |  91.26% | 0.00011 |   55.8s
     85  |     0.2525   91.19% |    0.2963  90.07% |  91.26% | 0.00007 |   60.2s
     90  |     0.2452   91.32% |    0.2914  90.33% |  91.26% | 0.00004 |   58.3s
     95  |     0.2457   91.47% |    0.2906  90.81% |  91.26% | 0.00002 |   46.7s
    100  |     0.2456   91.38% |    0.2861  90.78% |  91.26% | 0.00001 |   47.9s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 4 layers
  Int4 模式 (Conv=int4 光计算) 准确率: 91.26%
[disable_qat] Disabled QAT on 4 layers
  Float32 模式准确率:                84.33%
  Int4 量化损失:                     -6.93%

  模型已保存: spacenet_v1_mixed_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  策略:              Conv=int4 (光计算) + Linear=fp32 (电计算)
  模型:              OpticSpaceNetV1 Mixed (Conv=int4, Linear=fp32, ste)
  参数量:            268,210
  训练总耗时:        5099.3s (85.0min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   91.26%
  Float32 准确率:    84.33%
  FP32 基准 (参考):  90.15% (全 fp32)
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model3_spacenet_v2_mixed.py
设备: cpu, 模式: ste, act_bits: 8

============================================================
  Model 3 Mixed: KD + Conv=int4 (光计算) + Linear=fp32 (电计算)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 0] 加载教师 (ResNet-18)
  教师权重加载成功
  学生参数量: 268,210
  4×Conv → int4 QAT, 2×Linear → fp32

[Step 1] 转换学生: Conv=int4 QAT, Linear=fp32
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  nn.Linear:    2  ← 电计算 fp32 (不量化)
  BN (float32): 4, mode=ste, w4/a8
  训练噪声: std=0.02*scale (仅 int4 Conv 权重)
  int4 QAT Conv: 4, fp32 Linear: 2

  [Student (Conv=int4, Linear=fp32)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 QAT  ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 QAT  ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 QAT  ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 QAT  ] stage3.0                    32   1×1        32        32  100.0%
  [Linear       FP32 ] classifier.1                —     —          1024      1024  100.0%
  [Linear       FP32 ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] KD+混合精度训练 (120 epochs)
  蒸馏: T=4.0, α=0.7
  教师: ResNet-18 (fp32), 学生: Conv=int4 + Linear=fp32
----------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     9.7918   60.66% |    1.0507  71.43% |  71.43% |   92.2s
      5  |     6.4177   75.44% |    0.7628  78.98% |  78.98% |   97.0s
     10  |     5.6756   79.38% |    0.7099  81.94% |  83.17% |   81.4s
     15  |     5.0517   82.18% |    0.6728  83.54% |  84.41% |   81.1s
     20  |     4.8296   83.26% |    0.6162  84.11% |  85.15% |  100.2s
     25  |     4.6776   83.92% |    0.7200  81.74% |  86.39% |   99.9s
     30  |     4.4001   85.43% |    0.5323  86.85% |  86.96% |   91.5s
     35  |     4.3171   86.12% |    0.5651  85.85% |  88.28% |   90.9s
     40  |     4.1186   87.07% |    0.4249  88.70% |  88.70% |   90.7s
     45  |     4.0164   87.31% |    0.4627  88.94% |  88.94% |   91.7s
     50  |     3.9391   87.66% |    0.5054  88.26% |  89.35% |   58.4s
     55  |     3.8871   87.82% |    0.4196  89.70% |  89.70% |   51.3s
     60  |     3.7531   88.53% |    0.4552  89.39% |  89.70% |   55.1s
     65  |     3.7161   88.79% |    0.4935  87.91% |  89.80% |   51.6s
     70  |     3.6649   88.89% |    0.4535  88.96% |  89.80% |   56.1s
     75  |     3.6248   89.19% |    0.4524  89.52% |  89.80% |   55.4s
     80  |     3.5504   89.37% |    0.4805  88.63% |  89.80% |   54.7s
     85  |     3.5857   89.42% |    0.4296  89.56% |  90.04% |   82.5s
     90  |     3.4645   90.00% |    0.3972  90.09% |  90.48% |   78.5s
     95  |     3.4345   89.85% |    0.3893  90.37% |  90.48% |   80.7s
    100  |     3.3808   90.38% |    0.4181  89.70% |  90.48% |   79.4s
    105  |     3.3921   90.16% |    0.3780  90.09% |  90.56% |   78.8s
    110  |     3.3810   90.30% |    0.3774  90.61% |  91.07% |   81.8s
    115  |     3.3570   90.24% |    0.3797  90.48% |  91.07% |   72.7s
    120  |     3.2768   90.57% |    0.3516  91.07% |  91.13% |   71.9s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 4 layers
  Int4 模式 (Conv=int4 光计算) 准确率: 91.13%
[disable_qat] Disabled QAT on 4 layers
  Float32 模式准确率:                86.33%

  模型已保存: spacenet_v2_mixed_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  策略:              Conv=int4 (光计算) + Linear=fp32 (电计算)
  教师:              ResNet-18 (fp32)
  学生参数量:        268,210
  蒸馏:              T=4.0, α=0.7
  训练总耗时:        9220.8s (153.7min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   91.13%
  Float32 准确率:    86.33%
  FP32 KD 基准:      91.44% (全 fp32 KD)
============================================================
```


---


```powershell
PS E:\LT-Simulator\train-test> docker start LT-Simulator-container
LT-Simulator-container
PS E:\LT-Simulator\train-test> docker exec -it -w /workspace LT-Simulator-container /bin/bash
(moca_llm) root@a39a38d1a33b:/workspace# cd share/
(moca_llm) root@a39a38d1a33b:/workspace/share# ls
LT-Simulator_docker_v1.4.6-CCIC.tar  docs  scratch  train-firstround  train-test
(moca_llm) root@a39a38d1a33b:/workspace/share# cd train-test
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# ls
EXPERIMENTS.md                   log_mixed.md                  model3_spacenet_v2_int4.py    spacenet_v1_mixed_ste.pth
EXPERIMENTS.pdf                  log_noise_robustness.md       model3_spacenet_v2_mixed.py   spacenet_v1_phase4_ste.pth
OPTIC_QAT_README.md              log_phase4_fixed.md           model3_spacenet_v2_phase4.py  spacenet_v1_qat.pth
PHASE4_DESIGN.md                 log_phase4_original.md        model3_spacenet_v2_qat.py     spacenet_v2_distilled.pth
__pycache__                      log_qat_finetune.md           noise_robustness.py           spacenet_v2_int4.pth
baseline_vgg.pth                 model1_baseline.py            noise_robustness_v2.png       spacenet_v2_mixed_ste.pth
baseline_vgg_int4.pth            model1_baseline_int4.py       noise_robustness_v2.py        spacenet_v2_phase4_ste.pth
baseline_vgg_mixed_ste.pth       model1_baseline_mixed.py      optic_inference.py            spacenet_v2_qat.pth
baseline_vgg_phase4_lsqplus.pth  model1_baseline_phase4.py     optic_inference_mixed.py      teacher_resnet18.pth
baseline_vgg_phase4_ste.pth      model1_baseline_qat.py        optic_inference_phase4.py     test.ipynb
baseline_vgg_qat.pth             model2_spacenet_v1.py         optic_layers.py               train_mixed_runner.py
data                             model2_spacenet_v1_int4.py    optic_qat.py                  train_phase4_runner.py
example_load_gazelle_model.py    model2_spacenet_v1_mixed.py   optic_qat_v2.py               初赛文档
log.md                           model2_spacenet_v1_phase4.py  optic_qat_v3.py               复赛-test.md
log_fp32_baseline.md             model2_spacenet_v1_qat.py     spacenet_v1.pth
log_int4_scratch.md              model3_spacenet_v2.py         spacenet_v1_int4.pth

(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_phase4.py
Device: cpu
============================================================
  Optic-SpaceNet Phase 4: Optical Inference Migration
  Mode: QAT (pseudo-quant)
  Batch=1
============================================================

--- Loading Data ---
Train: 21600 imgs, Val: 5400 imgs
Classes: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

============================================================
  Model 1 Phase4 STE (VGG+BN)  [QAT mode: v3]
  Architecture: flat+BN
============================================================

  [1/3] Creating standard model...
  Params: 2,387,680

  [2/3] Converting to QAT (v3)...
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 6 (6 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 6, mode=ste, w4/a8

  [3/3] Loading QAT weights...
  Weights loaded from: baseline_vgg_phase4_ste.pth

  --- Native float32 evaluation ---
[disable_qat] Disabled QAT on 8 layers
  [Model 1 Phase4 STE (VGG+BN) float32] 5400 batches — acc=98.06%
  Float32 Accuracy: 98.06%
  Float32 Loss:     0.0777
  Float32 Time:     87.61s

  --- int4 QAT (optical computing simulation) evaluation ---
[enable_qat] Enabled QAT on 8 layers
  [Model 1 Phase4 STE (VGG+BN) int4-QAT] 5400 batches — acc=96.44%
  Int4 QAT Accuracy: 96.44%
  Int4 QAT Loss:     0.1440
  Int4 QAT Time:     163.58s

============================================================
  Model 2 Phase4 STE (SpaceNet V1)  [QAT mode: v3]
  Architecture: seq+BN
============================================================

  [1/3] Creating standard model...
  Params: 267,944

  [2/3] Converting to QAT (v3)...
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 4, mode=ste, w4/a8

  [3/3] Loading QAT weights...
  Weights loaded from: spacenet_v1_phase4_ste.pth

  --- Native float32 evaluation ---
[disable_qat] Disabled QAT on 6 layers
  [Model 2 Phase4 STE (SpaceNet V1) float32] 5400 batches — acc=92.57%
  Float32 Accuracy: 92.57%
  Float32 Loss:     0.2270
  Float32 Time:     45.96s

  --- int4 QAT (optical computing simulation) evaluation ---
[enable_qat] Enabled QAT on 6 layers
  [Model 2 Phase4 STE (SpaceNet V1) int4-QAT] 5400 batches — acc=74.70%
  Int4 QAT Accuracy: 74.70%
  Int4 QAT Loss:     1.0150
  Int4 QAT Time:     22.49s

============================================================
  Model 3 Phase4 STE (KD+SpaceNet)  [QAT mode: v3]
  Architecture: seq+BN
============================================================

  [1/3] Creating standard model...
  Params: 267,944

  [2/3] Converting to QAT (v3)...
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 4, mode=ste, w4/a8

  [3/3] Loading QAT weights...
  Weights loaded from: spacenet_v2_phase4_ste.pth

  --- Native float32 evaluation ---
[disable_qat] Disabled QAT on 6 layers
  [Model 3 Phase4 STE (KD+SpaceNet) float32] 5400 batches — acc=92.94%
  Float32 Accuracy: 92.94%
  Float32 Loss:     0.2737
  Float32 Time:     70.19s

  --- int4 QAT (optical computing simulation) evaluation ---
[enable_qat] Enabled QAT on 6 layers
  [Model 3 Phase4 STE (KD+SpaceNet) int4-QAT] 5400 batches — acc=78.06%
  Int4 QAT Accuracy: 78.06%
  Int4 QAT Loss:     1.0345
  Int4 QAT Time:     62.78s

============================================================
  Model 1 Phase4 LSQ+ (VGG)  [QAT mode: v2]
  Architecture: seq
============================================================

  [1/3] Creating standard model...
  Params: 2,386,272

  [2/3] Converting to QAT (v2)...
[prepare_model_phase4] Converted 8 layers to QAT v2 (mode=lsqplus, noise=False, bias=False)

  [3/3] Loading QAT weights...
  Weights loaded from: baseline_vgg_phase4_lsqplus.pth

  --- Native float32 evaluation ---
  [Model 1 Phase4 LSQ+ (VGG) float32] 5400 batches — acc=19.98%
  Float32 Accuracy: 19.98%
  Float32 Loss:     1752.9693
  Float32 Time:     34.66s

  --- int4 QAT (optical computing simulation) evaluation ---
  [Model 1 Phase4 LSQ+ (VGG) int4-QAT] 5400 batches — acc=51.93%
  Int4 QAT Accuracy: 51.93%
  Int4 QAT Loss:     1.2304
  Int4 QAT Time:     161.65s



====================================================================================================
  OPTIC-SPACENET PHASE 4: Optical Computing Inference Report
====================================================================================================

  [QAT Mode] int4 pseudo-quantization (matches training)
  Model                            Params  FP32 Acc  Int4 Acc Quant Loss
  ------------------------------------------------------------------------
  Model 1 Phase4 STE (VGG+BN)    2,387,680   98.06%   96.44%    1.61%
  Model 2 Phase4 STE (SpaceNet V1)  267,944   92.57%   74.70%   17.87%
  Model 3 Phase4 STE (KD+SpaceNet)  267,944   92.94%   78.06%   14.89%
  Model 1 Phase4 LSQ+ (VGG)      2,388,596   19.98%   51.93%  -31.94%

  Reference (training logs):
    Model 1 STE:  98.07% int4  (baseline_vgg_phase4_ste.pth)
    Model 2 STE:  92.87% int4  (spacenet_v1_phase4_ste.pth)
    Model 3 STE:  93.22% int4  (spacenet_v2_phase4_ste.pth)
    Model 1 LSQ+: 61.72% int4  (baseline_vgg_phase4_lsqplus.pth)
====================================================================================================
```


```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4_v3.py
设备: cpu

============================================================
  Model 2 Phase4 v3: int8 权重 + Gazelle 硬件噪声
  首层 stem FP32 (对齐率 37.5%), 其余 Conv+Linear int8
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944

[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 0 enabled + 4 fp32 (first layer)
  QAT Linear: 2, BN: 4
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [SpaceNet V1 (v4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   1×1         3         8   37.5%  w8
  [QATConv2d_v4 FP32 ] stage1.0                     8   2×2        32        32  100.0%  w8
  [QATConv2d_v4 FP32 ] stage2.0                    16   2×2        64        64  100.0%  w8
  [QATConv2d_v4 FP32 ] stage3.0                    32   1×1        32        32  100.0%  w8
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (100 epochs, lr=0.001, wd=0.0005)
  int8 权重 (硬件原生精度)
  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.2163   64.76% |    0.9311  74.61% |  74.61% | 0.00020 |   37.6s
      5  |     0.8920   77.99% |    0.7418  83.24% |  83.24% | 0.00100 |   37.9s
     10  |     0.7653   83.26% |    0.6909  84.80% |  86.72% | 0.00099 |   38.1s
     15  |     0.7221   85.67% |    0.6219  88.06% |  88.06% | 0.00097 |   37.1s
     20  |     0.6750   87.13% |    0.5954  88.93% |  89.33% | 0.00094 |   37.2s
     25  |     0.6495   88.48% |    0.6086  88.02% |  89.33% | 0.00090 |   37.5s
     30  |     0.6251   89.12% |    0.5526  91.04% |  91.04% | 0.00084 |   37.9s
     35  |     0.6077   90.01% |    0.5499  90.80% |  91.04% | 0.00078 |   53.1s
     40  |     0.5976   90.27% |    0.5410  91.13% |  91.15% | 0.00070 |   53.7s
     45  |     0.5747   91.21% |    0.5449  90.98% |  91.50% | 0.00063 |   41.5s
     50  |     0.5683   91.59% |    0.5294  91.67% |  91.94% | 0.00055 |   53.0s
     55  |     0.5610   91.41% |    0.5223  91.89% |  92.07% | 0.00046 |   41.5s
     60  |     0.5529   91.84% |    0.5192  92.31% |  92.31% | 0.00038 |   40.6s
     65  |     0.5463   92.07% |    0.5378  91.46% |  92.83% | 0.00031 |   41.3s
     70  |     0.5301   92.93% |    0.5049  92.83% |  92.83% | 0.00023 |   41.3s
     75  |     0.5315   92.65% |    0.5084  92.59% |  92.83% | 0.00017 |   42.7s
     80  |     0.5232   92.79% |    0.5072  92.54% |  92.83% | 0.00011 |   49.5s
     85  |     0.5211   92.87% |    0.4974  93.02% |  93.02% | 0.00007 |   53.9s
     90  |     0.5148   92.94% |    0.4993  92.72% |  93.02% | 0.00004 |   48.7s
     95  |     0.5119   93.40% |    0.4980  92.91% |  93.02% | 0.00002 |   48.9s
    100  |     0.5136   93.31% |    0.4996  92.81% |  93.20% | 0.00001 |   49.7s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int8 模式 (光计算模拟) 准确率: 93.20%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              93.24%
  Int8 量化损失:             0.04%

  模型已保存: spacenet_v1_phase4_v3_int8.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              SpaceNet V1 (bias=False)
  参数量:            267,944
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  首层:              FP32 (对齐率 37.5%)
  训练总耗时:        4371.6s (72.9min)
  硬件对齐率:        99.6%
  Int8 最佳准确率: 93.20%
  Float32 准确率:    93.24%
  旧版 int4 参考:    74.35% (Phase4, Conv QAT 全关)
  FP32 基准:         90.15%
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4_v3.py --wbits 4
设备: cpu

============================================================
  Model 2 Phase4 v3: int4 权重 + Gazelle 硬件噪声
  首层 stem FP32 (对齐率 37.5%), 其余 Conv+Linear int4
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944

[Step 1] 转换为 QAT v4 (int4 权重, Gazelle 噪声)
[prepare_model_v4] Gazelle HW-aware QAT: wint4/a8
  QAT Conv: 0 enabled + 4 fp32 (first layer)
  QAT Linear: 2, BN: 4
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [SpaceNet V1 (v4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   1×1         3         8   37.5%  w4
  [QATConv2d_v4 FP32 ] stage1.0                     8   2×2        32        32  100.0%  w4
  [QATConv2d_v4 FP32 ] stage2.0                    16   2×2        64        64  100.0%  w4
  [QATConv2d_v4 FP32 ] stage3.0                    32   1×1        32        32  100.0%  w4
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (120 epochs, lr=0.001, wd=0.0001)
  int4 权重 (保守, 硬件有余量)
  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.0467   64.46% |    0.7227  73.91% |  73.91% | 0.00020 |   41.9s
      5  |     0.6671   77.62% |    0.5270  81.61% |  81.61% | 0.00100 |   38.5s
     10  |     0.5418   82.38% |    0.4760  82.93% |  85.22% | 0.00100 |   37.6s
     15  |     0.4731   84.78% |    0.3713  87.57% |  88.17% | 0.00098 |   37.2s
     20  |     0.4315   86.33% |    0.3457  88.59% |  88.59% | 0.00096 |   37.3s
     25  |     0.3933   87.38% |    0.3323  88.57% |  89.44% | 0.00093 |   37.6s
     30  |     0.3752   88.05% |    0.3085  90.09% |  90.31% | 0.00089 |   37.5s
     35  |     0.3522   88.53% |    0.3098  90.13% |  90.31% | 0.00084 |   53.2s
     40  |     0.3438   89.19% |    0.2885  90.24% |  90.46% | 0.00079 |   53.1s
     45  |     0.3118   89.96% |    0.2789  90.72% |  90.98% | 0.00073 |   40.9s
     50  |     0.2911   90.58% |    0.2824  91.02% |  91.28% | 0.00067 |   52.9s
     55  |     0.2882   90.57% |    0.2835  91.54% |  91.54% | 0.00061 |   41.3s
     60  |     0.2827   90.88% |    0.3089  90.48% |  91.54% | 0.00054 |   40.8s
     65  |     0.2666   91.29% |    0.2942  90.94% |  91.54% | 0.00047 |   40.7s
     70  |     0.2514   91.94% |    0.2762  91.59% |  91.91% | 0.00040 |   40.9s
     75  |     0.2496   91.94% |    0.2664  91.48% |  91.91% | 0.00034 |   42.6s
     80  |     0.2292   92.44% |    0.2631  91.54% |  91.91% | 0.00028 |   48.4s
     85  |     0.2277   92.49% |    0.2603  91.74% |  91.91% | 0.00022 |   54.8s
     90  |     0.2154   92.75% |    0.2511  92.26% |  92.26% | 0.00017 |   50.6s
     95  |     0.2081   93.22% |    0.2465  92.02% |  92.52% | 0.00012 |   48.4s
    100  |     0.2038   93.18% |    0.2496  92.11% |  92.52% | 0.00008 |   42.1s
    105  |     0.1944   93.40% |    0.2558  92.44% |  92.52% | 0.00005 |   40.6s
    110  |     0.1881   93.44% |    0.2466  92.06% |  92.56% | 0.00003 |   42.4s
    115  |     0.1864   93.50% |    0.2451  92.31% |  92.56% | 0.00001 |   52.7s
    120  |     0.1848   93.36% |    0.2452  92.41% |  92.56% | 0.00001 |   42.2s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 92.56%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              91.76%
  Int4 量化损失:             -0.80%

  模型已保存: spacenet_v1_phase4_v3_int4.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              SpaceNet V1 (bias=False)
  参数量:            267,944
  权重量化:          int4 (保守)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  首层:              FP32 (对齐率 37.5%)
  训练总耗时:        5277.6s (88.0min)
  硬件对齐率:        99.6%
  Int4 最佳准确率: 92.56%
  Float32 准确率:    91.76%
  旧版 int4 参考:    74.35% (Phase4, Conv QAT 全关)
  FP32 基准:         90.15%
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4_v2.py
设备: cpu

============================================================
  Model 2 Phase4 v2: Conv+Linear 全 int4, bias=False
  修复: Conv 层 QAT 全开 (vs 旧版全关)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944
  4×Conv + 2×Linear → 全 int4 QAT, bias=False

[Step 1] 转换为 QAT v3: Conv+Linear→int4 QAT
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 4, mode=ste, w4/a8
  训练噪声: std=0.02*scale (仅 int4 Conv 权重)
  int4 QAT Conv: 4, int4 QAT Linear: 2, fp32 Linear: -2, BN: 4 (float32)

  [OpticSpaceNetV1 Phase4 v2 (Conv+Linear int4, bias=False)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 QAT  ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 QAT  ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 QAT  ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 QAT  ] stage3.0                    32   1×1        32        32  100.0%
  [QATLinear_v3 QAT  ] classifier.1                —     —          1024      1024  100.0%
  [QATLinear_v3 QAT  ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (100 epochs, lr=0.001, warmup=5, wd=0.0005)
  噪声注入: std=0.02*scale (仅 int4 权重层)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.0870   61.55% |    1.1137  61.20% |  61.20% | 0.00020 |   49.7s
      5  |     0.7283   74.75% |    0.6122  78.70% |  78.70% | 0.00100 |   40.2s
     10  |     0.6117   78.89% |    0.4718  84.15% |  84.15% | 0.00099 |   48.3s
     15  |     0.5356   81.69% |    0.4447  85.17% |  85.50% | 0.00097 |   31.3s
     20  |     0.5223   82.16% |    0.3735  87.61% |  87.61% | 0.00094 |   30.9s
     25  |     0.4693   84.00% |    0.4293  85.96% |  87.61% | 0.00090 |   55.1s
     30  |     0.4416   84.86% |    0.4236  85.74% |  88.37% | 0.00084 |   57.3s
     35  |     0.4188   85.86% |    1.6420  66.22% |  88.37% | 0.00078 |   52.0s
     40  |     0.3801   87.14% |    0.3812  88.28% |  88.37% | 0.00070 |   46.5s
     45  |     0.3778   87.17% |    0.3372  88.98% |  88.98% | 0.00063 |   50.7s
     50  |     0.3631   87.75% |    0.3868  87.35% |  88.98% | 0.00055 |   48.1s
     55  |     0.3466   88.59% |    0.3798  87.06% |  89.87% | 0.00046 |   47.4s
     60  |     0.3355   88.63% |    0.3239  89.44% |  89.87% | 0.00038 |   47.9s
     65  |     0.3264   88.79% |    0.3382  89.04% |  89.87% | 0.00031 |   46.1s
     70  |     0.3137   89.35% |    0.3148  89.30% |  89.87% | 0.00023 |   46.3s
     75  |     0.3063   89.34% |    0.4277  85.81% |  91.06% | 0.00017 |   46.9s
     80  |     0.3049   89.30% |    0.3442  88.89% |  91.06% | 0.00011 |   46.2s
     85  |     0.2821   90.36% |    0.4508  86.50% |  91.06% | 0.00007 |   37.8s
     90  |     0.2780   90.35% |    0.2932  90.83% |  91.06% | 0.00004 |   38.3s
     95  |     0.2858   90.08% |    0.3354  89.22% |  91.06% | 0.00002 |   56.7s
    100  |     0.2778   90.23% |    0.3832  88.19% |  91.06% | 0.00001 |   59.6s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 91.06%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:         87.54%
  Int4 量化损失:              -3.52%

  模型已保存: spacenet_v1_phase4_v2_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              OpticSpaceNetV1 Phase4 v2 (Conv+Linear int4, bias=False)
  参数量:            267,944
  量化策略:          Conv+Linear→int4 (全光计算)
  模式:              ste, w4/a8
  训练总耗时:        4525.3s (75.4min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   91.06%
  Float32 准确率:    87.54%
  FP32 基准 (参考):  90.15% (全 fp32)
  量化损失:          -3.52%
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model3_spacenet_v2_phase4_v2.py
设备: cpu

============================================================
  Model 3 Phase4 v2: KD + Conv+Linear 全 int4, bias=False
  修复: Conv 层 QAT 全开 (vs 旧版全关)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 0] 加载教师 (ResNet-18)
  教师权重加载成功
  学生参数量: 267,944
  4×Conv + 2×Linear → 全 int4 QAT, bias=False

[Step 1] 转换学生: Conv+Linear→int4 QAT, bias=False
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 4, mode=ste, w4/a8
  训练噪声: std=0.02*scale (仅 int4 Conv 权重)
  int4 QAT Conv: 4, int4 QAT Linear: 2, BN: 4 (float32)

  [Student (Conv+Linear int4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 QAT  ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 QAT  ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 QAT  ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 QAT  ] stage3.0                    32   1×1        32        32  100.0%
  [QATLinear_v3 QAT  ] classifier.1                —     —          1024      1024  100.0%
  [QATLinear_v3 QAT  ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] KD+Phase4 训练 (120 epochs, T=4.0, α=0.7)
  教师: ResNet-18 (fp32) → 学生: Conv+Linear int4, bias=False
----------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     9.7578   60.52% |    1.0173  71.13% |  71.13% |   74.6s
      5  |     6.4964   75.44% |    0.8989  77.19% |  78.15% |   80.5s
     10  |     5.6260   79.51% |    0.8805  79.15% |  80.96% |   76.9s
     15  |     5.2089   81.81% |    0.6174  84.13% |  84.37% |   74.9s
     20  |     4.9051   83.14% |    0.7321  82.41% |  85.02% |   69.9s
     25  |     4.6488   84.62% |    0.7925  80.31% |  86.04% |   70.8s
     30  |     4.5828   85.17% |    0.5173  87.54% |  87.54% |   69.8s
     35  |     4.4184   85.72% |    0.5193  86.26% |  87.54% |   70.1s
     40  |     4.1948   86.73% |    0.7246  83.28% |  88.96% |   56.6s
     45  |     4.1281   87.17% |    0.4604  89.00% |  89.00% |   56.5s
     50  |     4.0942   87.22% |    0.4408  89.41% |  89.41% |   84.8s
     55  |     4.0388   87.35% |    0.4872  87.57% |  89.56% |   47.2s
     60  |     3.9656   87.94% |    0.5298  87.26% |  89.74% |   47.0s
     65  |     4.0267   87.44% |    0.4530  88.83% |  89.74% |   69.6s
     70  |     3.8250   88.31% |    0.4619  88.37% |  89.74% |   73.1s
     75  |     3.7697   88.44% |    0.4171  89.93% |  89.93% |   52.5s
     80  |     3.7482   88.50% |    0.4735  88.17% |  89.93% |   52.5s
     85  |     3.6860   89.02% |    0.3655  90.85% |  90.85% |   45.8s
     90  |     3.6179   89.32% |    0.3374  91.50% |  91.50% |   45.1s
     95  |     3.5806   89.60% |    0.3628  90.63% |  91.50% |   46.4s
    100  |     3.5662   89.26% |    0.4107  89.52% |  91.50% |   46.6s
    105  |     3.5571   89.41% |    0.4244  89.20% |  91.50% |   46.1s
    110  |     3.5806   89.49% |    0.3752  90.52% |  91.50% |   45.5s
    115  |     3.4829   89.91% |    0.6337  85.70% |  91.50% |   45.5s
    120  |     3.4862   89.78% |    0.6703  84.89% |  91.50% |   46.5s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (全光计算) 准确率: 91.50%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:          80.98%

  模型已保存: spacenet_v2_phase4_v2_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  学生模型:          OpticSpaceNet (Conv+Linear int4, bias=False)
  教师模型:          ResNet-18 (fp32)
  参数量:            267,944
  蒸馏:              T=4.0, α=0.7
  训练总耗时:        7203.7s (120.1min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   91.50%
  Float32 准确率:    80.98%
  量化损失:          -10.52%
  FP32 KD 基准:      91.44% (全 fp32 KD)
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4_v3.py
设备: cpu

============================================================
  Model 2 Phase4 v3: int8 权重 + Gazelle 硬件噪声
  首层 stem FP32 (对齐率 37.5%), 其余 Conv+Linear int8
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944

[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 3 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 4
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [SpaceNet V1 (v4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   1×1         3         8   37.5%  w8
  [QATConv2d_v4 QAT  ] stage1.0                     8   2×2        32        32  100.0%  w8
  [QATConv2d_v4 QAT  ] stage2.0                    16   2×2        64        64  100.0%  w8
  [QATConv2d_v4 QAT  ] stage3.0                    32   1×1        32        32  100.0%  w8
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (100 epochs, lr=0.001, wd=0.0005)
  int8 权重 (硬件原生精度)
  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.2230   64.34% |    0.9473  74.44% |  74.44% | 0.00020 |   46.0s
      5  |     0.8894   78.24% |    0.7620  82.30% |  82.30% | 0.00100 |   50.3s
     10  |     0.7808   83.02% |    0.6559  86.67% |  86.67% | 0.00099 |   50.3s
     15  |     0.7213   85.47% |    0.6109  88.52% |  88.52% | 0.00097 |   47.0s
     20  |     0.6747   87.57% |    0.6072  89.00% |  89.24% | 0.00094 |   48.7s
     25  |     0.6530   88.23% |    0.5778  89.85% |  89.85% | 0.00090 |   49.2s
     30  |     0.6362   88.86% |    0.5557  90.85% |  90.85% | 0.00084 |   47.4s
     35  |     0.6182   89.75% |    0.5540  91.07% |  91.07% | 0.00078 |   47.8s
     40  |     0.5890   90.56% |    0.5516  90.76% |  91.20% | 0.00070 |   47.8s
     45  |     0.5843   90.97% |    0.5349  92.04% |  92.04% | 0.00063 |   49.7s
     50  |     0.5672   91.30% |    0.5366  91.44% |  92.04% | 0.00055 |   49.0s
     55  |     0.5568   91.76% |    0.5292  92.09% |  92.19% | 0.00046 |   49.2s
     60  |     0.5487   92.19% |    0.5155  92.33% |  92.33% | 0.00038 |   52.4s
     65  |     0.5421   92.42% |    0.5170  92.33% |  92.46% | 0.00031 |   50.5s
     70  |     0.5328   92.59% |    0.5072  92.69% |  92.69% | 0.00023 |   48.9s
     75  |     0.5289   92.90% |    0.5171  92.50% |  92.76% | 0.00017 |   47.4s
     80  |     0.5196   92.92% |    0.5063  93.11% |  93.11% | 0.00011 |   48.0s
     85  |     0.5179   92.88% |    0.5043  92.98% |  93.11% | 0.00007 |   49.0s
     90  |     0.5163   93.35% |    0.5009  92.87% |  93.11% | 0.00004 |   48.9s
     95  |     0.5144   93.16% |    0.5050  92.67% |  93.11% | 0.00002 |   47.4s
    100  |     0.5125   93.07% |    0.4976  92.72% |  93.11% | 0.00001 |   47.5s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int8 模式 (光计算模拟) 准确率: 93.11%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              93.02%
  Int8 量化损失:             -0.09%

  模型已保存: spacenet_v1_phase4_v3_int8.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              SpaceNet V1 (bias=False)
  参数量:            267,944
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  首层:              FP32 (对齐率 37.5%)
  训练总耗时:        4866.3s (81.1min)
  硬件对齐率:        99.6%
  Int8 最佳准确率: 93.11%
  Float32 准确率:    93.02%
  旧版 int4 参考:    74.35% (Phase4, Conv QAT 全关)
  FP32 基准:         90.15%
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model3_spacenet_v2_phase4_v2.py
设备: cpu

============================================================
  Model 3 Phase4 v2: KD + Conv+Linear 全 int4, bias=False
  修复: Conv 层 QAT 全开 (vs 旧版全关)
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

[Step 0] 加载教师 (ResNet-18)
  教师权重加载成功
  学生参数量: 267,944
  4×Conv + 2×Linear → 全 int4 QAT, bias=False

[Step 1] 转换学生: Conv+Linear→int4 QAT, bias=False
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 4, mode=ste, w4/a8
  训练噪声: std=0.02*scale (仅 int4 Conv 权重)
  int4 QAT Conv: 4, int4 QAT Linear: 2, BN: 4 (float32)

  [Student (Conv+Linear int4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v3 QAT  ] stem.0                       3   1×1         3         8   37.5%
  [QATConv2d_v3 QAT  ] stage1.0                     8   2×2        32        32  100.0%
  [QATConv2d_v3 QAT  ] stage2.0                    16   2×2        64        64  100.0%
  [QATConv2d_v3 QAT  ] stage3.0                    32   1×1        32        32  100.0%
  [QATLinear_v3 QAT  ] classifier.1                —     —          1024      1024  100.0%
  [QATLinear_v3 QAT  ] classifier.4                —     —           256       256  100.0%
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] KD+Phase4 训练 (120 epochs, T=4.0, α=0.7)
  教师: ResNet-18 (fp32) → 学生: Conv+Linear int4, bias=False
----------------------------------------------------------------------
  Epoch |    KD Loss Train Acc |  Val Loss  Val Acc |     Best |    Time
  -----------------------------------------------------------------
      1  |     9.7578   60.52% |    1.0173  71.13% |  71.13% |   64.8s
      5  |     6.4964   75.44% |    0.8989  77.19% |  78.15% |   63.7s
     10  |     5.6260   79.51% |    0.8805  79.15% |  80.96% |   64.1s
     15  |     5.2089   81.81% |    0.6174  84.13% |  84.37% |   65.5s
     20  |     4.9051   83.14% |    0.7321  82.41% |  85.02% |   63.8s
     25  |     4.6488   84.62% |    0.7925  80.31% |  86.04% |   64.3s
     30  |     4.5828   85.17% |    0.5173  87.54% |  87.54% |   64.3s
     35  |     4.4184   85.72% |    0.5193  86.26% |  87.54% |   64.7s
     40  |     4.1948   86.73% |    0.7246  83.28% |  88.96% |   67.4s
     45  |     4.1281   87.17% |    0.4604  89.00% |  89.00% |   68.4s
     50  |     4.0942   87.22% |    0.4408  89.41% |  89.41% |   63.0s
     55  |     4.0388   87.35% |    0.4872  87.57% |  89.56% |   63.4s
     60  |     3.9656   87.94% |    0.5298  87.26% |  89.74% |   63.6s
     65  |     4.0267   87.44% |    0.4530  88.83% |  89.74% |   65.4s
     70  |     3.8250   88.31% |    0.4619  88.37% |  89.74% |   63.3s
     75  |     3.7697   88.44% |    0.4171  89.93% |  89.93% |   60.8s
     80  |     3.7482   88.50% |    0.4735  88.17% |  89.93% |   46.2s
     85  |     3.6860   89.02% |    0.3655  90.85% |  90.85% |   46.7s
     90  |     3.6179   89.32% |    0.3374  91.50% |  91.50% |   46.7s
     95  |     3.5806   89.60% |    0.3628  90.63% |  91.50% |   52.7s
    100  |     3.5662   89.26% |    0.4107  89.52% |  91.50% |   51.3s
    105  |     3.5571   89.41% |    0.4244  89.20% |  91.50% |   50.5s
    110  |     3.5806   89.49% |    0.3752  90.52% |  91.50% |   51.8s
    115  |     3.4829   89.91% |    0.6337  85.70% |  91.50% |   56.4s
    120  |     3.4862   89.78% |    0.6703  84.89% |  91.50% |   61.4s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (全光计算) 准确率: 91.50%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:          80.98%

  模型已保存: spacenet_v2_phase4_v2_ste.pth

============================================================
  训练完成 — 结果汇总
============================================================
  学生模型:          OpticSpaceNet (Conv+Linear int4, bias=False)
  教师模型:          ResNet-18 (fp32)
  参数量:            267,944
  蒸馏:              T=4.0, α=0.7
  训练总耗时:        7163.8s (119.4min)
  硬件对齐率:        99.6%
  Int4 最佳准确率:   91.50%
  Float32 准确率:    80.98%
  量化损失:          -10.52%
  FP32 KD 基准:      91.44% (全 fp32 KD)
============================================================
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_lsq.py
设备: cpu

============================================================
  Model 2 LSQ+ int8: 可学习 scale/zero_point + Gazelle 噪声
  修复: in_scale 真正参与前向, int8 激活, BN 保留
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944

[Step 1] 转换为 LSQ+ int8
[prepare_model_lsq] LSQ+ int8 QAT: w8/a8
  LSQ Conv: 3 enabled + 1 fp32 (first layer)
  LSQ Linear: 2, BN: 4
  LSQ+ learnable params: scale + zero_point per layer
  首层 Conv 保留 FP32

[Step 2] 训练 (100 epochs, LSQ warmup=10)
  前 10 epochs: STE fallback (稳定激活分布)
  后 90 epochs: LSQ+ (可学习 scale/zp, lr=0.1x)
[set_lsq_lr] Weight params: 14, LSQ params: 24 (lr=0.0001)
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.2294   62.92% |    0.9576  74.24% |  74.24% | 0.00020 |   46.0s  [STE]
      5  |     0.8811   77.31% |    0.7500  82.48% |  82.48% | 0.00100 |   57.5s  [STE]
     10  |     0.7864   81.87% |    0.6637  86.44% |  86.44% | 0.00099 |   58.0s  [STE]
  >>> Epoch 11: 切换到 LSQ+ (可学习 scale/zp) <<<
[set_lsq_lr] Weight params: 14, LSQ params: 24 (lr=0.0001)
     15  |     0.7655   81.96% |    0.7658  86.93% |  86.93% | 0.00100 |   63.7s  [LSQ+]
     20  |     0.6912   85.20% |    0.7510  87.67% |  87.67% | 0.00100 |   63.4s  [LSQ+]
     25  |     0.6493   86.90% |    0.7290  88.96% |  88.96% | 0.00100 |   53.4s  [LSQ+]
     30  |     0.6218   88.05% |    0.7069  90.09% |  90.09% | 0.00100 |   53.8s  [LSQ+]
     35  |     0.6103   88.37% |    0.6847  90.26% |  90.26% | 0.00100 |   60.5s  [LSQ+]
     40  |     0.6013   88.74% |    0.7135  89.78% |  90.35% | 0.00100 |   62.7s  [LSQ+]
     45  |     0.5795   89.75% |    0.7204  90.63% |  90.69% | 0.00100 |   56.5s  [LSQ+]
     50  |     0.5749   89.80% |    0.6813  91.22% |  91.22% | 0.00100 |   64.7s  [LSQ+]
     55  |     0.5677   90.17% |    0.7120  90.67% |  91.22% | 0.00100 |   53.2s  [LSQ+]
     60  |     0.5599   90.39% |    0.7060  91.17% |  91.89% | 0.00100 |   62.4s  [LSQ+]
     65  |     0.5519   90.87% |    0.7317  90.63% |  92.09% | 0.00100 |   57.8s  [LSQ+]
     70  |     0.5534   90.62% |    0.6895  91.78% |  92.15% | 0.00100 |   54.6s  [LSQ+]
     75  |     0.5466   90.93% |    0.7049  91.56% |  92.15% | 0.00100 |   47.8s  [LSQ+]
     80  |     0.5446   90.91% |    0.7220  91.24% |  92.15% | 0.00100 |   52.1s  [LSQ+]
     85  |     0.5438   91.02% |    0.7102  91.17% |  92.15% | 0.00100 |   45.5s  [LSQ+]
     90  |     0.5346   91.45% |    0.7034  91.87% |  92.15% | 0.00100 |   35.9s  [LSQ+]
     95  |     0.5355   91.32% |    0.6780  92.80% |  92.80% | 0.00100 |   36.3s  [LSQ+]
    100  |     0.5242   91.88% |    0.7002  92.02% |  92.80% | 0.00100 |   36.5s  [LSQ+]

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  LSQ+ int8 模式准确率: 92.67%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:   62.52%
  LSQ+ 量化损失:        -30.15%

  模型已保存: spacenet_v1_lsq_int8.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              SpaceNet V1 (LSQ+ int8, bias=False)
  参数量:            271,298
  LSQ+ 可学习层:     6
  训练策略:          STE warmup 10ep + LSQ+ 90ep
  训练总耗时:        5355.7s (89.3min)
  LSQ+ int8 最佳:    92.80%
  Float32:           62.52%
  STE int8 参考:     93.11% (Phase4 v3)
  FP32 基准:         90.15%
============================================================
```

---

```powershell
PS E:\LT-Simulator\train-test> docker start LT-Simulator-container
LT-Simulator-container
PS E:\LT-Simulator\train-test> docker exec -it -w /workspace LT-Simulator-container /bin/bash
(moca_llm) root@a39a38d1a33b:/workspace# cd share/train-test
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int8.py
Device: cpu
============================================================
  Optic-SpaceNet INT8: Optical Inference Migration
  Model:  SpaceNet V1 Phase4 v3 (INT8, Gazelle-optimized)
  Weight: spacenet_v1_phase4_v3_int8.pth
  Mode:   QAT (int8 pseudo-quant)
  Batch:  1, full
============================================================

--- Loading Data ---
Train: 21600 imgs, Val: 5400 imgs
Classes: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

============================================================
  Model 2 Phase4 v3 INT8  [QAT mode: int8]
  Architecture: SpaceNet V1 (seq+BN, bias=False)
============================================================

  [1/3] Creating standard model...
  Params: 267,944

  [2/3] Converting to QAT v4 (int8 weight + int8 act, stem FP32)...
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 3 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 4
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [3/3] Loading INT8 QAT weights...
  Weights loaded from: spacenet_v1_phase4_v3_int8.pth

  --- Native float32 evaluation (QAT disabled) ---
[disable_qat] Disabled QAT on 6 layers
  [Model 2 Phase4 v3 INT8 float32] 5400 batches — acc=93.02%
  Float32 Accuracy: 93.02%
  Float32 Loss:     0.2462
  Float32 Time:     5.77s

  --- int8 QAT (optical computing simulation) evaluation ---
[enable_qat] Enabled QAT on 6 layers
  [Model 2 Phase4 v3 INT8 int8-QAT] 5400 batches — acc=92.94%
  Int8 QAT Accuracy: 92.94%
  Int8 QAT Loss:     0.2455
  Int8 QAT Time:     59.07s
  Quantization Loss: +0.07% ([OK] tiny)



==============================================================================================================
  OPTIC-SPACENET INT8: Optical Computing Inference & MOPs Report
  Model 2 SpaceNet V1 Phase4 v3 — 当前最佳 INT8 模型
==============================================================================================================

  ------------------------------------------------------------
  [Accuracy] 精度评估 (QAT int8 伪量化)
  ------------------------------------------------------------
  模型:               Model 2 Phase4 v3 INT8
  参数量:             267,944
  Float32 准确率:     93.02%
  Int8 QAT 准确率:    92.94%
  量化损失:           +0.07%
  Float32 耗时:       5.8s
  Int8 QAT 耗时:      59.1s

  训练参考:
    训练 Int8 最佳:   93.11% (Phase4 v3, 100 epochs)
    训练 Float32:     93.02%
    FP32 基准:        90.15%
    旧版 int4 (bug):  74.35%


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
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int8.py
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

  [2/3] Converting to optical (OpticConv2d + OpticLinear, int8)...

  Model 2 Phase4 v3 INT8 (Optical) 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [OpticConv2d] stem.0                       3   1×1          3          8   37.5%
  [OpticConv2d] stage1.0                     8   2×2         32         32   100.0%
  [OpticConv2d] stage2.0                    16   2×2         64         64   100.0%
  [OpticConv2d] stage3.0                    32   1×1         32         32   100.0%
  [OpticLinear] classifier.1               —     —           1024       1024   100.0%
  [OpticLinear] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  [3/3] Evaluating via osimulator...
  [Model 2 Phase4 v3 INT8 optic] 5400 batches, report every 540 batch(es)
  [Model 2 Phase4 v3 INT8 optic]  540/5400 ( 10.0%) acc=84.63%  elapsed=1571s  ETA=14142s
  [Model 2 Phase4 v3 INT8 optic] 1080/5400 ( 20.0%) acc=85.37%  elapsed=3014s  ETA=12056s
  [Model 2 Phase4 v3 INT8 optic] 1620/5400 ( 30.0%) acc=84.57%  elapsed=4638s  ETA=10822s
  [Model 2 Phase4 v3 INT8 optic] 2160/5400 ( 40.0%) acc=84.54%  elapsed=6325s  ETA=9487s
  [Model 2 Phase4 v3 INT8 optic] 2700/5400 ( 50.0%) acc=84.70%  elapsed=7836s  ETA=7836s
  [Model 2 Phase4 v3 INT8 optic] 3240/5400 ( 60.0%) acc=84.54%  elapsed=9902s  ETA=6602s
  [Model 2 Phase4 v3 INT8 optic] 3780/5400 ( 70.0%) acc=84.68%  elapsed=12091s  ETA=5182s
  [Model 2 Phase4 v3 INT8 optic] 4320/5400 ( 80.0%) acc=84.68%  elapsed=13867s  ETA=3467s
  [Model 2 Phase4 v3 INT8 optic] 4860/5400 ( 90.0%) acc=84.51%  elapsed=15477s  ETA=1720s
  [Model 2 Phase4 v3 INT8 optic] 5400/5400 (100.0%) acc=84.43%  elapsed=16949s  ETA=0s
  [Model 2 Phase4 v3 INT8 optic] DONE — 5400 batches, acc=84.43%, total=16949s
  Optical Accuracy: 84.43%
  Optical Time:     16948.96s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 32400, 总耗时: 16765.639s, 总运算量: 6.56e+09 MACs



==============================================================================================================
  OPTIC-SPACENET INT8: Optical Computing Inference & MOPs Report
  Model 2 SpaceNet V1 Phase4 v3 — 当前最佳 INT8 模型
==============================================================================================================

  ------------------------------------------------------------
  [Accuracy] Optic osimulator Hardware Simulation (独立测试集)
  ------------------------------------------------------------
  模型:               Model 2 Phase4 v3 INT8
  光计算准确率:       84.43%
  osimulator 耗时:    16949.0s


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



==============================================================================================================
  OPTIC-SPACENET INT8: Optical Computing Inference & MOPs Report
  Model 2 SpaceNet V1 Phase4 v3 — 当前最佳 INT8 模型
==============================================================================================================

  ------------------------------------------------------------
  [Accuracy] Optic osimulator Hardware Simulation (独立测试集)
  ------------------------------------------------------------
  模型:               Model 2 Phase4 v3 INT8
  光计算准确率:       84.43%
  osimulator 耗时:    16949.0s


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
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int8.py
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



==============================================================================================================
  OPTIC-SPACENET INT8: Optical Computing Inference & MOPs Report
  Model 2 SpaceNet V1 Phase4 v3 — 当前最佳 INT8 模型
==============================================================================================================

  ------------------------------------------------------------
  [Accuracy] Optic osimulator Hardware Simulation (独立测试集)
  ------------------------------------------------------------
  模型:               Model 2 Phase4 v3 INT8
  光计算准确率:       93.28%
  osimulator 耗时:    14840.6s


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
```

如果需要现场演示的话跑部分，例如：
`python optic_inference_int8.py --quick 50`


```powershell
 [optic]  199/200 ( 99.5%) acc=83.42%  elapsed=522s  ETA=3s
    [osimulator] (1x1024x32) @ (32x16) input=uint8 ... done (0.6s)
    [osimulator] (1x64x64) @ (64x32) input=uint8 ... done (0.5s)
    [osimulator] (1x64x32) @ (32x16) input=uint8 ... done (0.1s)
    [osimulator] (1x1x1024) @ (1024x256) input=uint8 ... done (1.3s)
    [osimulator] (1x1x256) @ (256x10) input=uint8 ... done (0.0s)
  [optic]  200/200 (100.0%) acc=83.50%  elapsed=525s  ETA=0s
  [optic] DONE — 200 batches, acc=83.50%, total=525s
  Optical Accuracy: 83.50%  Time: 524.7s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 1000, 总耗时: 522.370s, 总运算量: 1.91e+08 MACs

====================================================================================================
  Model 3 KD+INT4 — Container Verification Report
====================================================================================================
  Optic osimulator: 83.50%  |  Time: 525s
  Training ref: 91.50% int4 (KD from ResNet-18 97.83%)

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
```


```powershell
 [optic]   19/20 ( 95.0%) acc=100.00%  elapsed=2781s  ETA=146s
    [osimulator] (1x4096x32) @ (32x32) input=uint8 ... done (3.9s)
    [osimulator] (1x4096x288) @ (288x32) input=uint8 ... done (42.0s)
    [osimulator] (1x1024x288) @ (288x64) input=uint8 ... done (14.8s)
    [osimulator] (1x1024x576) @ (576x64) input=uint8 ... done (38.6s)
    [osimulator] (1x256x576) @ (576x128) input=uint8 ... done (19.0s)
    [osimulator] (1x256x1152) @ (1152x128) input=uint8 ... done (32.5s)
  [optic]   20/20 (100.0%) acc=100.00%  elapsed=2932s  ETA=0s
  [optic] DONE — 20 batches, acc=100.00%, total=2932s
  Optical Accuracy: 100.00%  Time: 2931.7s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 120, 总耗时: 2931.145s, 总运算量: 3.10e+09 MACs

====================================================================================================
  Model 1 Mixed (VGG) — Container Verification Report
====================================================================================================
  Optic osimulator: 100.00%  |  Time: 2932s
  Training ref: 98.26% int4 Mixed

==============================================================================================================
  Mixed 模型光计算 MOPs 统计 — Model 1 Baseline VGG (Conv=int4 光, Linear=fp32 电)
  Gazelle 硬件: 8x2 tile, 4w8a for Conv, Linear 保留电计算
==============================================================================================================

  Layer            Type    C_in C_out Kernel      Input    ConvOut   Pool  Patch Padded   Align    RawMOPs    OptMOPs   ElecMOPs      Compute
  ------------------------------------------------------------------------------------------------------------------------
  conv1_1          Conv       3    32    3x3      64x64      64x64   None     27     32  84.4%    3.5389M    4.1943M    0.0000M [Optical]
  conv1_2          Conv      32    32    3x3      64x64      64x64 Max2x2    288    288 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv2_1          Conv      32    64    3x3      32x32      32x32   None    288    288 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv2_2          Conv      64    64    3x3      32x32      32x32 Max2x2    576    576 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv3_1          Conv      64   128    3x3      16x16      16x16   None    576    576 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv3_2          Conv     128   128    3x3      16x16      16x16 Max2x2   1152   1152 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  fc1              Linear  8192   256      -          -          -   None   8192   8192 100.0%    2.0972M    0.0000M    2.0972M [Electronic]
  fc2              Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0000M    0.0026M [Electronic]
  ------------------------------------------------------------------------------------------------------------------------
  Total                                                                                          156.6336M  155.1892M    2.0997M

  [MOPs] 光计算占比: 98.67%  |  总 MOPs: 156.63 M
  补零浪费: 0.6554 M (conv1_1 展平=27→32)
  [Note] Mixed 策略: 6 Conv 在 Gazelle 光计算 (int4), 2 Linear 在 CPU/GPU 电计算 (fp32)
```

```powershell
 [optic-LSQ]   49/50 ( 98.0%) acc=95.92%  elapsed=155s  ETA=3s
    [osimulator] (1x1024x32) @ (32x16) input=uint8 ... done (0.8s)
    [osimulator] (1x64x64) @ (64x32) input=uint8 ... done (0.6s)
    [osimulator] (1x64x32) @ (32x16) input=uint8 ... done (0.1s)
    [osimulator] (1x1x1024) @ (1024x256) input=uint8 ... done (1.5s)
    [osimulator] (1x1x256) @ (256x10) input=uint8 ... done (0.1s)
  [optic-LSQ]   50/50 (100.0%) acc=96.00%  elapsed=159s  ETA=0s
  [optic-LSQ] DONE — 50 batches, acc=96.00%, total=159s
  Optical Accuracy: 96.00%  Time: 158.6s
  [Note] LSQ uses fake engine (float matmul) — per-channel scales preserved

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 250, 总耗时: 122.487s, 总运算量: 4.76e+07 MACs

====================================================================================================
  Model 2 LSQ+ INT8 — Container Verification Report
====================================================================================================
  Optic osimulator: 96.00%  |  Time: 159s
  Training ref: 92.80% int8 LSQ+

==============================================================================================================
  LSQ+ INT8 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 LSQ+
  Gazelle 硬件: 8x2 tile, 8a8w, stem 电计算, 其余光计算
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

  [MOPs] 光计算占比: 90.65%  |  总 MOPs: 1.0511 M
  [Note] LSQ+ 模型优势: scale/zp 可直接导出为硬件配置, 无需软件量化
```

```powershell
 [optic]  199/200 ( 99.5%) acc=83.42%  elapsed=619s  ETA=3s
    [osimulator] (1x1024x32) @ (32x16) input=uint8 ... done (0.8s)
    [osimulator] (1x64x64) @ (64x32) input=uint8 ... done (0.7s)
    [osimulator] (1x64x32) @ (32x16) input=uint8 ... done (0.1s)
    [osimulator] (1x1x1024) @ (1024x256) input=uint8 ... done (1.5s)
    [osimulator] (1x1x256) @ (256x10) input=uint8 ... done (0.0s)
  [optic]  200/200 (100.0%) acc=83.50%  elapsed=622s  ETA=0s
  [optic] DONE — 200 batches, acc=83.50%, total=622s
  Optical Accuracy: 83.50%  Time: 622.3s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 1000, 总耗时: 582.542s, 总运算量: 1.91e+08 MACs

====================================================================================================
  Model 3 KD+INT4 — Container Verification Report
====================================================================================================
  Optic osimulator: 83.50%  |  Time: 622s
  Training ref: 91.50% int4 (KD from ResNet-18 97.83%)

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
```


---
## 7-11
```powershell
PS E:\LT-Simulator\train-test> docker exec -it -w /workspace LT-Simulator-container /bin/bash
(moca_llm) root@a39a38d1a33b:/workspace# cd share/train-test
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int4.py
Device: cpu
============================================================
  Optic-SpaceNet INT4: In-Container Optical Inference
  Model 2 Phase4 v2 (int4, 91.06%)  |  Weight: spacenet_v1_phase4_v2_ste.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  Model 2 Phase4 v2 INT4  [Optic mode: osimulator]
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
  [Note] osimulator uses native 8a8w — QAT int4 weights quantized to int8 (lossless)

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
  [optic]  540/5400 ( 10.0%) acc=85.93%  elapsed=1383s  ETA=12448s
  [optic] 1080/5400 ( 20.0%) acc=87.13%  elapsed=2754s  ETA=11016s
  [optic] 1620/5400 ( 30.0%) acc=87.59%  elapsed=4114s  ETA=9599s
  [optic] 2160/5400 ( 40.0%) acc=87.87%  elapsed=5591s  ETA=8387s
  [optic] 2700/5400 ( 50.0%) acc=88.41%  elapsed=7762s  ETA=7762s
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int4.py --qat
Device: cpu
============================================================
  Optic-SpaceNet INT4: In-Container Optical Inference
  Model 2 Phase4 v2 (int4, 91.06%)  |  Weight: spacenet_v1_phase4_v2_ste.pth
  Mode: QAT
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0

============================================================
  Model 2 Phase4 v2 INT4  [QAT mode: int4]
============================================================

  [1/3] Creating model...
  Params: 267,944

  [2/3] Converting to QAT v3 (int4 weight, int8 act)...
[prepare_model_v3] 量化策略: Conv=int4 QAT, Linear=fp32 (电计算)
  QATConv2d_v3: 4 (4 enabled)  ← 光计算 int4
  QATLinear_v3: 2  ← 光计算 int4
  BN (float32): 4, mode=ste, w4/a8

  [3/3] Loading weights: spacenet_v1_phase4_v2_ste.pth
[disable_qat] Disabled QAT on 6 layers
  [fp32] 5400 batches — acc=91.43%
  Float32: 91.43% (93.8s)
[enable_qat] Enabled QAT on 6 layers
  [int4-QAT] 5400 batches — acc=94.57%
  Int4 QAT: 94.57% (68.8s)
  Quant Loss: -3.15%

====================================================================================================
  Model 2 Phase4 v2 INT4 — Container Verification Report
====================================================================================================
  QAT float32: 91.43%  |  QAT int4: 94.57%  |  Quant Loss: -3.15%
  Training ref: 91.06% int4

==============================================================================================================
  INT4 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v2
  Gazelle 硬件: 8x2 tile, act=int8, weight=int4, stem 电计算
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
  光计算占比:          90.65%
  总 MOPs:             1.0511 M
  [Note] stem 展平=3 对齐率仅 37.5%, int4 下噪声过大, 保留电计算
```

```powershell
[optic]   49/50 ( 98.0%) acc=87.76%  elapsed=145s  ETA=3s
    [osimulator-LSQ] (1x1024x32) @ (32x16) ... done (0.8s)
    [osimulator-LSQ] (1x64x64) @ (64x32) ... done (0.7s)
    [osimulator-LSQ] (1x64x32) @ (32x16) ... done (0.1s)
    [osimulator-LSQ] (1x1x1024) @ (1024x256) ... done (1.5s)
    [osimulator-LSQ] (1x1x256) @ (256x10) ... done (0.0s)
  [optic]   50/50 (100.0%) acc=88.00%  elapsed=148s  ETA=0s
  [optic] DONE — 50 batches, acc=88.00%, total=148s
  Optical Accuracy: 88.00%  Time: 148.5s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 250, 总耗时: 156.985s, 总运算量: 4.76e+07 MACs

====================================================================================================
  Model 2 Phase4 v2 INT4 — Container Verification Report
====================================================================================================
  Optic osimulator: 88.00%  |  Time: 148s
  Training ref: 91.06% int4

==============================================================================================================
  INT4 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v2
  Gazelle 硬件: 8x2 tile, act=int8, weight=int4, stem 电计算
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
  光计算占比:          90.65%
  总 MOPs:             1.0511 M
  [Note] stem 展平=3 对齐率仅 37.5%, int4 下噪声过大, 保留电计算
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4_v3.py --wbits 4
设备: cpu

============================================================
  Model 2 Phase4 v3: int4 权重 + Gazelle 硬件噪声
  首层 stem FP32 (对齐率 37.5%), 其余 Conv+Linear int4
============================================================
训练: 21600, 验证: 5400
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944

[Step 1] 转换为 QAT v4 (int4 权重, Gazelle 噪声)
[prepare_model_v4] Gazelle HW-aware QAT: wint4/a8
  QAT Conv: 3 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 4
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [SpaceNet V1 (v4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   1×1         3         8   37.5%  w4
  [QATConv2d_v4 QAT  ] stage1.0                     8   2×2        32        32  100.0%  w4
  [QATConv2d_v4 QAT  ] stage2.0                    16   2×2        64        64  100.0%  w4
  [QATConv2d_v4 QAT  ] stage3.0                    32   1×1        32        32  100.0%  w4
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (120 epochs, lr=0.001, wd=0.0001)
  int4 权重 (保守, 硬件有余量)
  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.0642   63.82% |    0.7805  72.02% |  72.02% | 0.00020 |  239.4s
      5  |     0.6880   76.99% |    0.5490  81.00% |  81.00% | 0.00100 |   37.6s
     10  |     0.5821   81.37% |    0.4375  84.96% |  84.96% | 0.00100 |   36.9s
     15  |     0.5191   83.42% |    0.3843  86.48% |  86.48% | 0.00098 |   37.6s
     20  |     0.4748   85.01% |    0.3977  86.63% |  87.37% | 0.00096 |   38.0s
     25  |     0.4474   85.93% |    0.3573  88.06% |  88.41% | 0.00093 |   38.1s
     30  |     0.4059   87.00% |    0.4097  86.30% |  88.52% | 0.00089 |   38.4s
     35  |     0.3892   87.68% |    0.3338  89.20% |  89.20% | 0.00084 |   36.9s
     40  |     0.3842   88.08% |    0.3321  88.70% |  89.72% | 0.00079 |   39.8s
     45  |     0.3532   88.50% |    0.3140  90.00% |  90.00% | 0.00073 |   40.9s
     50  |     0.3456   89.00% |    0.3742  87.98% |  90.00% | 0.00067 |   40.1s
     55  |     0.3309   89.22% |    0.4420  85.37% |  90.48% | 0.00061 |   41.1s
     60  |     0.3184   89.73% |    0.3509  88.93% |  90.63% | 0.00054 |   40.0s
     65  |     0.3184   89.79% |    0.3628  87.89% |  90.63% | 0.00047 |   53.0s
     70  |     0.2959   90.42% |    0.3763  88.37% |  90.63% | 0.00040 |   51.9s
     75  |     0.2939   90.15% |    0.2893  90.87% |  90.87% | 0.00034 |   40.6s
     80  |     0.2882   90.78% |    0.2977  90.54% |  90.94% | 0.00028 |   47.6s
     85  |     0.2746   90.75% |    0.3280  89.89% |  90.94% | 0.00022 |   44.3s
     90  |     0.2686   91.06% |    0.2988  90.91% |  91.19% | 0.00017 |   41.2s
     95  |     0.2565   91.41% |    0.2777  91.07% |  91.19% | 0.00012 |   41.3s
    100  |     0.2544   91.60% |    0.2785  91.19% |  91.39% | 0.00008 |   41.2s
    105  |     0.2507   91.66% |    0.2628  91.46% |  91.52% | 0.00005 |   41.7s
    110  |     0.2465   91.75% |    0.2777  91.33% |  91.94% | 0.00003 |   41.3s
    115  |     0.2314   92.10% |    0.2660  91.67% |  91.94% | 0.00001 |   42.0s
    120  |     0.2381   92.04% |    0.2744  91.20% |  91.94% | 0.00001 |   42.8s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int4 模式 (光计算模拟) 准确率: 91.94%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              82.80%
  Int4 量化损失:             -9.15%

  模型已保存: spacenet_v1_phase4_v3_int4.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              SpaceNet V1 (bias=False)
  参数量:            267,944
  权重量化:          int4 (保守)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  首层:              FP32 (对齐率 37.5%)
  训练总耗时:        5179.6s (86.3min)
  硬件对齐率:        99.6%
  Int4 最佳准确率: 91.94%
  Float32 准确率:    82.80%
  旧版 int4 参考:    74.35% (Phase4, Conv QAT 全关)
  FP32 基准:         90.15%
============================================================
```


```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int4_v2.py --qat --quick 200
Device: cpu
============================================================
  Optic-SpaceNet INT4 v4: osimulator-Compatible Inference
  Model 2 Phase4 v4 (int4, stem=FP32)  |  Weight: spacenet_v1_phase4_v3_int4.pth
  Mode: QAT
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0

============================================================
  Model 2 Phase4 v4 INT4  [QAT mode: int4]
============================================================

  [1/3] Creating model...
  Params: 267,944

  [2/3] Converting to QAT v4 (int4 weight, int8 act, stem FP32)...
[prepare_model_v4] Gazelle HW-aware QAT: wint4/a8
  QAT Conv: 3 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 4
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [3/3] Loading weights: spacenet_v1_phase4_v3_int4.pth
[disable_qat] Disabled QAT on 6 layers
  [fp32] 200 batches — acc=89.00%
  Float32: 89.00% (2.1s)
[enable_qat] Enabled QAT on 6 layers
  [int4-QAT] 200 batches — acc=97.00%
  Int4 QAT: 97.00% (2.7s)
  Quant Loss: -8.00%

====================================================================================================
  Model 2 Phase4 v4 INT4 — Container Verification Report
====================================================================================================
  QAT float32: 89.00%  |  QAT int4: 97.00%  |  Quant Loss: -8.00%
  Training ref: TBD (train with: model2_spacenet_v1_phase4_v3.py --wbits 4)
  Expected: ~90-92% optical accuracy (stem FP32 matching inference)

==============================================================================================================
  INT4 v4 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v4
  Gazelle 硬件: 8x2 tile, act=int8, weight=int8, stem 电计算
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
  光计算占比:          90.65%
  总 MOPs:             1.0511 M
  [Note] stem FP32 电计算 (训练时 first_conv_fp32=True, 推理一致)
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_lsq.py
Device: cpu
============================================================
  Optic-SpaceNet LSQ+: In-Container Optical Inference
  Model 2 LSQ+ (int8, 92.80%)  |  Weight: spacenet_v1_lsq_int8.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  Model 2 LSQ+ INT8  [Optic mode: LSQ quant → real osimulator]
============================================================

  [1/3] Loading LSQ+ model with learned scales/zp...
[prepare_model_lsq] LSQ+ int8 QAT: w8/a8
  LSQ Conv: 3 enabled + 1 fp32 (first layer)
  LSQ Linear: 2, BN: 4
  LSQ+ learnable params: scale + zero_point per layer
  首层 Conv 保留 FP32
[enable_qat] Enabled QAT on 6 layers
  Params: 271,298

  LSQ+ (Original) 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  综合硬件对齐率: 0.0% (总展平 0 → 补零后 0)

  [2/3] Patching LSQ layers: LSQ quantize → fake engine matmul...
  Patched 5 layers (stem kept electronic)

  [3/3] Evaluating...
  [optic-LSQ] 5400 batches, report every 540 batch(es)
  [optic-LSQ]  540/5400 ( 10.0%) acc=92.59%  elapsed=1801s  ETA=16205s
  [optic-LSQ] 1080/5400 ( 20.0%) acc=92.59%  elapsed=3982s  ETA=15929s
  [optic-LSQ] 1620/5400 ( 30.0%) acc=92.65%  elapsed=5833s  ETA=13610s
  [optic-LSQ] 2160/5400 ( 40.0%) acc=92.87%  elapsed=7663s  ETA=11494s
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_lsq.py
Device: cpu
============================================================
  Optic-SpaceNet LSQ+: In-Container Optical Inference
  Model 2 LSQ+ (int8, 92.80%)  |  Weight: spacenet_v1_lsq_int8.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  Model 2 LSQ+ INT8  [Optic mode: LSQ quant → real osimulator]
============================================================

  [1/3] Loading LSQ+ model with learned scales/zp...
[prepare_model_lsq] LSQ+ int8 QAT: w8/a8
  LSQ Conv: 3 enabled + 1 fp32 (first layer)
  LSQ Linear: 2, BN: 4
  LSQ+ learnable params: scale + zero_point per layer
  首层 Conv 保留 FP32
[enable_qat] Enabled QAT on 6 layers
  Params: 271,298

  LSQ+ (Original) 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  综合硬件对齐率: 0.0% (总展平 0 → 补零后 0)

  [2/3] Patching LSQ layers: LSQ quantize → fake engine matmul...
  Patched 5 layers (stem kept electronic)

  [3/3] Evaluating...
  [optic-LSQ] 5400 batches, report every 540 batch(es)
  [optic-LSQ]  540/5400 ( 10.0%) acc=92.59%  elapsed=1801s  ETA=16205s
  [optic-LSQ] 1080/5400 ( 20.0%) acc=92.59%  elapsed=3982s  ETA=15929s
  [optic-LSQ] 1620/5400 ( 30.0%) acc=92.65%  elapsed=5833s  ETA=13610s
  [optic-LSQ] 2160/5400 ( 40.0%) acc=92.87%  elapsed=7663s  ETA=11494s
  [optic-LSQ] 2700/5400 ( 50.0%) acc=93.00%  elapsed=9419s  ETA=9419s
  [optic-LSQ] 3240/5400 ( 60.0%) acc=93.12%  elapsed=11045s  ETA=7363s
  [optic-LSQ] 3780/5400 ( 70.0%) acc=92.99%  elapsed=12832s  ETA=5499s
  [optic-LSQ] 4320/5400 ( 80.0%) acc=92.94%  elapsed=14606s  ETA=3651s
  [optic-LSQ] 4860/5400 ( 90.0%) acc=92.74%  elapsed=16375s  ETA=1819s
  [optic-LSQ] 5400/5400 (100.0%) acc=92.76%  elapsed=18118s  ETA=0s
  [optic-LSQ] DONE — 5400 batches, acc=92.76%, total=18118s
  Optical Accuracy: 92.76%  Time: 18118.4s
  [Note] LSQ quant (learned scales) → real osimulator matmul
  [Note] LSQ's per-channel scales make data quantization-friendly;
         _matmul_real re-quantization preserves accuracy

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 18038.276s, 总运算量: 5.15e+09 MACs

====================================================================================================
  Model 2 LSQ+ INT8 — Container Verification Report
====================================================================================================
  Optic osimulator: 92.76%  |  Time: 18118s
  Training ref: 92.80% int8 LSQ+

==============================================================================================================
  LSQ+ INT8 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 LSQ+
  Gazelle 硬件: 8x2 tile, 8a8w, stem 电计算, 其余光计算
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

  [MOPs] 光计算占比: 90.65%  |  总 MOPs: 1.0511 M
  [Note] LSQ+ 模型优势: scale/zp 可直接导出为硬件配置, 无需软件量化
```

```powershell
# python optic_inference_mixed_model1.py --quick 50

[optic]   49/50 ( 98.0%) acc=100.00%  elapsed=7585s  ETA=155s
    [osimulator] (1x4096x32) @ (32x32) input=uint8 ... done (3.8s)
    [osimulator] (1x4096x288) @ (288x32) input=uint8 ... done (30.3s)
    [osimulator] (1x1024x288) @ (288x64) input=uint8 ... done (14.0s)
    [osimulator] (1x1024x576) @ (576x64) input=uint8 ... done (35.3s)
    [osimulator] (1x256x576) @ (576x128) input=uint8 ... done (17.6s)
    [osimulator] (1x256x1152) @ (1152x128) input=uint8 ... done (27.8s)
  [optic]   50/50 (100.0%) acc=100.00%  elapsed=7714s  ETA=0s
  [optic] DONE — 50 batches, acc=100.00%, total=7714s
  Optical Accuracy: 100.00%  Time: 7713.8s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 300, 总耗时: 7688.931s, 总运算量: 7.76e+09 MACs

====================================================================================================
  Model 1 Mixed (VGG) — Container Verification Report
====================================================================================================
  Optic osimulator: 100.00%  |  Time: 7714s
  Training ref: 98.26% int4 Mixed

==============================================================================================================
  Mixed 模型光计算 MOPs 统计 — Model 1 Baseline VGG (Conv=int4 光, Linear=fp32 电)
  Gazelle 硬件: 8x2 tile, 4w8a for Conv, Linear 保留电计算
==============================================================================================================

  Layer            Type    C_in C_out Kernel      Input    ConvOut   Pool  Patch Padded   Align    RawMOPs    OptMOPs   ElecMOPs      Compute
  ------------------------------------------------------------------------------------------------------------------------
  conv1_1          Conv       3    32    3x3      64x64      64x64   None     27     32  84.4%    3.5389M    4.1943M    0.0000M [Optical]
  conv1_2          Conv      32    32    3x3      64x64      64x64 Max2x2    288    288 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv2_1          Conv      32    64    3x3      32x32      32x32   None    288    288 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv2_2          Conv      64    64    3x3      32x32      32x32 Max2x2    576    576 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv3_1          Conv      64   128    3x3      16x16      16x16   None    576    576 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv3_2          Conv     128   128    3x3      16x16      16x16 Max2x2   1152   1152 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  fc1              Linear  8192   256      -          -          -   None   8192   8192 100.0%    2.0972M    0.0000M    2.0972M [Electronic]
  fc2              Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0000M    0.0026M [Electronic]
  ------------------------------------------------------------------------------------------------------------------------
  Total                                                                                          156.6336M  155.1892M    2.0997M

  [MOPs] 光计算占比: 98.67%  |  总 MOPs: 156.63 M
  补零浪费: 0.6554 M (conv1_1 展平=27→32)
  [Note] Mixed 策略: 6 Conv 在 Gazelle 光计算 (int4), 2 Linear 在 CPU/GPU 电计算 (fp32)
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_int4.py
Device: cpu
============================================================
  Optic-SpaceNet INT4: In-Container Optical Inference
  Model 2 Phase4 v2 (int4, 91.06%)  |  Weight: spacenet_v1_phase4_v2_ste.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  Model 2 Phase4 v2 INT4  [Optic mode: osimulator]
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
  [Note] osimulator 原生 8a8w. QAT int4→optical int8 重量化非无损:
         int4 grid (scale=max/7, 16级) → int8 grid (scale=max/127, 256级)
         叠加 per-channel→per-tensor 输入量化差异, 预期光学精度 ~88%
         (QAT 参考: ~94% on test set, 91.06% on val set)

  Optical 层名                          C_in   K      展平长度  补零后  对齐率
  ------------------------------------------------------------------------
  [Conv2d]      stem.0                       3   1×1          3          8   37.5%
  [OpticConv2d] stage1.0                     8   2×2         32         32   100.0%
  [OpticConv2d] stage2.0                    16   2×2         64         64   100.0%
  [OpticConv2d] stage3.0                    32   1×1         32         32   100.0%
  [OpticLinear] classifier.1               —     —           1024       1024   100.0%
  [OpticLinear] classifier.4               —     —            256        256   100.0%
  综合硬件对齐率: 99.6% (总展平 1411 → 补零后 1416)

  [3/3] Evaluating via osimulator (预期 ~88%, 见 EXPERIMENTS.md §16)...
  [optic] 5400 batches, report every 540 batch(es)
  [optic]  540/5400 ( 10.0%) acc=85.74%  elapsed=1828s  ETA=16451s
  [optic] 1080/5400 ( 20.0%) acc=87.22%  elapsed=3619s  ETA=14478s
  [optic] 1620/5400 ( 30.0%) acc=87.47%  elapsed=5404s  ETA=12609s
  [optic] 2160/5400 ( 40.0%) acc=87.41%  elapsed=7142s  ETA=10713s
  [optic] 2700/5400 ( 50.0%) acc=88.00%  elapsed=8669s  ETA=8669s
  [optic] 3240/5400 ( 60.0%) acc=87.96%  elapsed=10994s  ETA=7329s
  [optic] 3780/5400 ( 70.0%) acc=88.07%  elapsed=13869s  ETA=5944s
  [optic] 4320/5400 ( 80.0%) acc=88.17%  elapsed=16727s  ETA=4182s
  [optic] 4860/5400 ( 90.0%) acc=88.00%  elapsed=18384s  ETA=2043s
  [optic] 5400/5400 (100.0%) acc=87.94%  elapsed=20316s  ETA=0s
  [optic] DONE — 5400 batches, acc=87.94%, total=20316s
  Optical Accuracy: 87.94%  Time: 20316.1s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 20247.814s, 总运算量: 5.15e+09 MACs

====================================================================================================
  Model 2 Phase4 v2 INT4 — Container Verification Report
====================================================================================================
  Optic osimulator: 87.94%  |  Time: 20316s
  ---
  QAT 参考: 91.06% (训练 val) / ~94% (test set)
  Optic 预期: ~88% (int4→int8 重量化 + per-channel→per-tensor 导致 ~6% 损失)
  ---
  根因 (详见 EXPERIMENTS.md §16):
    1. 权重 int4→int8: 量化网格 scale=max/7 → scale=max/127
    2. 激活 per-channel→per-tensor: im2col 后通道维度被展平
    3. stem QAT→FP32 电子: BN 统计量不匹配

==============================================================================================================
  INT4 模型光计算 MOPs 统计 — Model 2 SpaceNet V1 Phase4 v2
  Gazelle 硬件: 8x2 tile, act=int8, weight=int4, stem 电计算
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
  光计算占比:          90.65%
  总 MOPs:             1.0511 M
  [Note] stem 展平=3 对齐率仅 37.5%, 保留电计算 (FP32)
  [Note] 预期光学精度 ~88%, QAT 参考 ~94% (test) / 91.06% (val)
  [Note] 6% 损失来源: int4→int8 重量化 + per-channel→per-tensor + stem 不一致
  [Note] 详见 EXPERIMENTS.md §16
```

---
## 7-12

```powershell
PS E:\LT-Simulator\train-test> python model3_spacenet_v2_phase4_v3.py
设备: cpu

============================================================
  Model 3 Phase4 v3: KD + int8 + Gazelle 噪声
  stem FP32 (匹配 osimulator) + Conv/Linear int8 QAT
============================================================
训练: 21600, 验证: 5400
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
      1  |     9.5193   62.10% |    1.4417  73.11% |  73.11% | 0.00020 |  388.4s
      5  |     6.0730   77.40% |    1.1680  80.72% |  81.46% | 0.00100 |   86.2s
     10  |     5.3751   80.74% |    1.2319  81.50% |  84.31% | 0.00099 |   82.4s
     15  |     4.7842   83.70% |    1.0655  86.96% |  86.96% | 0.00097 |   84.0s
     20  |     4.4285   85.53% |    1.0396  87.02% |  87.98% | 0.00094 |   85.4s
     25  |     4.1779   86.53% |    0.9802  87.98% |  89.39% | 0.00090 |   57.6s
     30  |     3.9716   87.69% |    0.9482  88.59% |  89.39% | 0.00084 |   58.3s
     35  |     3.8605   88.39% |    0.9344  89.72% |  90.04% | 0.00078 |   58.6s
     40  |     3.6864   88.89% |    0.9666  90.11% |  90.33% | 0.00070 |   56.9s
     45  |     3.5438   89.57% |    0.8900  90.98% |  90.98% | 0.00063 |   52.7s
     50  |     3.5252   89.68% |    0.9486  90.22% |  91.31% | 0.00055 |   52.7s
     55  |     3.4183   90.14% |    0.8615  91.50% |  91.56% | 0.00046 |   52.0s
     60  |     3.3933   90.31% |    0.8752  91.35% |  91.76% | 0.00038 |   51.8s
     65  |     3.2659   90.86% |    0.8621  91.78% |  91.81% | 0.00031 |   52.4s
     70  |     3.1905   90.99% |    0.8681  91.74% |  91.96% | 0.00023 |   52.1s
     75  |     3.1148   91.42% |    0.8541  91.93% |  92.06% | 0.00017 |   52.0s
     80  |     3.0738   91.81% |    0.8542  91.81% |  92.17% | 0.00011 |   51.2s
     85  |     3.0551   91.67% |    0.8618  91.96% |  92.17% | 0.00007 |   52.9s
     90  |     3.0546   91.44% |    0.8599  91.98% |  92.17% | 0.00004 |   52.2s
     95  |     3.0331   91.93% |    0.8621  92.11% |  92.17% | 0.00002 |   51.2s
    100  |     3.0227   91.84% |    0.8538  92.35% |  92.35% | 0.00001 |   50.9s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int8 模式 (光计算模拟) 准确率: 92.35%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              92.31%
  Int8 量化损失:             -0.04%

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
  训练总耗时:        6589.2s (109.8min)
  硬件对齐率:        99.6%
  Int8 最佳准确率: 92.35%
  Float32 准确率:    92.31%
  FP32 KD 基准:      91.44% (全 fp32 KD)
  osimulator 预期:   ~92.4%% (训练推理配置对齐, 应接近训练精度)
```

```powershell
PS E:\LT-Simulator\train-test> docker start LT-Simulator-container
LT-Simulator-container
PS E:\LT-Simulator\train-test> docker exec -it -w /workspace LT-Simulator-container /bin/bash
(moca_llm) root@a39a38d1a33b:/workspace# cd train-test
bash: cd: train-test: No such file or directory
(moca_llm) root@a39a38d1a33b:/workspace# cd share/train-test
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_kd.py
Device: cpu
============================================================
  Optic-SpaceNet KD+INT4: In-Container Optical Inference
  Model 3 KD Phase4 v2 (int4, 91.50%)  |  Weight: spacenet_v2_phase4_v2_ste.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  Model 3 KD+INT4  [Optic mode: osimulator]
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
  [Note] osimulator uses native 8a8w — QAT int4 weights quantized to int8 (lossless)

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
  [optic]  540/5400 ( 10.0%) acc=84.81%  elapsed=1471s  ETA=13241s
  [optic] 1080/5400 ( 20.0%) acc=84.54%  elapsed=3208s  ETA=12830s
  [optic] 1620/5400 ( 30.0%) acc=84.57%  elapsed=5079s  ETA=11852s
  [optic] 2160/5400 ( 40.0%) acc=84.35%  elapsed=7460s  ETA=11190s
  [optic] 2700/5400 ( 50.0%) acc=83.78%  elapsed=9845s  ETA=9845s
  [optic] 3240/5400 ( 60.0%) acc=84.17%  elapsed=11647s  ETA=7765s
  [optic] 3780/5400 ( 70.0%) acc=84.23%  elapsed=13272s  ETA=5688s
  [optic] 4320/5400 ( 80.0%) acc=84.19%  elapsed=15287s  ETA=3822s
  [optic] 4860/5400 ( 90.0%) acc=84.18%  elapsed=17279s  ETA=1920s
  [optic] 5400/5400 (100.0%) acc=84.33%  elapsed=19173s  ETA=0s
  [optic] DONE — 5400 batches, acc=84.33%, total=19173s
  Optical Accuracy: 84.33%  Time: 19173.0s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 19018.597s, 总运算量: 5.15e+09 MACs

====================================================================================================
  Model 3 KD+INT4 — Container Verification Report
====================================================================================================
  Optic osimulator: 84.33%  |  Time: 19173s
  Training ref: 91.50% int4 (KD from ResNet-18 97.83%)

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
```

```powershell
PS E:\LT-Simulator\train-test> docker exec -it -w /workspace LT-Simulator-container /bin/bash
(moca_llm) root@a39a38d1a33b:/workspace# cd share/train-test
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_kd.py --weight spacenet_v2_phase4_v3_int8.pth
Device: cpu
============================================================
  Optic-SpaceNet KD+INT4: In-Container Optical Inference
  Model 3 KD Phase4 v2 (int4, 91.50%)  |  Weight: spacenet_v2_phase4_v2_ste.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  Model 3 KD+INT4  [Optic mode: osimulator]
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
  [Note] osimulator uses native 8a8w — QAT int4 weights quantized to int8 (lossless)

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
  [optic]  540/5400 ( 10.0%) acc=84.81%  elapsed=2020s  ETA=18176s
  [optic] 1080/5400 ( 20.0%) acc=85.28%  elapsed=4010s  ETA=16040s
  [optic] 1620/5400 ( 30.0%) acc=85.43%  elapsed=5902s  ETA=13772s
```

```powershell
# python optic_inference_kd.py --weight spacenet_v2_phase4_v3_int8.pth --quick 50

[optic]   49/50 ( 98.0%) acc=95.92%  elapsed=144s  ETA=3s
    [osimulator] (1x1024x32) @ (32x16) input=uint8 ... done (0.7s)
    [osimulator] (1x64x64) @ (64x32) input=uint8 ... done (0.6s)
    [osimulator] (1x64x32) @ (32x16) input=uint8 ... done (0.1s)
    [osimulator] (1x1x1024) @ (1024x256) input=uint8 ... done (1.4s)
    [osimulator] (1x1x256) @ (256x10) input=uint8 ... done (0.0s)
  [optic]   50/50 (100.0%) acc=96.00%  elapsed=147s  ETA=0s
  [optic] DONE — 50 batches, acc=96.00%, total=147s
  Optical Accuracy: 96.00%  Time: 146.6s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 250, 总耗时: 146.049s, 总运算量: 4.76e+07 MACs

====================================================================================================
  Model 3 KD — Container Verification Report
====================================================================================================
  Optic osimulator: 96.00%  |  Time: 147s
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
```

```powershell
[optic]   99/100 ( 99.0%) acc=100.00%  elapsed=14804s  ETA=150s
    [osimulator] (1x4096x32) @ (32x32) input=uint8 ... done (4.4s)
    [osimulator] (1x4096x288) @ (288x32) input=uint8 ... done (44.9s)
    [osimulator] (1x1024x288) @ (288x64) input=uint8 ... done (18.7s)
    [osimulator] (1x1024x576) @ (576x64) input=uint8 ... done (41.1s)
    [osimulator] (1x256x576) @ (576x128) input=uint8 ... done (20.9s)
    [osimulator] (1x256x1152) @ (1152x128) input=uint8 ... done (33.7s)
  [optic]  100/100 (100.0%) acc=100.00%  elapsed=14967s  ETA=0s
  [optic] DONE — 100 batches, acc=100.00%, total=14967s
  Optical Accuracy: 100.00%  Time: 14967.5s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 600, 总耗时: 14974.907s, 总运算量: 1.55e+10 MACs

====================================================================================================
  Model 1 Mixed (VGG) — Container Verification Report
====================================================================================================
  Optic osimulator: 100.00%  |  Time: 14967s
  Training ref: 98.26% int4 Mixed

==============================================================================================================
  Mixed 模型光计算 MOPs 统计 — Model 1 Baseline VGG (Conv=int4 光, Linear=fp32 电)
  Gazelle 硬件: 8x2 tile, 4w8a for Conv, Linear 保留电计算
==============================================================================================================

  Layer            Type    C_in C_out Kernel      Input    ConvOut   Pool  Patch Padded   Align    RawMOPs    OptMOPs   ElecMOPs      Compute
  ------------------------------------------------------------------------------------------------------------------------
  conv1_1          Conv       3    32    3x3      64x64      64x64   None     27     32  84.4%    3.5389M    4.1943M    0.0000M [Optical]
  conv1_2          Conv      32    32    3x3      64x64      64x64 Max2x2    288    288 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv2_1          Conv      32    64    3x3      32x32      32x32   None    288    288 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv2_2          Conv      64    64    3x3      32x32      32x32 Max2x2    576    576 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv3_1          Conv      64   128    3x3      16x16      16x16   None    576    576 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv3_2          Conv     128   128    3x3      16x16      16x16 Max2x2   1152   1152 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  fc1              Linear  8192   256      -          -          -   None   8192   8192 100.0%    2.0972M    0.0000M    2.0972M [Electronic]
  fc2              Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0000M    0.0026M [Electronic]
  ------------------------------------------------------------------------------------------------------------------------
  Total                                                                                          156.6336M  155.1892M    2.0997M

  [MOPs] 光计算占比: 98.67%  |  总 MOPs: 156.63 M
  补零浪费: 0.6554 M (conv1_1 展平=27→32)
  [Note] Mixed 策略: 6 Conv 在 Gazelle 光计算 (int4), 2 Linear 在 CPU/GPU 电计算 (fp32)
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_kd.py --weight spacenet_v2_phase4_v3_int8.pth
Device: cpu
============================================================
  Optic-SpaceNet KD: In-Container Optical Inference
  Model 3 KD Phase4 v3 (int8+KD, TBD)  |  Weight: spacenet_v2_phase4_v3_int8.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  [optic]  540/5400 ( 10.0%) acc=93.52%  elapsed=2044s  ETA=18396s
  [optic] 1080/5400 ( 20.0%) acc=93.15%  elapsed=4511s  ETA=18043s
  [optic] 1620/5400 ( 30.0%) acc=93.52%  elapsed=6995s  ETA=16323s
  [optic] 2160/5400 ( 40.0%) acc=93.47%  elapsed=9480s  ETA=14220s
  [optic] 2700/5400 ( 50.0%) acc=93.37%  elapsed=12030s  ETA=12030s
  [optic] 3240/5400 ( 60.0%) acc=93.21%  elapsed=15155s  ETA=10103s
```

```powershell
(moca_llm) root@a39a38d1a33b:/workspace/share/train-test# python optic_inference_kd.py --weight spacenet_v2_phase4_v3_int8.pth
Device: cpu
============================================================
  Optic-SpaceNet KD: In-Container Optical Inference
  Model 3 KD Phase4 v3 (int8+KD, TBD)  |  Weight: spacenet_v2_phase4_v3_int8.pth
  Mode: Optic (default)
============================================================

--- Loading Test Set ---
Full: 27000 | Test: 5400 imgs | Test/Val overlap: 0
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
  [optic]  540/5400 ( 10.0%) acc=93.52%  elapsed=2044s  ETA=18396s
  [optic] 1080/5400 ( 20.0%) acc=93.15%  elapsed=4511s  ETA=18043s
  [optic] 1620/5400 ( 30.0%) acc=93.52%  elapsed=6995s  ETA=16323s
  [optic] 2160/5400 ( 40.0%) acc=93.47%  elapsed=9480s  ETA=14220s
  [optic] 2700/5400 ( 50.0%) acc=93.37%  elapsed=12030s  ETA=12030s
  [optic] 3240/5400 ( 60.0%) acc=93.21%  elapsed=15155s  ETA=10103s
  [optic] 3780/5400 ( 70.0%) acc=93.31%  elapsed=16869s  ETA=7229s
  [optic] 4320/5400 ( 80.0%) acc=93.29%  elapsed=18611s  ETA=4653s
  [optic] 4860/5400 ( 90.0%) acc=93.17%  elapsed=20220s  ETA=2247s
  [optic] 5400/5400 (100.0%) acc=93.26%  elapsed=21784s  ETA=0s
  [optic] DONE — 5400 batches, acc=93.26%, total=21784s
  Optical Accuracy: 93.26%  Time: 21783.6s

--- Optical Engine Statistics ---
  [OpticalEngine 统计] 调用: 27000, 总耗时: 21553.957s, 总运算量: 5.15e+09 MACs

====================================================================================================
  Model 3 KD — Container Verification Report
====================================================================================================
  Optic osimulator: 93.26%  |  Time: 21784s
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
```

```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_phase4_v3.py --variant A
设备: cpu

============================================================
  Model 1 Phase4 v3: Baseline VGG int8 + Gazelle 硬件噪声
  变体 A: 首层 conv1_1 FP32, 其余 Conv+Linear int8
============================================================
训练: 21600, 验证: 5400
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
      1  |     1.4011   56.69% |    1.0188  69.96% |  69.96% | 0.00020 |  380.7s
      5  |     0.8698   79.84% |    0.6878  85.83% |  85.83% | 0.00100 |  354.6s
     10  |     0.6340   90.03% |    0.5358  91.81% |  91.81% | 0.00099 |  194.4s
     15  |     0.5339   93.63% |    0.5127  92.37% |  94.31% | 0.00097 |  195.3s
     20  |     0.4931   94.78% |    0.4300  95.22% |  95.39% | 0.00094 |  204.2s
     25  |     0.4566   95.90% |    0.4469  94.65% |  96.04% | 0.00090 |  200.0s
     30  |     0.4412   96.59% |    0.3879  96.81% |  96.81% | 0.00084 |  196.1s
     35  |     0.4175   97.28% |    0.3940  96.69% |  97.00% | 0.00078 |  197.2s
     40  |     0.3990   98.04% |    0.4033  95.67% |  97.00% | 0.00070 |  262.4s
     45  |     0.3895   98.36% |    0.3740  97.26% |  97.26% | 0.00063 |  307.1s
     50  |     0.3815   98.59% |    0.3627  97.70% |  97.70% | 0.00055 |  175.6s
     55  |     0.3706   99.03% |    0.3702  97.17% |  97.76% | 0.00046 |  177.0s
     60  |     0.3642   99.28% |    0.3604  97.76% |  97.83% | 0.00038 |  225.7s
     65  |     0.3614   99.29% |    0.3628  97.61% |  97.94% | 0.00031 |  214.1s
     70  |     0.3565   99.51% |    0.3588  97.96% |  97.96% | 0.00023 |  205.1s
     75  |     0.3535   99.61% |    0.3546  97.89% |  98.13% | 0.00017 |  205.1s
     80  |     0.3492   99.69% |    0.3536  97.87% |  98.13% | 0.00011 |  196.5s
     85  |     0.3478   99.69% |    0.3537  98.06% |  98.15% | 0.00007 |  198.2s
     90  |     0.3472   99.75% |    0.3532  98.00% |  98.15% | 0.00004 |  196.6s
     95  |     0.3448   99.79% |    0.3522  98.02% |  98.15% | 0.00002 |  197.6s
    100  |     0.3459   99.81% |    0.3518  98.06% |  98.15% | 0.00001 |  170.6s

[Step 3] 最终评估 (val set)
[enable_qat] Enabled QAT on 8 layers
  Int8 模式 (光计算模拟) 准确率: 98.15%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:          98.13%
  Int8 量化损失:               -0.02%

  模型已保存: baseline_vgg_phase4_v3_int8.pth

============================================================
  训练完成 — 结果汇总 (变体 A)
============================================================
  模型:              Baseline VGG (flat+BN, bias=False)
  参数量:            2,387,168
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  电计算层 (FP32):   conv1_1
  训练总耗时:        21613.5s (360.2min)
  硬件对齐率:        100.0%
  Int8 最佳准确率:   98.15%
  Float32 准确率:    98.13%
  量化损失:          -0.02%
  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%
  推理脚本:          python optic_inference_int8_model1.py --variant A
```

```powershell
PS E:\LT-Simulator\train-test> python model1_baseline_phase4_v3.py --variant B
设备: cpu

============================================================
  Model 1 Phase4 v3: Baseline VGG int8 + Gazelle 硬件噪声
  变体 B: 首层 conv1_1 FP32 + conv3_2 FP32, 其余 Conv+Linear int8
============================================================
训练: 21600, 验证: 5400
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
      1  |     1.4261   55.77% |    0.9529  75.02% |  75.02% | 0.00020 |  364.4s
      5  |     0.8319   81.85% |    0.7895  82.61% |  86.24% | 0.00100 |  362.6s
     10  |     0.6203   90.07% |    0.4977  93.22% |  93.22% | 0.00099 |  194.5s
     15  |     0.5359   93.16% |    0.4825  93.56% |  94.37% | 0.00097 |  194.0s
     20  |     0.4899   94.85% |    0.4201  95.69% |  95.69% | 0.00094 |  203.8s
     25  |     0.4621   95.75% |    0.3921  96.20% |  96.20% | 0.00090 |  200.0s
     30  |     0.4360   96.83% |    0.4133  95.72% |  96.81% | 0.00084 |  195.0s
     35  |     0.4228   97.25% |    0.3901  96.50% |  96.85% | 0.00078 |  199.4s
     40  |     0.4021   97.99% |    0.3665  97.46% |  97.46% | 0.00070 |  260.5s
     45  |     0.3942   98.06% |    0.3728  97.20% |  97.61% | 0.00063 |  308.4s
     50  |     0.3807   98.67% |    0.4082  95.70% |  97.61% | 0.00055 |  174.4s
     55  |     0.3740   98.89% |    0.3582  97.59% |  97.61% | 0.00046 |  174.0s
     60  |     0.3673   99.06% |    0.3612  97.54% |  97.72% | 0.00038 |  225.0s
     65  |     0.3596   99.26% |    0.3552  97.83% |  97.98% | 0.00031 |  214.4s
     70  |     0.3555   99.48% |    0.3538  97.98% |  97.98% | 0.00023 |  205.0s
     75  |     0.3510   99.60% |    0.3517  97.81% |  97.98% | 0.00017 |  206.1s
     80  |     0.3487   99.68% |    0.3528  97.91% |  97.98% | 0.00011 |  194.7s
     85  |     0.3468   99.73% |    0.3502  97.96% |  97.98% | 0.00007 |  196.0s
     90  |     0.3459   99.74% |    0.3506  97.96% |  98.02% | 0.00004 |  195.5s
     95  |     0.3455   99.73% |    0.3483  97.91% |  98.02% | 0.00002 |  197.6s
    100  |     0.3444   99.76% |    0.3493  97.93% |  98.02% | 0.00001 |  196.7s

[Step 3] 最终评估 (val set)
[enable_qat] Enabled QAT on 8 layers
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128
  Int8 模式 (光计算模拟) 准确率: 98.02%
[disable_qat] Disabled QAT on 8 layers
  Float32 模式准确率:          98.07%
  Int8 量化损失:               0.06%

  模型已保存: baseline_vgg_phase4_v3_int8_vB.pth

============================================================
  训练完成 — 结果汇总 (变体 B)
============================================================
  模型:              Baseline VGG (flat+BN, bias=False)
  参数量:            2,387,168
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  电计算层 (FP32):   conv1_1 + conv3_2
  训练总耗时:        21491.8s (358.2min)
  硬件对齐率:        100.0%
  Int8 最佳准确率:   98.02%
  Float32 准确率:    98.07%
  量化损失:          +0.06%
  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%
  推理脚本:          python optic_inference_int8_model1.py --variant B
```

```powershell
PS E:\LT-Simulator\train-test> python model2_spacenet_v1_phase4_v3.py
设备: cpu

============================================================
  Model 2 Phase4 v3: int8 权重 + Gazelle 硬件噪声
  首层 stem FP32 (对齐率 37.5%), 其余 Conv+Linear int8
============================================================
训练: 16200, 验证: 5400, 留出测试: 5400 (见 eurosat_split)
类别: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

参数量: 267,944

[Step 1] 转换为 QAT v4 (int8 权重, Gazelle 噪声)
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 3 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 4
  硬件噪声: GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-04, ADC_lsb=0.0015)
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [SpaceNet V1 (v4)] 层名                           C_in      K      展平长度       补零后      对齐率
  ------------------------------------------------------------------------
  [QATConv2d_v4 FP32 ] stem.0                       3   1×1         3         8   37.5%  w8
  [QATConv2d_v4 QAT  ] stage1.0                     8   2×2        32        32  100.0%  w8
  [QATConv2d_v4 QAT  ] stage2.0                    16   2×2        64        64  100.0%  w8
  [QATConv2d_v4 QAT  ] stage3.0                    32   1×1        32        32  100.0%  w8
  综合硬件对齐率: 99.6% (展平总长度 1411 → 补零后 1416)

[Step 2] 训练 (100 epochs, lr=0.001, wd=0.0005)
  int8 权重 (硬件原生精度)
  GazelleNoise(DAC_ENOB=7.5, TIA_σ=5.3e-4) — 硬件匹配噪声
----------------------------------------------------------------------
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
  --------------------------------------------------------------------
      1  |     1.2825   61.63% |    1.0081  71.74% |  71.74% | 0.00020 |   36.1s
      5  |     0.9379   75.95% |    0.8552  78.15% |  78.15% | 0.00100 |   35.0s
     10  |     0.8417   80.63% |    0.7397  83.13% |  83.13% | 0.00099 |   35.9s
     15  |     0.7720   83.23% |    0.6886  85.07% |  85.07% | 0.00097 |   56.3s
     20  |     0.7155   85.64% |    0.6597  86.63% |  86.76% | 0.00094 |   67.3s
     25  |     0.6773   86.54% |    0.6120  88.19% |  88.19% | 0.00090 |   39.9s
     30  |     0.6598   87.59% |    0.6012  88.76% |  89.11% | 0.00084 |   50.1s
     35  |     0.6382   88.47% |    0.5799  90.02% |  90.02% | 0.00078 |   46.4s
     40  |     0.6130   89.43% |    0.5704  90.22% |  90.22% | 0.00070 |   39.3s
     45  |     0.6112   89.70% |    0.5656  90.19% |  90.22% | 0.00063 |   44.7s
     50  |     0.5863   90.77% |    0.5468  91.11% |  91.11% | 0.00055 |   37.8s
     55  |     0.5655   91.54% |    0.5450  91.20% |  91.20% | 0.00046 |   43.2s
     60  |     0.5564   91.66% |    0.5473  91.06% |  91.35% | 0.00038 |   42.2s
     65  |     0.5559   91.71% |    0.5472  91.00% |  91.48% | 0.00031 |   41.4s
     70  |     0.5477   92.32% |    0.5580  90.80% |  91.48% | 0.00023 |   39.3s
     75  |     0.5347   92.38% |    0.5397  91.20% |  91.91% | 0.00017 |   41.4s
     80  |     0.5296   92.56% |    0.5299  91.46% |  91.91% | 0.00011 |   44.6s
     85  |     0.5265   92.65% |    0.5240  91.96% |  91.96% | 0.00007 |   44.8s
     90  |     0.5215   93.02% |    0.5246  91.89% |  91.96% | 0.00004 |   45.7s
     95  |     0.5219   93.05% |    0.5270  91.87% |  91.96% | 0.00002 |   44.0s
    100  |     0.5233   93.07% |    0.5220  92.06% |  92.06% | 0.00001 |   43.9s

[Step 3] 最终评估
[enable_qat] Enabled QAT on 6 layers
  Int8 模式 (光计算模拟) 准确率: 92.06%
[disable_qat] Disabled QAT on 6 layers
  Float32 模式准确率:              91.76%
  Int8 量化损失:             -0.30%

  模型已保存: spacenet_v1_phase4_v3_int8.pth

============================================================
  训练完成 — 结果汇总
============================================================
  模型:              SpaceNet V1 (bias=False)
  参数量:            267,944
  权重量化:          int8 (硬件原生 8-bit)
  噪声模型:          Gazelle (DAC 7.5 + TIA)
  首层:              FP32 (对齐率 37.5%)
  训练总耗时:        4370.0s (72.8min)
  硬件对齐率:        99.6%
  Int8 最佳准确率: 92.06%
  Float32 准确率:    91.76%
  旧版 int4 参考:    74.35% (Phase4, Conv QAT 全关)
  FP32 基准:         90.15%
```

```powershell
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

---
## 7-14

```powershell
PS E:\LT-Simulator\train-test> python -u model1_baseline_phase4_v3.py --variant A 2>&1 | tee train_A_v2.log
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
  Int8 量化损失:               0.06%

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

```powershell
PS E:\LT-Simulator\train-test> python -u model1_baseline_phase4_v3.py --variant B 2>&1 | tee train_B_v2.log
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

```powershell
PS E:\LT-Simulator\train-test> python -u optic_inference_int8_model1.py --variant A --qat --batch 256 2>&1 | tee qat_A_v2.log
Device: cpu
============================================================
  Optic-SpaceNet Model 1 INT8: In-Container Optical Inference
  Model:  Baseline VGG Phase4 v3 (变体 A)
  Weight: baseline_vgg_phase4_v3_int8.pth
  Mode:   QAT (pseudo-quant)  |  Batch: 256  full
============================================================

--- Loading Independent Test Set ---
Full: 27000 | Test(now): 5400 | split=eurosat_split (test∩train=0)

[Mode: QAT] PyTorch pseudo-quantization cross-validation...

============================================================
  Model 1 Phase4 v3 INT8 (变体 A)  [QAT mode: int8]
============================================================

  [1/3] Creating model...
  Params: 2,387,168

  [2/3] Converting to QAT v4 (int8, 首层 conv1_1 FP32)...
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 5 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 6
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)

  [3/3] Loading INT8 QAT weights: baseline_vgg_phase4_v3_int8.pth

  --- Native float32 (QAT disabled) ---
[disable_qat] Disabled QAT on 8 layers
  [Model 1 Phase4 v3 INT8 (变体 A) fp32] 22 batches — acc=97.91%
  Float32: 97.91% (62.7s)

  --- int8 QAT (光计算模拟) ---
[enable_qat] Enabled QAT on 8 layers
  [Model 1 Phase4 v3 INT8 (变体 A) int8] 22 batches — acc=97.89%
  Int8 QAT: 97.89% (21.3s)
  Quant Loss: +0.02%

====================================================================================================
  Model 1 INT8 (变体 A) — Container Verification Report
====================================================================================================
  QAT float32: 97.91%  |  QAT int8: 97.89%  |  Quant Loss: +0.02%
  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%


==============================================================================================================
  Model 1 INT8 光计算 MOPs 统计 — Baseline VGG Phase4 v3 (变体 A)
  Gazelle 硬件: 8×2 tile, 8a8w12o | 电计算层 (FP32): ['conv1_1']
==============================================================================================================

  Layer      Type    C_in C_out Kernel      Input    ConvOut   Pool  Patch Padded   Align    RawMOPs    OptMOPs   ElecMOPs      Compute
  ------------------------------------------------------------------------------------------------------------------------
  conv1_1    Conv       3    32    3x3      64x64      64x64   None     27     32  84.4%    3.5389M    0.0000M    3.5389M [Electronic]
  conv1_2    Conv      32    32    3x3      64x64      64x64 Max2x2    288    288 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv2_1    Conv      32    64    3x3      32x32      32x32   None    288    288 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv2_2    Conv      64    64    3x3      32x32      32x32 Max2x2    576    576 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv3_1    Conv      64   128    3x3      16x16      16x16   None    576    576 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv3_2    Conv     128   128    3x3      16x16      16x16 Max2x2   1152   1152 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  fc1        Linear  8192   256      -          -          -   None   8192   8192 100.0%    2.0972M    2.0972M    0.0000M [Optical]
  fc2        Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0026M    0.0000M [Optical]
  ------------------------------------------------------------------------------------------------------------------------
  Total                                                                                    156.6336M  153.0947M    3.5389M

  ------------------------------------------------------------
  [MOPs] 光计算占比汇总 (变体 A)
  ------------------------------------------------------------
  总原始 MOPs:           156.6336 M
  光计算 MOPs (有效):    153.0947 M
  电子计算 MOPs:         3.5389 M
  总有效 MOPs:           156.6336 M
  -------------------------------------
  ** 光计算占比:         97.74%  ([OK] 达标 (≥50%))
  光计算补零浪费:        0 (光计算层均对齐 8 的倍数) [OK]
```

```powershell
PS E:\LT-Simulator\train-test> python -u optic_inference_int8_model1.py --variant B --qat --batch 256 2>&1 | tee qat_B_v2.log
Device: cpu
============================================================
  Optic-SpaceNet Model 1 INT8: In-Container Optical Inference
  Model:  Baseline VGG Phase4 v3 (变体 B)
  Weight: baseline_vgg_phase4_v3_int8_vB.pth
  Mode:   QAT (pseudo-quant)  |  Batch: 256  full
============================================================

--- Loading Independent Test Set ---
Full: 27000 | Test(now): 5400 | split=eurosat_split (test∩train=0)

[Mode: QAT] PyTorch pseudo-quantization cross-validation...

============================================================
  Model 1 Phase4 v3 INT8 (变体 B)  [QAT mode: int8]
============================================================

  [1/3] Creating model...
  Params: 2,387,168

  [2/3] Converting to QAT v4 (int8, 首层 conv1_1 FP32 + conv3_2 FP32)...
[prepare_model_v4] Gazelle HW-aware QAT: wint8/a8
  QAT Conv: 5 enabled + 1 fp32 (first layer)
  QAT Linear: 2, BN: 6
  首层 Conv 保留 FP32 (对齐率低, 电计算更高效)
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128

  [3/3] Loading INT8 QAT weights: baseline_vgg_phase4_v3_int8_vB.pth

  --- Native float32 (QAT disabled) ---
[disable_qat] Disabled QAT on 8 layers
  [Model 1 Phase4 v3 INT8 (变体 B) fp32] 22 batches — acc=98.04%
  Float32: 98.04% (51.5s)

  --- int8 QAT (光计算模拟) ---
[enable_qat] Enabled QAT on 8 layers
  [Variant B] conv3_2 保持 FP32 (电计算): 128→128
  [Model 1 Phase4 v3 INT8 (变体 B) int8] 22 batches — acc=97.96%
  Int8 QAT: 97.96% (20.9s)
  Quant Loss: +0.07%

====================================================================================================
  Model 1 INT8 (变体 B) — Container Verification Report
====================================================================================================
  QAT float32: 98.04%  |  QAT int8: 97.96%  |  Quant Loss: +0.07%
  参考: FP32 基准 97.17% | int4 Mixed 98.26% | int4 STE 96.46%


==============================================================================================================
  Model 1 INT8 光计算 MOPs 统计 — Baseline VGG Phase4 v3 (变体 B)
  Gazelle 硬件: 8×2 tile, 8a8w12o | 电计算层 (FP32): ['conv1_1', 'conv3_2']
==============================================================================================================

  Layer      Type    C_in C_out Kernel      Input    ConvOut   Pool  Patch Padded   Align    RawMOPs    OptMOPs   ElecMOPs      Compute
  ------------------------------------------------------------------------------------------------------------------------
  conv1_1    Conv       3    32    3x3      64x64      64x64   None     27     32  84.4%    3.5389M    0.0000M    3.5389M [Electronic]
  conv1_2    Conv      32    32    3x3      64x64      64x64 Max2x2    288    288 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv2_1    Conv      32    64    3x3      32x32      32x32   None    288    288 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv2_2    Conv      64    64    3x3      32x32      32x32 Max2x2    576    576 100.0%   37.7487M   37.7487M    0.0000M [Optical]
  conv3_1    Conv      64   128    3x3      16x16      16x16   None    576    576 100.0%   18.8744M   18.8744M    0.0000M [Optical]
  conv3_2    Conv     128   128    3x3      16x16      16x16 Max2x2   1152   1152 100.0%   37.7487M    0.0000M   37.7487M [Electronic]
  fc1        Linear  8192   256      -          -          -   None   8192   8192 100.0%    2.0972M    2.0972M    0.0000M [Optical]
  fc2        Linear   256    10      -          -          -   None    256    256 100.0%    0.0026M    0.0026M    0.0000M [Optical]
  ------------------------------------------------------------------------------------------------------------------------
  Total                                                                                    156.6336M  115.3459M   41.2877M

  ------------------------------------------------------------
  [MOPs] 光计算占比汇总 (变体 B)
  ------------------------------------------------------------
  总原始 MOPs:           156.6336 M
  光计算 MOPs (有效):    115.3459 M
  电子计算 MOPs:         41.2877 M
  总有效 MOPs:           156.6336 M
  -------------------------------------
  ** 光计算占比:         73.64%  ([OK] 达标 (≥50%))
  光计算补零浪费:        0 (光计算层均对齐 8 的倍数) [OK]
```






