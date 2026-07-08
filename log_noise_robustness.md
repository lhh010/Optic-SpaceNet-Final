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
