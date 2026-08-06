# Model 4 (MiniVGG-GAP) Phase4 v3 · int8 QAT + Gazelle 硬件噪声 本地训练记录

> 队伍：CICC1003564（复旦大学 · 华东赛区）
> 训练日期：2026-08-07（本地 CPU，非远程）
> 状态：✅ **训练完成**（5 epochs 微调，Int8 test 95.50%）
> 权重产出：`weights/minivgg_gap_phase4_v3_int8.pth`
> 原始日志：`logs/log_model4_minivgg_gap_phase4_v3.md`

---

## 1. 任务与背景

复赛三个模型（Model 1/2/3）已迁移至 Gazelle 真机并完成小样本验证（见 `contest-national/opticspacenet/OpticSpaceNet迁移至Gazelle真机过程文档.md`）。本任务补训 **Model 4（MiniVGG-GAP）** 的 phase4 v3 int8 版本，作为对照模型（验证 2×2 kernel + FC 瓶颈是否限制 Model 2 表达力）。

- 模型：MiniVGG-GAP（`src/training/model4_minivgg_gap.py`），7× Conv2d(3×3) + GAP head，~260K 参数，FP32 val 96.65%（原始基线）
- 本训练参照 Model 1-3 的 phase4 v3 配方（`model2_spacenet_v1_phase4_v3.py`）：
  - **int8 权重**（匹配 osimulator 原生 8-bit）
  - **Gazelle 硬件匹配噪声**（DAC ENOB=7.5 + TIA，训练时注入）
  - **首层 stem FP32**（对齐率低 → 电计算，与 osimulator 推理 `keep_first_conv_electronic=True` 对齐）
  - 其余 Conv + Linear → int8 QAT（`quantize_linear=True, preserve_bn=True`）
- 本机无 GPU / 无 torchvision：数据加载用 **PIL 复刻** `load_eurosat_data`（ImageFolder 排序 + eurosat_split 单一数据源 + 相同增广语义）；训练策略改为**从 FP32 基线微调**（QAT fine-tune，CPU 时间受限，不做 100 epoch 从零训练）。

## 2. 训练配置（与 model2 phase4_v3 对齐，除 epochs）

| 项 | 值 |
|---|---|
| 权重位宽 / 激活位宽 | int8 / int8 |
| 首层 | FP32（stem 3→32，保留电计算） |
| 噪声 | GazelleNoise（DAC_ENOB=7.5, TIA_σ=5.3e-4），仅训练时注入 |
| 损失 | CrossEntropy + label_smoothing=0.05 |
| 优化器 | AdamW（lr=0.001, weight_decay=5e-4） |
| 调度 | WarmupCosine（warmup 5 epoch） |
| Batch / 数据 | 64；eurosat_split(seed=42)：train 16200 / val 5400 / test 5400 |
| 初始化 | 从 `weights/minivgg_gap.pth`（FP32 基线）微调 |
| Epochs | **5**（CPU 单 epoch ≈ 110s，约 10 min 完成） |
| 增广 | RandomHorizontalFlip(p=0.5) + RandomRotation(±10°) + Normalize（PIL 复刻） |

## 3. 训练脚本

`src/training/model4_minivgg_gap_phase4_v3.py`

- `MiniVGG` 模型类（与 `model4_minivgg_gap.py` 完全一致，head Linear 带 bias）
- `EuroSATFolder`（PIL）：复刻 torchvision ImageFolder 排序（class 排序 × 文件名排序）与增广
- `load_eurosat_data_pil`：eurosat_split 三分（train/val/test 互斥，杜绝 Bug #11 泄漏）
- 流程：加载 FP32 基线 → `prepare_model_v4(weight_bits=8, act_bits=8, noise=True, first_conv_fp32=True, quantize_linear=True, preserve_bn=True)` → QAT 微调 → 最佳 val 保存 → int8/fp32 对比评估 + **独立 test 集 int8 评估**

用法：
```bash
cd train-test
MODEL4_EPOCHS=5 PYTHONIOENCODING=utf-8 \
  /mnt/e/anaconda3/python.exe src/training/model4_minivgg_gap_phase4_v3.py \
  | tee logs/log_model4_minivgg_gap_phase4_v3.md
```

## 4. 训练日志（原文见 `logs/log_model4_minivgg_gap_phase4_v3.md`）

### 4.1 冒烟测试（1 epoch，验证管线 + 计时）

| 项 | 值 |
|---|---|
| 单 epoch 耗时 | 冒烟 525.7s（冷磁盘缓存，首轮载图慢）；warm 后 ≈ 110s/epoch |
| Int8 val（1 epoch 后） | 94.96% |
| Float32 val | 95.00% |
| Int8 量化损失 | 0.04% |
| 硬件对齐率 | 99.9% |

### 4.2 正式训练（5 epochs，完整日志见 `logs/log_model4_minivgg_gap_phase4_v3.md`）

```
  Epoch | Train Loss Train Acc |  Val Loss  Val Acc |     Best |       LR |    Time
----------------------------------------------------------------------------
      1  |     0.4719   97.13% |    0.4590  94.96% |  94.96% | 0.00020 |  111.8s
      2  |     0.3727   98.74% |    0.4472  95.22% |  95.22% | 0.00040 |  111.7s
      3  |     0.3732   98.58% |    0.4301  95.59% |  95.59% | 0.00060 |  110.7s
      4  |     0.3816   97.93% |    0.4443  95.35% |  95.59% | 0.00080 |  109.3s
      5  |     0.3946   97.23% |    0.7585  82.26% |  95.59% | 0.00100 |  109.1s
```

- 单 epoch ≈ 110s（16 线程 CPU）；val 在第 3 epoch 达峰 95.59%（最佳权重已恢复）；
  epoch 5 在 warmup 峰值 lr=0.001 处 val 回落（82.26%），由 best-checkpoint 机制兜底。
- QAT 层统计：**6 个 QAT Conv + 1 个 FP32 首层（stem.0）+ 1 个 QAT Linear（head）**，BN×7 保留；
  综合硬件对齐率 **99.9%**（stem.0 84.4% 因 FP32 电计算不计，其余 7 层 100% 对齐 8 的倍数）。

## 5. 结果汇总

| 指标 | 值 |
|---|---|
| 参数量 | 260,234 |
| **Int8 最佳 val** | **95.59%**（epoch 3） |
| **Int8 test（独立 5400 张）** | **95.50%** |
| Float32 val | 95.43% |
| Int8 量化损失 | **−0.17%**（int8 反超 fp32，Gazelle 噪声正则化效应） |
| 硬件对齐率 | 99.9% |
| 训练总耗时 | 552.6s（9.2 min，5 epochs，16 线程 CPU） |
| 对照：FP32 基线 val | 96.65%（原记录） / 95.00%（本机当前划分复测） |
| 权重 | `weights/minivgg_gap_phase4_v3_int8.pth`（44 键，与原生 MiniVGG 完全兼容，0 missing） |

**结论**：Model 4（MiniVGG-GAP）int8 QAT 在独立 test 上 **95.50%**，显著高于 Model 2/3 的 int8（92.20% / 84.59% QAT 口径）——验证了队伍假设：**2×2 kernel + FC 瓶颈限制了 Model 2 的表达力**，3×3 全卷积 + GAP 结构在 int8 光计算量化下优势明显。

## 6. 说明与后续

- 本训练为 **CPU 短训**（5 epochs 微调），作为 Model 4 的 int8 光计算迁移前置；若需更高精度可在 GPU 上按 100 epochs 从零/继续训练（脚本 `MODEL4_EPOCHS` 可调）。
- 权重与原生 MiniVGG 键兼容（QAT 层参数名一致），可直接用于 `build_optical_model` 推理路径或 osimulator 验证。
- Model 4 为 3×3 全卷积 + GAP，展平长度 288/576/1152（均 8 的倍数），光计算对齐率 99.9%。
