# Auto-Research 探索交付：光计算最优端到端方案

> 队伍 CICC1003564 · 第十届集创赛曦智 Gazelle 命题 · 2026-08
> 探索目录：`Ltsimulator-test/auto_research/`（代码 + runs + docs）

---

## 一、核心洞察（Insight 汇总）

### 1. MACs 是第一性约束，架构搜索收益远超优化器
- 17M MACs（Model 4）→ **1.38M MACs（J1）**，12× 压缩，精度仅降 1.9pt（FP32 口径）
- **全 1×1 kernel 系统性最优**：1×1 展平长度 = 通道数，与硬件 tile 对齐（im2col 需 8 倍数，1×1 天然满足）；深层 1×1 等效全连接，在 8×2 tile 上无冗余
- **更宽 ≠ 更好**：G3x（3.08M MACs）反不如 H1（1.21M MACs）
- stem 用 stride=2 保留 32×32 底层特征，优于 stride=4

### 2. QAT 范式：噪声匹配训练是核心，优化器选择 SGD
- **SGD+Momentum 最优**（97.43%）：动量平均天然抑制量化噪声梯度
- Muon 次之（97.22%），AdamW 第三（96.93%）——正交化动量在 QAT 下反而鲁棒
- 噪声结构必须匹配真机：**绝对加性底噪**（σ_total≈4.49 counts，与信号幅度无关），非 sim 的"底噪+信号相关"
- 强增强（Rot90/ResizedCrop/ColorJitter）破坏 QAT 稳定性（−1.26pt）——量化噪声下的数据增强要保守

### 3. 真机与模拟器本质差异（gazelle-crossval 结论，决定性输入）
- osimulator：底噪 + 信号相关，每次调用随机，结构错误
- 真机：纯绝对加性底噪（σ_total≈4.5 counts，σ_static 慢漂主导 4.37）
- **uint4 上板不可行**（信号 ±1.5 counts < 噪声底 4.4，SNR<1）
- 校准是部署的必需品：`compass_cali` + per-layer alpha/beta

### 4. 部署工程要点（实测踩坑）
- compass_sdk 篡改 sys.argv → 部署脚本禁位置参数，一律环境变量
- FPGA block-matmul m≥3 行回绕 → m≤2 tiling
- **反量化公式**：y = x_scale·w_scale·y_int **−** x_scale·zp·w_scale·col_sum（易错：符号 + 缺 x_scale）
- 每光计算层后 BN 必须显式应用（torch BN 在部署 numpy 前向中容易漏）
- **真机需 per-layer alpha/beta 校准**：hw = alpha·ideal + beta + eps（alpha≈1.03-1.06）

---

## 二、探索路线图（Rounds 1-4）

| Round | 内容 | 关键结果 |
|---|---|---|
| R1 | QAT 范式对照（17M MACs） | SGD 97.43% 最优；绝对底噪价值小但为正 |
| R2 | 架构搜索 ≤2M MACs | **J1 冠军** 1.38M MACs / 95.52% |
| R3 | J1 精调 | J1_long(160ep) **96.30%**；SWA 无效（avg<best） |
| R4 | 真机部署验证 | FAKE 96.40% ≈ QAT；**真机 90.60%** |

### 完整精度链条（EuroSAT test）
```
FP32 Model 4 (17M MACs)     96.65%
J1 QAT 软件 (1.38M MACs)    96.30%
J1 FAKE numpy (离线)        96.40%
J1 真机 Gazelle (校准后)    90.60%   ← 硬件噪声极限
```
硬件 gap 5.7pt：每层绝对噪声底（resid_std 3-10%）× 7 层光计算累积。

---

## 三、最优端到端方案（J1 · 1.38M MACs）

### 模型结构
```
stem:  Conv3x3 s2 3→16 + BN + ReLU + MaxPool2      [电计算]
stage1: Conv1x1 16→32 + BN + ReLU + MaxPool2        [光计算]
stage2: Conv1x1 32→64 + BN + ReLU
        Conv1x1 64→64 + BN + ReLU + MaxPool2        [光计算]
stage3: Conv1x1 64→128 + BN + ReLU
        Conv1x1 128→128 + BN + ReLU                 [光计算]
head:   GAP → FC 128→128 ReLU → FC 128→10
```
- 参数 45.9K，MACs 1.38M，bias=False（conv）
- 每 conv 后 BN（部署需显式导出应用）

### 训练配方（configs/r3_J1_long.json）
```
channels=[16,32,64,128], stem_stride=2, fast_downsample=True
kernels=[1,1,1], head_dims=[128]
epochs=160, optimizer=SGD(lr=0.05), warmup=5, cosine
label_smoothing=0.05, aug=standard(HFlip+Rot10), batch=64
QAT v5: activation_style=osim(per-tensor uint8+zp), output_quant=12bit
        output_noise=True, ratio=0.0392 (绝对加性), weight_noise=False
```

### 部署流程（真机 SOP）
1. **校准**：EBR 检查 ≥8 → `compass_cali`（~10 min）→ MNIST canary（预期 94-95%）
2. **权重导出**：`export_j1.py`（int8 权重 + scale + 每层 BN + test 数据）
3. **per-layer 校准**：`calibrate_real.py` 用真实激活分布拟合 alpha/beta → `calib_j1_real.json`
4. **真机推理**：`run_j1_gazelle.py`（m≤2 tiling + 反量化 + BN + alpha/beta 校准）
5. **注意**：每次 `compass_cali` 后须重做 per-layer 校准（硬件漂移）

### 关键文件
| 文件 | 作用 |
|---|---|
| `auto_research/src/qat_v5.py` | QAT v5（量化原语 + 层实现） |
| `auto_research/src/models.py` | 可参数化 MiniVGG（J1 构建） |
| `auto_research/src/runner.py` | config-driven 训练引擎 |
| `auto_research/configs/r3_J1_long.json` | 冠军配方 |
| `auto_research/runs/r3_J1_long_3b6c03f6/best.pth` | 冠军权重 |
| `Gazelle-national/mnist/export_j1.py` | 真机权重导出 |
| `Gazelle-national/mnist/run_j1_gazelle.py` | 真机推理（FAKE/HW） |
| `Gazelle-national/mnist/calibrate_j1.py` | per-layer 校准（随机/真实激活） |
| `auto_research/docs/round1-4_notes.md` | 各轮探索记录 |

---

## 四、未探索方向（诚实记录）

- **SWA/EMA**：J1 上无效（avg 95.87% < best 96.44%）；R1 的大模型上未测
- **Muon**：值得 1 个廉价 arm，但 Beyond Outliers 显示 FP 排名不迁移 QAT
- **stem 电计算 vs 全光**：用户判定 stem 电计算非关键优化，未深入
- **噪声累积感知训练**：当前 0.0392 是单层口径；多层累积特性（≈√7×）未建模，若继续迭代可试 per-layer 噪声递增
- **强增强**：已验证破坏 QAT，但若配合更长训练/噪声退火或有空间（未测）
