# LT-Simulator · 光计算迁移实验

三个 EuroSAT (10 类遥感, 64×64) CNN → **Gazelle 光计算硬件 (8×2 光学矩阵乘法器)** 的量化迁移实验代码与记录。

> 完整实验轨迹见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)；报告向总结见 [`docs/SUMMARY.md`](docs/SUMMARY.md)。

---

## 目录结构 (2026-07-16 重构)

```
train-test/
├── src/                       # 全部 Python 代码
│   ├── _pathsetup.py          # sys.path 引导 (入口脚本自动调用, 无需手动)
│   ├── core/      optic_layers.py            # 光计算核心库 (OpticalEngine / OpticConv2d / OpticLinear)
│   ├── qat/       optic_qat*.py              # 量化训练库 (v1/v2/v3/v4/lsq)
│   ├── data/      eurosat_split.py           # 数据三分 split (单一数据源)
│   ├── training/  train_*_runner.py          # 训练器 (Phase4Trainer / MixedPrecisionTrainer)
│   │              model1_*.py, model2_*.py, model3_*.py   # 模型定义 + 训练入口
│   └── scripts/   optic_inference_*.py       # 容器推理 / QAT 交叉验证 / osimulator 评估
│                  noise_robustness*.py       # 噪声鲁棒性扫描
│                  plot_compute_vs_accuracy.py
│                  example_load_gazelle_model.py
├── weights/                   # 全部 .pth 权重 (25 个, git 跟踪)
├── docs/                      # 文档: EXPERIMENTS / SUMMARY / TODO / PHASE4_DESIGN / OPTIC_QAT_README / 复赛-test
│   └── figures/               # 图表: compute_vs_accuracy_final.{png,pdf,svg}, noise_robustness*.png
├── logs/                      # 运行日志: log.md, log_*.md, *.log
├── data/                      # 数据集 (EuroSAT_RGB) — 原地保留, gitignore
├── osimulator/                # Gazelle 光计算仿真器 (外部 vendored 包) — 原地保留, gitignore
├── 初赛文档/                   # 初赛材料 (PDF) — 原地保留, gitignore
└── README.md
```

**原地保留、未迁移**: `data/` (脚本用 CWD 相对路径 `data/EuroSAT_RGB` 读取)、`osimulator/` (外部 vendored 包, `import osimulator` 依赖其内部结构)、`初赛文档/`、`.obsidian/`、`.claude/`。

---

## 如何运行

> ⚠️ **始终从仓库根目录运行** (CWD = `train-test/`)。脚本内部的数据/权重路径 (`data/EuroSAT_RGB`、`weights/...`) 都相对于仓库根。

每个入口脚本顶部有一段 3 行的 `sys.path` 引导 (调用 `src/_pathsetup.py`)，把 `src/` 各子目录与仓库根加入 `sys.path`，因此代码里原有的扁平 import (`from optic_qat_v4 import ...`、`from optic_layers import ...`、`import osimulator`) **无需修改**即可继续工作。

### 训练 (干净三分 split: train 16200 / val 5400 / test 5400)
```bash
python src/training/model1_baseline_phase4_v3.py --variant A   # Model 1 int8
python src/training/model2_spacenet_v1_phase4_v3.py            # Model 2 int8
python src/training/model3_spacenet_v2_phase4_v3.py            # Model 3 int8 + KD
```

### 容器推理 / 评估
```bash
# QAT 伪量化交叉验证 (秒级, 不需要 osimulator)
python src/scripts/optic_inference_int8.py        --qat --batch 256   # Model 2 int8
python src/scripts/optic_inference_int8_model1.py --variant A --qat --batch 256
python src/scripts/optic_inference_kd.py          --qat --batch 256   # Model 3 (注: --qat 走 int4)

# osimulator 真实光计算硬件仿真 (容器内, 小时级)
python src/scripts/optic_inference_int8.py                    # Model 2 全量 5400
python src/scripts/optic_inference_kd.py                      # Model 3 全量 5400
python src/scripts/optic_inference_int8.py --quick 500        # 抽样
```

### 出图
```bash
python src/scripts/plot_compute_vs_accuracy.py     # 输出到 docs/figures/
```

### 权重
权重统一在 `weights/`，代码里已写成 `weights/<name>.pth`。如需新增权重，保存到 `weights/`。

---

## 备注
- `osimulator/` 内的 `.so` 是 Linux x86_64 / CPython 3.9 编译产物; 在 Windows 上 `import osimulator` 会失败并回退到 `FakeOpticalEngine` (打印一条 WARN)。真实光计算评估在 Linux 容器内进行 (osimulator 已 pip 安装到 site-packages)。
- 三模型当前最佳真机精度 (干净 split, 全量 5400): Model 2 int8 **90.43%** / Model 3 int8+KD **90.28%**; Model 1 int8 仅抽样 (q50: A 98% / B 100%)。详见 `docs/SUMMARY.md`。
