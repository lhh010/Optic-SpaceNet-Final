# Optic-SpaceNet · 光学 CNN 在轨加速系统

三个 EuroSAT (10 类遥感, 64×64) CNN → **Gazelle 光计算硬件 (8×2 光学矩阵乘法器)** 的量化迁移实验代码与记录。

> 完整实验轨迹见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)；报告向总结见 [`docs/SUMMARY.md`](docs/SUMMARY.md)。

---

## 目录结构

```
.
├── src/                          # 全部 Python 代码
│   ├── _pathsetup.py             # sys.path 引导 (入口脚本自动调用, 无需手动)
│   ├── download_eurosat.py       # 数据集下载/解压/验证
│   ├── core/   optic_layers.py          # 光计算核心库 (OpticalEngine / OpticConv2d / OpticLinear)
│   ├── qat/    optic_qat*.py            # 量化训练库 (v1/v2/v3/v4/lsq)
│   ├── data/   eurosat_split.py         # 数据三分 split (单一数据源)
│   ├── training/ train_*_runner.py      # 训练器 (Phase4Trainer / MixedPrecisionTrainer)
│   │            model1_*.py, model2_*.py, model3_*.py   # 模型定义 + 训练入口
│   └── scripts/ optic_inference_*.py    # 容器推理 / QAT 交叉验证 / osimulator 评估
│                noise_robustness*.py     # 噪声鲁棒性扫描
│                plot_*.py               # 出图脚本
│                example_load_gazelle_model.py
├── demo/                         # 光计算推理演示系统 (前端 + 后端 + 远程服务)
│   ├── web/   index.html, app.js        # 单页前端 (零外网依赖, 已 vendored)
│   │         vendor/                    # Tailwind Play CDN + 等宽字体本地化
│   ├── server/ app.py                   # FastAPI 后端 (本地 FP32 + 远程光计算)
│   │          inference_local.py        # 本地推理路径 (FP32 / fake-optical)
│   │          model_trace.py            # 逐层 traced forward + 模型定义
│   │          remote_client.py          # 远程光计算 HTTP 客户端
│   │          compare.py, render.py     # 光|电逐层对比 + grid 渲染
│   │          metrics.py                # 展板静态指标
│   ├── remote/ optic_server.py          # 容器内常驻光计算推理服务 (stdlib)
│   ├── docs/   design.md               # 演示系统总设计
│   │           api.md                   # API 契约 (前后端 + 远程服务)
│   │           frontend-design.md, frontend-scroll-design.md  # 前端设计文档
│   ├── tests/                           # pytest 测试套件
│   └── deploy.sh                        # 一键部署 (同步 → 容器重启 → SSH 隧道 → 本地后端)
├── weights/                      # 全部 .pth 权重 (25 个, git 跟踪)
├── docs/                         # 文档: EXPERIMENTS / SUMMARY / TODO / PHASE4_DESIGN / OPTIC_QAT_README
│   └── figures/                  # 图表: compute_vs_accuracy_final.{png,pdf,svg}, phase_evolution 等
├── logs/                         # 运行日志: log.md, log_*.md, *.log, comparison.csv
├── data/                         # 数据集 (EuroSAT_RGB) — gitignore, 需自行下载
├── 复赛文档/                      # 复赛材料 (设计报告/验证报告/技术数据/答辩PPT)
├── .gitignore
└── README.md
```

**不在仓库中、需自行准备**:
- `data/EuroSAT_RGB/` — 数据集, 从 ModelScope 下载 (见下方数据集准备)

---

## 数据集准备

远程仓库**不包含**数据集文件 (`data/EuroSAT_RGB/` 和 `data/EuroSAT_RGB.zip` 已在 `.gitignore` 中排除)。使用前需要自行下载:

### 下载地址

[[ModelScope — EuroSAT_RGB](https://www.modelscope.cn/datasets/lhh010/EuroSAT_RGB)](https://www.modelscope.cn/datasets/lhh010/EuroSAT_RGB)
### 准备步骤

```bash
# 1. 从上方 ModelScope 链接下载 EuroSAT_RGB.zip (约 90MB)
# 2. 将 zip 文件放到项目根目录的 data/ 下
# 3. 运行解压脚本
python scripts/download_eurosat.py
```

脚本会自动:
1. 解压 `data/EuroSAT_RGB.zip` 到 `data/`
2. 智能定位图像根目录 (自动识别 `Forest`/`River` 等类别文件夹)
3. 验证数据集完整性 (图像总数、10 类别名)

**预期结果**: 27000 张 64×64 RGB 遥感图像, 10 个类别 (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`)。

**备用下载源**: [Zenodo](https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip) (脚本中已注释备用代码)。

---
## 如何运行

> ⚠️ **始终从仓库根目录运行** (CWD = 项目根目录)。脚本内部的数据/权重路径 (`data/EuroSAT_RGB`、`weights/...`) 都相对于仓库根。

每个入口脚本顶部有一段 `sys.path` 引导 (调用 `src/_pathsetup.py`)，把 `src/` 各子目录与仓库根加入 `sys.path`，因此代码里原有的扁平 import (`from optic_qat_v4 import ...`、`from optic_layers import ...`、`import osimulator`) **无需修改**即可继续工作。

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
python src/scripts/plot_phase_evolution.py         # 各 phase 精度演进
python src/scripts/plot_accuracy_bars.py           # 精度柱状图
python src/scripts/plot_eurosat_samples.py         # 数据集样本展示
python src/scripts/plot_optical_ratio.py           # 光计算占比可视化
python src/scripts/plot_six_stage_climb.py         # 六阶段爬坡图
```

### 权重

权重统一在 `weights/`，代码里已写成 `weights/<name>.pth`。如需新增权重，保存到 `weights/`。

---

## Demo 演示系统

`demo/` 目录包含一个完整的 **光计算推理演示系统**，用于向评委现场展示 Model 3 (SpaceNet V2 + KD, int8) 的真机光计算推理，并与本地 FP32 电计算逐层对比。

### 系统架构

```
浏览器单页前端 (demo/web/)
   │ HTTP
本地后端 demo/server/ (FastAPI + uvicorn)
   ├─ FP32 路径: 本地 PyTorch → forward_traced() 逐层激活/耗时
   ├─ 光计算路径: remote_client → SSH 隧道 (ssh -L 8765)
   │     → 容器内常驻 optic_server (demo/remote/, 纯 stdlib http.server)
   └─ 降级链: 远程超时/失败 → 本地 FakeOpticalEngine (同 int8 伪量化)
             → 仍失败则 503; meta.degraded 标记, 前端提示
```

### 快速启动 (本地测试, 无需远程容器)

```bash
# 从仓库根目录启动本地后端
# 自动使用 fake-optical 引擎 (本地 int8 伪量化)
uvicorn demo.server.app:app --port 8000

# 浏览器打开
# http://localhost:8000
```

启动后页面自动：
1. 从干净 test 集随机抽一张 EuroSAT 图像
2. 同时走 FP32 电计算 + int8 光计算两条路径
3. 展示逐层 6 个计算层的 feature map 光|电对比
4. 显示预测结果 + top-k 概率对比 + 指标面板

### 完整部署 (需远程 Gazelle 容器)

```bash
# 设置远程主机环境变量
export REMOTE_HOST=your-server-host

# 一键部署: 同步文件 → 重启容器服务 → SSH 隧道 → 起本地后端
bash demo/deploy.sh

# 建立 SSH 隧道后启动本地后端
uvicorn demo.server.app:app --port 8000
```

详见 [`demo/docs/design.md`](demo/docs/design.md) 和 [`demo/docs/api.md`](demo/docs/api.md)。

### 前端交互说明

1. **抽图**: 下拉选择类别 (10 类 + 随机)，点击「抽图」按钮，自动触发推理
2. **上传图片**: 支持评委上传任意图片 (自动 resize/center-crop 到 64×64)
3. **滚动叙事**: 单栏滚动浏览 6 层 stage 的光|电逐层对比 (余弦相似度、相对误差分布直方图)
4. **结果面板**: 查看 FP32 vs int8 预测对比、top-k 概率、关键指标 (光计算占比/MOPs/精度/参数)
5. **健康状态灯**: 顶栏实时显示远程引擎状态
   - 🟢 青: `gazelle-osim` 真机已连接
   - 🟡 金: `fake-optical` 降级模式 (远程不可用)
   - 🔴 红: 远程不可用

---

## 当前最佳精度

| 模型 | 参数量 | FP32 基准 | int8 val | osim 真机 | 光计算占比 | 单张耗时 |
|------|--------|-----------|----------|-----------|------------|----------|
| Model 1 (VGG) 变体 A | ~2.39M | 97.17% | 97.87% | 98.15% (q650) | 97.74% | ~150s |
| Model 1 (VGG) 变体 B | ~2.39M | 97.17% | 98.02% | 97.54% (q650) | 73.64% | ~150s |
| Model 2 (SpaceNet V1) | ~268K | 90.15% | 92.06% | **90.43%** (全量) | 90.65% | ~2.5s |
| Model 3 (SpaceNet V2+KD) | ~268K | 91.44% | 91.83% | **90.28%** (全量) | 90.65% | ~2.5s |

> 三模型当前最佳真机精度 (干净 split, Model 2/3 全量 5400): Model 2 int8 **90.43%** / Model 3 int8+KD **90.28%**; Model 1 int8 仅抽样 (q650: A 98.15% / B 97.54%)。详见 [`docs/SUMMARY.md`](docs/SUMMARY.md)。

---

## 部署推荐

| 场景 | 推荐 | 理由 |
|------|------|------|
| **轻量落地 (首选)** | Model 2 v3 int8 | 268K 参数、真机 90.43%、反超 FP32 基准 +0.28%、全量 3.7h |
| **同等轻量 + KD** | Model 3 v3 int8+KD | 与 M2 真机打平 (90.28%)，若需 KD 故事线可选 |
| **最高精度 (受限)** | Model 1 int8 变体 B | val 98.02%、q650 真机 97.54%，但 2.39M 参数 + ~150s/张 |

---

## 备注

- `osimulator/` 内的 `.so` 是 Linux x86_64 / CPython 3.9 编译产物; 在 Windows 上 `import osimulator` 会失败并回退到 `FakeOpticalEngine` (打印一条 WARN)。真实光计算评估在 Linux 容器内进行 (osimulator 已 pip 安装到 site-packages)。
- 训练/推理脚本所需的 `data/EuroSAT_RGB` 数据集未纳入 git 跟踪; 首次使用前请按 数据集准备 步骤下载。
- Demo 演示系统零外网依赖 (Tailwind Play CDN + 等宽字体均已 vendored 到 `demo/web/vendor/`)，适合现场网络不可控的环境。
