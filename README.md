# Optic-SpaceNet 决赛提交包 · CICC1003564

> 干净的决赛提交材料合集：初赛 / 复赛 / 决赛三阶段的光计算真机代码与文档。
> 组包日期：2026-08-21。各阶段详情见对应子目录 README / 文档。

## 📺 演示视频（B 站）

- 决赛 PPT 讲解视频：<https://www.bilibili.com/video/BV13A8z6zEJm>

## 目录结构

```
决赛提交包/
├── README.md                ← 本文件
├── 01_初赛_MNIST/            ← MNIST 三层 MLP 光计算真机迁移（LSQ+/STE/DSQ）
│   ├── docs/                 ← 初赛三份文档（设计/验证/技术数据）+ 迁移过程文档
│   └── src/                  ← 训练、量化、部署脚本（含 run_mnist_gazelle.py 等）
├── 02_复赛_EuroSAT仿真/       ← EuroSAT CNN → osimulator 仿真迁移（M1-M3）
│   ├── 文档/                 ← 复赛三份文档（加封面版 PDF + MD）
│   ├── src/                  ← OpticConv2d/OpticLinear 映射层、QAT v3/v4、训练与推理
│   ├── weights/              ← M1/M2/M3 int8 权重
│   └── osimulator/           ← 官方仿真器（GAZELLE_ARCHITECTURE.md 等）
├── 03_决赛_EuroSAT真机/       ← EuroSAT 模型族 M1/M4-M10 Gazelle 真机全量验证
│   ├── 文档/                 ← 决赛三份文档（加封面版 PDF + MD）+ 答辩 PPT/讲稿
│   ├── eurosat_research/     ← 架构搜索（R1-R8）、v8 QAT 训练、X0 联合设计、全部权重
│   ├── opticspacenet/        ← 真机部署脚本、逐列校准、全量跑批日志与错误明细
│   ├── mnist/j1_board/       ← 板端校准工具（calibrate_col.py / probe_dump.py）
│   └── crossval/             ← 真机-仿真器噪声配对实验（E0-E7）
└── docs/                     ← Model_Overview / Board_Deploy_Config 等全局参考
```

## 快速开始

### 环境要求
- 本地：Windows / Linux，Python 3.9 + PyTorch 2.8（CPU 即可训练与 FAKE 对拍）
- 板端：Gazelle 真机（Python 3.6 + compass_sdk 1.0.2，root），经跳板机 SSH 访问
- 数据：EuroSAT_RGB（27000 张，ModelScope: lhh010/EuroSAT_RGB）；MNIST（初赛，已含）

### 决赛主链路（03_决赛_EuroSAT真机）

1. 训练（v8 噪声鲁棒 QAT）：`eurosat_research/configs/x0*_160.json` + `eurosat_research/src/qat_v8.py`
2. 导出 int8 + FAKE 数值对拍：`eurosat_research/x0/scripts/export_ds3.py`
3. 真机逐列校准：`mnist/j1_board/`（probe_dump → calibrate_col.py，同窗口背靠背）
4. 真机全量推理：`opticspacenet/` run 脚本（路径 A HTTP / 路径 B 板端直跑）
5. 上板放行判据与调度纪律：见 `03_决赛_EuroSAT真机/文档/02_验证报告.md` §2

### 复赛链路（02_复赛_EuroSAT仿真）

`src/core/optic_layers.py`（光计算映射）→ `src/qat/optic_qat_v4.py`（int8 QAT）→ 容器内 osimulator 全量验证。

### 初赛链路（01_初赛_MNIST）

`src/` 下训练三种量化方法（LSQ+/STE/DSQ）→ 导出 int4 权重 → 真机部署（10000 张全量，gap ≤0.1pt）。

## 关键结果速览

| 阶段 | 任务 | 口径 | 结果 |
|---|---|---|---|
| 初赛 | MNIST 10 类 | 真机 10000 张 | LSQ+ 97.35% / STE 96.43% / DSQ 94.79%（与软件参考 gap ≤0.1pt）|
| 复赛 | EuroSAT 10 类 | osim 仿真全量 5400 | M2 90.43% / M3 90.28% |
| 决赛 | EuroSAT 10 类 | **真机全量 5400** | **M10 95.33%（SOTA，gap −1.43pt）/ M9 94.43%** |

## 注意事项

- 复赛/决赛 EuroSAT 数据集未随包分发（27000 张），从 ModelScope `lhh010/EuroSAT_RGB` 下载后放入各阶段 `data/` 目录；初赛 MNIST 已包含。
- 讲解视频不随包分发，见顶部 B 站链接。
- 部署脚本禁位置参数（compass_sdk 会篡改 sys.argv），一律环境变量传参；sudo 用 `sudo env VAR=...`。
- FPGA 存在 m≥3 行回绕 bug，`optical_mm` 已固化 m≤2 tiling——不要改。
- 共享板 SOP（放行判据四项全过 / 窗口化调度 / 20min 新鲜校准）见决赛验证报告 §2，上板前必读。