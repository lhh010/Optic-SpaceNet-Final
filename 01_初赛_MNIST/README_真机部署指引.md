# 01_初赛_MNIST · 目录说明

> 本目录包含初赛 MNIST 任务的**训练侧代码与文档**（三种量化方法 QAT：LSQ+ / STE / DSQ）。

## 真机部署在哪里？

**真机部署脚本不在本目录**，位于 `../03_决赛_EuroSAT真机/mnist/`（决赛期间整理的板端部署套件，与初赛权重配合使用）：

| 文件（相对 03_决赛_EuroSAT真机/mnist/） | 说明 |
|---|---|
| `run_mnist_gazelle.py` | 真机推理主脚本（raw/scale/advance 三模式，环境变量传参） |
| `w1/w2/w3_int4*.npy` | 三层 int4 权重（dsq/ste/lsqplus） |
| `test_images.npy / test_labels.npy` | 测试数据（可替换为官方提供数据） |
| **`MNIST_现场演示Runbook.md`** | **现场演示速查：连接→判据→校准→运行→数据接入→故障排查** |
| `MNIST迁移至Gazelle真机过程文档.md` | 完整迁移过程记录 |
| `DEPLOY_LOG.md` | 简版排查流水（连接链路、三大坑位根因表） |

## 本目录内容

- `docs/` — 初赛三份正式文档（设计/验证/技术数据）+ CODE_INDEX + 架构说明
- `src/` — 训练与仿真代码：`quantization/`（三种量化实现）、`models/`（QAT 模型）、`training/`、`inference/`（numpy 仿真推理）、`scripts/`（训练/对比/鲁棒性入口）

## 复现路径

训练（本地，产出 int4 权重）：`src/scripts/train_qat_{lsqplus,ste,dsq}.py` → 仿真验证：`src/scripts/compare_methods.py` → 真机部署：见上表 Runbook。