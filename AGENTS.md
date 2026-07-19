# Optic-SpaceNet 光计算迁移项目

EuroSAT (10 类遥感，64×64) CNN → Gazelle 8×2 光计算加速器量化迁移。

## 关键事实

- **本地仓库**: `/Users/ms.chen/Projects/2607-ciciec/Ltsimulator-test`
- **远程主机**: `$REMOTE_HOST`（已配置 Docker context `$REMOTE_HOST`）
- **Docker 镜像**: `lightelligence.docker:lt-simulator_v1.4.6`
- **可用容器**: `gazelle_sim`
- **废弃容器**: `lt-simulator`（已删除；无 license 文件，报 `license error: 276`）
- **容器内 Python**: `/local/miniconda/envs/moca_llm/bin/python`
- **容器内 simulator**: `import osimulator; from osimulator.api import load_gazelle_model`
- **数据目录**: `data/EuroSAT_RGB`（EuroSAT RGB 数据集，需下载）
- **光计算核心参数**: 8×2 tile，8a8w12o，输入需非负，只支持 MAC（无 bias）

## 远程路径约定

仓库建议同步到：

```
/mnt/data/personal/mschen/Projects/2607-ciciec/Ltsimulator-test
```

数据放在仓库下的 `data/EuroSAT_RGB`。

## 常用命令

切换 Docker context:
```bash
docker context use $REMOTE_HOST
```

进入容器（调试用）:
```bash
docker exec -it gazelle_sim bash
```

在容器内运行目标脚本:
```bash
cd /workspace/Ltsimulator-test
/local/miniconda/envs/moca_llm/bin/python optic_inference_mixed_model1.py --quick 100
```

使用 tmux 跑长任务:
```bash
docker exec -it gazelle_sim bash -c "
  cd /workspace/Ltsimulator-test &&
  /local/miniconda/envs/moca_llm/bin/python optic_inference_mixed_model1.py --quick 100
"
```

## 性能预期

- Model 1（VGG）在光模拟器上极慢：约 150 秒/图像。
- `--quick 100` 预计约 4 小时。
- osimulator 单个大 GEMM（如 3136×756×64）耗时约 80 秒，会占用约 90 个 CPU core。
- 宿主机 `$REMOTE_HOST`：256 核 Xeon 6980P，377GB RAM。

## 关键文件

| 文件 | 说明 |
|------|------|
| `model1_baseline.py` / `model2_spacenet_v1.py` / `model3_spacenet_v2.py` | FP32 基线训练 |
| `model*_phase4_v3.py`, `model*_mixed.py`, `model*_lsq.py` | Phase 6 / 最终方案 |
| `optic_qat_v3.py`, `optic_qat_v4.py`, `optic_qat_lsq.py` | QAT 量化模块 |
| `optic_layers.py` | 光计算推理层（OpticalEngine, OpticConv2d, OpticLinear） |
| `optic_inference_*.py` | 在 osimulator 或 QAT 模式下评估 |
| `EXPERIMENTS.md`, `PHASE4_DESIGN.md`, `OPTIC_QAT_README.md` | 实验记录与设计文档 |

## 注意

- 容器内默认 Python 找不到 `osimulator`，必须使用 `/local/miniconda/envs/moca_llm/bin/python`。
- 第二个 simulator 容器无 license，需要重新用 SN 激活才能使用。
- 模型权重文件较大，同步或复制到容器时需注意空间。
