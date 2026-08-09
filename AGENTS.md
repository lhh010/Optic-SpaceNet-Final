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

## 真机共享板 SOP（2026-08-09 抢占事件后增补）

Gazelle 真机（`ssh -J huadong3564@140.206.121.211:2036 uisrc@10.102.13.37`）为多队常态共享，无错峰机制。每次开实验窗口前必须执行：

1. **他人使用侦测**：`who` + `ps aux | grep -i gazelle` + 检查 :8000 等端口占用；他队（10.102.12.4，password 认证，root 跑 server_gazelle.py）会未校准直写器件寄存器抢占光器件。发现占用→保存日志、等待释放，不动对方进程。
2. **放行判据（旧判据已失效，必须按新判据）**：EBR≥8 和 MNIST canary 对"他队负载后的物理瞬态"均不敏感（实例：EBR 9.69 + canary 98% 时窗口仍是坏的）。新判据四项**全部达标**才开窗：EBR≥8；error_std 对基线偏差 <±2%；MNIST canary 对基线 <0.5pt；EuroSAT 200 样本 mini-run 正常。
3. **被他人使用后**：物理瞬态约 40 min 自行恢复，但必须 fresh `compass_cali` + 新判据验证全部通过才重开窗；calib→跑批背靠背。
4. **窗口内对照实验用 ABA 设计**（scalar→col→scalar 各 1000），第三次重复跑专门检测窗口内漂移，单 AB 对照在窗口劣化时会拿到不可判读的数字。
5. 板端 sudo 密码 5182 已被他队现场试出（其以 root 跑 server 可直接抢占器件），建议更换并同步本文件。

事件完整归因：`eurosat_research/x0/results/C1_incident_analysis.md` + `C1_board_forensics.md`。

## Round X0（2026-08-09，首轮架构-硬件联合设计）

- 总结文档：`eurosat_research/docs/round_x0_arch_hw_codesign.md`（理论框架、A/B/C 组结论、上板 SOP、模型注册表）；过程报告 `eurosat_research/x0/results/`。
- **模型注册表 M5-M10**（`eurosat_research/weights/`，git 强制入库）：M5-M8 = 队友两阶段 v8（X0 已复测确认，无新信息量）；**M9 = `m9_j1w075ds3_v8probe15.pth`（J1 w0.75+conv3s2，1.52M，≤2M 冠军）、M10 = `m10_ds3pool3_v8probe15.pth`（+stem max3，2.56M，总冠军 96.66）——上板评测进行中（C2）**。
- v8 单阶段 160ep 为当前标准训练口径（`configs/x0*_160.json`）；Pareto 前沿与判读见 `docs/plot_pareto_v8.png`。关键反转：v8 噪声下 w200 反超 rf_s2k3，J1 已非甜点，2.5-4.6M 为甜点区间。
- 真机 SOTA：c3d + 逐列校准 **94.60%**（C1，+1.5pt）；ds3 系板端部署用 `x0/scripts/run_ds3_gazelle.py`（conv3s2 光计算），上板前必须本地 FAKE 对拍。
