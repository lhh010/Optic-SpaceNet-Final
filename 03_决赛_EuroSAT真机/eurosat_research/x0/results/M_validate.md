# M-validate — M5-M8 两阶段 v8 独立复测报告

日期：2026-08-09 · 任务：对队友报告的 M5 96.65 / M6 95.22 / M7 95.00 / M8 96.26 做独立复测（test 5400 clean，qat eval 形态），并核对 MACs。

## 0. ckpt 定位（重要前提）

- 容器 `gazelle_sim` 的 `auto_research/runs/`（99 个 run）、`logs/`、`configs/` 中**不存在任何 M 系 run 目录/日志**；本地 `eurosat_research/runs/` 同样没有。
- 实际最终 ckpt 在**本地** `eurosat_research/weights/`（队友在 Windows 机器训练，config `data_dir=E:/...`，仅 weights + configs 同步回本地）：
  - `weights/m5_j1rf_stem5_v8probe15.pth`（MD5 4f432c02…）
  - `weights/m6_j1_v8probe15.pth`（MD5 aee72b48…）
  - `weights/m7_j1w075_v8probe15.pth`（MD5 e52d2bd3…；注意 m7 的 clean 阶段 pth 缺失）
  - `weights/m8_rf_stem5_v8probe15.pth`（MD5 a02a1df3…；与 `m8_rf_stem5_clean.pth` 同尺寸但 MD5 不同，非误复制）
- 架构/训练配置来源：`eurosat_research/README.md` §Model 5-8 + `configs/m{5,6,7,8}_*_v8probe15.json`。M7=w075、M6=J1(head128)、M8=rf_stem5、M5=rf_s2k3+stem k5。
- 因此**无 run 内 eval 脚本/日志可交叉参照**，本报告以独立 eval 为唯一依据。

## 1. eval 口径（与 x0r_* 完全相同）

脚本：`eurosat_research/scripts/eval_m_ckpt.py`（本次新建，已同步容器同名路径）。
流程：`build_model(cfg)` → `prepare_model_v8`（同 config 的 probe 标定；`model.eval()` 下 v8 所有噪声分量 `if self.training` 门控 = off，量化保留：weight int8 per-channel + 输入 uint8 affine + ADC 12-bit 输出量化）→ 载入 ckpt → `evaluate_full(test_loader)`（seed 42，test 全量 5400）。

**管线 sanity**：同脚本复测 `runs/x0r_rf_stem5_160_62eb06cb/best.pth` → **95.8889%**，与该 run summary.json 的 test acc 完全一致（95.89）；MACs 2163968 一致。口径验证通过。

执行命令（容器 gazelle_sim GPU0，双卡空闲）：
```bash
cd /workspace/Ltsimulator-test/auto_research && \
/local/miniconda/envs/moca_llm/bin/python scripts/eval_m_ckpt.py \
  --config configs/<m>.json --ckpt weights/<m>.pth --gpu 0 \
  --data-dir /workspace/Ltsimulator-test/data/EuroSAT_RGB
```
四个 ckpt 均 40/40 tensors 全量加载，无 missing/unexpected key。

## 2. 复测结果对照表

| 模型 | 架构 | 报告 acc | **复测 acc** | 差异 | 报告 MACs | 实测 MACs | MACs 核对 |
|---|---|---|---|---|---|---|---|
| M5 | rf_s2k3 + stem k5 | 96.65 | **96.61** | −0.04 | 5.31M | 5,309,696 | ✓ |
| M6 | J1（head128） | 95.22 | **95.28** | +0.06 | 1.38M | 1,377,536 | ✓ |
| M7 | w075（C0=12） | 95.00 | **94.98** | −0.02 | 0.86M | 861,440 | ✓ |
| M8 | rf_stem5 | 96.26 | **96.20** | −0.06 | 2.16M | 2,163,968 | ✓ |

（复测 macro-F1：M5 96.53 / M6 95.10 / M7 94.79 / M8 96.09；params：100,250 / 50,330 / 32,246 / 51,098。）

**结论：四个报告数字全部确认**（偏差 ≤0.06pt，在 val/test 波动与 ckpt 选择差以内）；MACs 四档全部核对一致。Pareto 图（`docs/plot_pareto_v8.py` 红 X）无需修正。

## 3. M 系 vs X0 同档模型：信息量判读

| 档位 | M 系（两阶段） | X0 单阶段同档 | 差值 | 判读 |
|---|---|---|---|---|
| 0.86M | M7 94.98 | x0r_w075 94.65 | +0.33 | 种子噪声量级内（±0.35~0.45） |
| 1.38M | M6 95.28 | x0_ctrl(J1) 95.57 | −0.29 | 种子噪声量级内 |
| 2.16M | M8 96.20 | x0r_rf_stem5 95.89 | +0.31 | 种子噪声量级内 |
| 5.31M | M5 96.61 | （无同架构） | — | 比 x0r_rf_s2k3（4.52M, 96.39）+0.22 但贵 0.79M；**被 x0_ds3pool3（2.56M, 96.66 mean）以一半 MACs 严格支配** |

**判读：M 系不提供任何超出 X0 已有架构的信息量。** 三个可对照档位与 X0 单阶段的差异全部落在种子噪声量级（参考 dsconv3 两 seed 差 0.45pt），两阶段协议（clean160→v8 60ep）相对单阶段 v8 160ep 既无稳定收益也无稳定损失——协议优劣需多种子才能定论，单 seed 下应视为等价。M5 作为唯一新架构点（rf_s2k3+stem5），精度被 MACs 一半的 ds3pool3 追平/反超，不上前沿。

## 4. 副作用与说明

- 拷入容器的文件（均为本地仓库同名文件的镜像，未覆盖任何已有文件）：`auto_research/configs/m{5,6,7,8}_*_v8probe15.json`、`auto_research/weights/m{5,6,7,8}_*_v8probe15.pth`、`auto_research/scripts/eval_m_ckpt.py`。
- 容器 src 与本地 src 逐文件 md5 一致（runner/models/qat_v8/metrics/config；data.py 仅 data_dir 默认值差异，eval 用 `--data-dir` 显式覆盖）。
- 未动 runs/ 他人目录，未 git commit，未触碰开发板。
