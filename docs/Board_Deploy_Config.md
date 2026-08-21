# Model 1/4/5/6/7/8/9/10 上板配置对照（收官版）

> 队伍 CICC1003564 · 2026-08-09 初版 · **2026-08-17 收官更新**（全部模型验证完成）
> 依据：`contest-national/opticspacenet/`（路径 A）、`mnist/j1_board/` + `eurosat_research/x0/`（路径 B）、
> `OpticSpaceNet迁移至Gazelle真机过程文档.md`、`x0/results/C1_col_calib_design.md` / `C1_board_results.md`、
> 板上校准产物（`opticspacenet/*.meta.npz` 实读）。

## 1. 结论速览（2026-08-17 收官版）

**全部模型上板验证已收官**，详细结果见 `contest-national/决赛文档/02_验证报告.md` 与 `opticspacenet/board_validation_20260815-17/`：

| 模型 | 路径 | 上板结果 | 状态 |
|---|---|---|---|
| M1a | B（板端）| **20/20 = 100%**（08-17）| ✅ 抽样完成 |
| M4 | A | **全量 94.19%**（gap −1.55）| ✅ 全量 |
| M5 | A | **全量 90.17%**（gap −6.3）| ✅ 全量 |
| M6 | A | **全量 92.78%**（gap −2.7）| ✅ 全量 |
| M7 | A+B | 路径 A 77.3% / 板端 81.72% | ❌ 架构不匹配（MaxPool 噪声有偏传播，判定关闭）|
| M8 | A | **全量 87.78%**（gap −8.0）| ✅ 全量 |
| M9 | B | **全量 94.43%**（canonical）| ✅ 最终交付 |
| M10 | B | **全量 95.33%**（SOTA）| ✅ 最终交付 |

<details><summary>原部署配置表（历史参考，2026-08-10）</summary>

| 模型 | 光计算层（板上实测口径） | 权重 | 校准体系 | 注册状态 | 上板状态 | 主要缺口 |
|---|---|---|---|---|---|---|
| **M1a** | 7：conv1_2…fc2（conv1_1 电算） | `baseline_vgg_phase4_v3_int8.pth` (v4.1) | A：analyze_layers per-column npz | ✅ model1a | ✅ 已上板（20 张 100%，流程级） | 单图 ~94s，全量需分段 |
| **M1b** | 6：conv3_2 也电算（revert） | `..._vB.pth` (v4.1) | A | ✅ model1b | ⏸️ 未跑 | analyze_layers 层名映射对 6 层会错标（见 §4.1） |
| **M4** | 7：stage1.0…stage3.3 + head FC；**stem FP32 电算** | `minivgg_gap_phase4_v3_int8.pth` (v4.1) | A：`m4_calib.npz` ✅（7 层实测） | ✅ model4 | ✅ 已上板（50 张 96.0%） | head 光算含 bias，需确认反量化后加回 |
| **M5** | 5：stage1.0 / stage2.0 / stage2.3 / stage3.0 / stage3.3；stem k5 / head FP32 电算（v8 语义） | `m5_j1rf_stem5_v8probe15.pth` | —（无任何板端产物） | ✅ model5（仅 40/40 张量加载验证） | ⏸️ 未上板 | ⚠️ ①head 被默认转光算（与 v8 head_fp32 训练语义冲突）；②analyze_layers 层名映射崩/错标；③stage2 3×3 只能在路径 A 跑（路径 B 无普通 3×3 支持）；④无 FAKE 对拍 |
| **M6** | 5：同上（stem k3）；v8 语义同 M5 | `m6_j1_v8probe15.pth` | — | ✅ model6（同上） | ⏸️ 未上板 | 同 M5 ①②④；可参考 c3d 部署（head 光算 h1/h2 有 bias 处理） |
| **M7** | 5：同上（channels 12/24/48/96） | `m7_j1w075_v8probe15.pth` | — | ❌ **未注册**（任务 C） | ⏸️ | 注册 + ①②④ |
| **M8** | 5：同上（stem k5） | `m8_rf_stem5_v8probe15.pth` | — | ❌ **未注册**（任务 C） | ⏸️ | 注册 + ①②④ |

> 注：模型 1/4 的历史"7 光计算层"均含 head FC；M4 的 stem 为 FP32 电计算
> （`retrain_v41_m4.log`：QAT Conv 6 enabled + 1 fp32 first layer；与 `m4_calib.npz` 7 层吻合）。

</details>

> **原 §4.3 部署缺口四项已全部关闭**（head 电算 convert_linear=False、analyze_layers 层名分支、model7/8 注册、FAKE 对拍）；M1 板端由队友完成（export_m1 / run_m1_gazelle / probe_dump_m1 / calibrate_any_m1，修复 conv im2col 缺失与行采样）；M7 板端深诊断产物（probe_m7v2 / board_m7_diag / m7_ccic.sh）见 `board_validation_20260815-17/`。

## 2. 两套部署路径

| | 路径 A：opticspacenet HTTP | 路径 B：J1 板端 numpy |
|---|---|---|
| 推理侧 | 本地 Windows torch（OpticConv2d/OpticLinear 全复用）+ 板上 `server_gazelle.py` HTTP | 板上纯 numpy runner（`mnist/j1_board/run_j1_gazelle.py`；ds3：`x0/scripts/run_ds3_gazelle.py`） |
| 模型 | model1a/1b/2/3/4/5/6（MODEL_REGISTRY） | c3d/J1 系（`deploy_c3d`、`weights_w075ds3` 等 export 包：npy+meta.json） |
| 校准 | `analyze_layers.py` → per-column affine npz（权重 md5 索引，raw MAC 域修正 `y=(y_hw−b_j)/a_j`） | 标量 calib json 或 **逐列 calib json**（`calibrate_col.py` 产物，折叠进反量化；`J1_CALIB_COL` / `DS3_CALIB_COL`） |
| tiling | m≤1024 分块（文档实测 2048 可跑） | **m≤2 分块**（FPGA m≥3 行回绕 bug，ds3 runner 注释） |
| REP | REP=4 平均（性价比拐点） | 无 REP（单次 matmul） |
| 真机成绩 | M4 50 张 96.0%；M2 gap −0.5~−3.0 | **c3d + 逐列 calib 94.60%（C1，n=1000）**；全量 SOTA = M10 95.33% |

## 3. 两套校准体系对照（analyze_layers vs calibrate_col）

| 项 | analyze_layers（路径 A） | **calibrate_col**（路径 B，C1 SOTA） |
|---|---|---|
| 拟合 | 逐列（per-output-channel）仿射 a_j/b_j，raw MAC 域 | 逐列 α_c/β_c，`y_f = x_scale·(ws_c/α_c)·(y_hw − β_c − α_c·x_zp·col_sum_c)` 折叠进反量化（零额外算子） |
| 数学 | 等价（都是 per-column affine，仅注入位置不同） | 同左 |
| 自检 | ❌ 无 SE / 结构 SNR / 留出验证 | ✅ SE(α,β) + 结构 SNR（实测 α 3–23、β 7–51 全 >1）+ 50/50 留出验证（方差改善 4.5–25.2%/层） |
| 纪律 | 坑③：~1h 量级漂移 → 分段再校准（run_full.sh 每段开头 40 张重拟合） | **同窗口背靠背**：probe pairs → calibrate_col → 跑批连续完成；calib json **不可跨窗口复用**（w1 vs w2 留出改善 −24.4%→−7.8% 实证） |
| 覆盖 | 按模型分支写死层名（model1/model4/else） | 任意层列表（`CALIB_COL_LAYERS`），缺失层回退标量 |
| 产物 | calib.npz + .meta（权重 md5 → 层名） | calib_col.json（板上 `/home/uisrc/j1/`，**未入库 git**） |

**结论**：两者数学等价，差异在"自检 + 同窗口纪律"。路径 A 现无逐列自检流程，路径 B 的
calibrate_col 是 94.60% SOTA 的支撑；M5-M8 上板前应决策：走 A（补自检）还是走 B（export 权重，
但 M5 的 stage2 3×3 需要给 runner 加普通 3×3 conv 支持，目前 B 只有 1×1 / conv3s2）。

## 4. 逐模型细节与缺口

### 4.1 Model 1（a/b）
- 板上实测校准产物 `m1_calib.npz`：7 层 conv1_2/conv2_1/conv2_2/conv3_1/conv3_2/fc1/fc2（n_col 32/64/64/128/128/256/10），与文档 §1 一致
- 变体 B：`revert_optic_to_conv2d(model, "conv3_2")` → 6 光计算层；⚠️ `analyze_layers.py` 的 model1 分支
  固定 7 个名字，6 层时 `[:n_layers]` 截断会把第 6 层错标为 conv3_1（md5 索引不受影响，仅日志/归档名错）
- 性能：单图 ~94s（fc1 k=8192）→ 20 min 窗口 ≈ 12 张，全量 5400 需大量分段
- 校准产物漂移实证（同权重 model2，跨文件）：fc1 b_mean 11257→6876→6609（不同窗口差 ~4000 counts）、
  a_mean 1.0099→1.0189 → 印证"校准不可复用"，20 min 规则必要性

### 4.2 Model 4
- 训练（`retrain_v41_m4.log`）：stem FP32 电计算（QAT Conv 6 + 1 fp32）+ head Linear QAT（光算）
- 板上校准产物 `m4_calib.npz`：7 层 stage1.0/…/stage3.3/head（n_col 48/48/72/72/96/96/10）✅ 与训练一致
- head Linear(96,10) **有 bias**：光算部署必须"反量化后加 bias"（C2 部署 bug 教训；板上
  `run_j1_gazelle.py` 的 head bias 支持是手动合并的补丁，路径 A 的 OpticLinear 需确认同一语义）
- 真机：50 张 96.0%（v4.0 权重）；v4.1 权重即部署版，下次上板重采校准

### 4.3 Model 5 / 4.4 Model 6（J1 家族，v8）
- 文档口径（X0 注册表 + J1Arch docstring）：光计算层 = stage1.0/stage2.0/stage2.3/stage3.0/stage3.3 共 **5 conv**，
  stem/head **FP32 电计算**（v8 `stem_fp32`/`head_fp32` 训练语义）
- ⚠️ **缺口① head 转换**：`gazelle_engine.build_model` 调 `build_optical_model` 未传 `convert_linear=False`，
  而默认 `convert_linear=True` → head.2/head.4 会被转成 OpticLinear（光算）→ 与 v8 head_fp32 训练语义冲突。
  需显式决策：head 电算（对齐 v8 训练，改 build_model）或 head 光算（对齐 c3d 部署 94.60% 经验，
  需逐列 calib 覆盖 h1/h2 + bias 处理）
- ⚠️ **缺口② 校准层名**：`analyze_layers.py` 无 model5/6 分支，落 else（5 个名字）；若 5 层 → 名字错标
  （stage2.0 标成 stage2），若 head 被转光算 7 层 → `names[li]` IndexError 直接崩
- ⚠️ **缺口③ M5 stage2 3×3**：路径 B runner 只支持 1×1（optical_conv1x1）与 conv3s2（ds3），
  **无普通 3×3 光计算实现** → M5 只能走路径 A（OpticConv2d 通用 im2col）；M6/M7/M8 全 1×1，两条路径皆可
- ⚠️ **缺口④ 验证深度**：model5/6 只有 `_verify_deploy.py` 的 40/40 张量加载 + 形状检查，
  **无本地 FAKE 对拍、无板上校准产物、无跑批**（X0 SOP §2.5 要求新 runner 上板前 FAKE 对拍）
- M6 参考点：c3d 部署（同架构）head 光算 h1/h2 + 标量/逐列 calib + bias，真机 93.80/94.60%

### 4.5 Model 7 / 4.6 Model 8
- 权重已入库（`train-test/weights/m7_j1w075_v8probe15.pth` / `m8_rf_stem5_v8probe15.pth`，X0 复测 94.98/96.20）
- ❌ MODEL_REGISTRY 无 model7/model8（任务 C）；J1Arch 参数化已支持：
  M7 = `J1Arch(channels=(12,24,48,96), kernels=(1,1,1), stem_kernel=3)`
  M8 = `J1Arch(channels=(16,32,64,128), kernels=(1,1,1), stem_kernel=5)`
- 注册后受 §4.3 全部缺口约束；M8 与 M6 同构（仅 stem k5），M7 通道更窄（光阵列列数少）

## 5. 20 分钟校准节奏（2026-08-09 用户指示，硬性规则）

- **规则**：连续跑批每 ~20 min 必须重新校准一次，否则准确率显著下降（时漂比此前"~1h 量级"（坑③）更快）。
- **动作**（二选一，按所走路径）：
  - 路径 A：`bash run_calib.sh MODEL=modelX LIMIT=40 BATCH=8 REP=4 CALIB_OUT=calib_<ts>.npz`
    （~分钟级）→ 立即 `run_client.sh ... CORRECTION=calib_<ts>.npz` 继续下一段
  - 路径 B：`probe_dump*.py` 重采 pairs → `calibrate_col.py`（<1 min）→ 跑批；calib json 同窗口不可复用
- **计划影响**：跑批段 ≤20 min/段（原 SOP 上板 M4→M5→M6 每段 ≤25 min 需收缩）；ABA 对照
  第三次重复（A′）同时承担窗口内漂移检测；段间校准与跑批背靠背（同窗口）。
- 佐证：C1 板上 error_std 小时级波动（4.67→6.09→4.92）；A′−A = +1.0pt（~1h 窗口内漂移）；
  同权重跨窗口 b 漂移 ~4000 counts（§4.1 实测）；用户实测 20 min 后精度显著下降。

## 6. 下次上板前置清单（本轮不做）

- [ ] M5/M6：决策 head 电算 or 光算 → 相应改 `gazelle_engine.build_model`（`convert_linear=False` 或保留+补 h1/h2 calib）
- [ ] `analyze_layers.py`：加 model5/6/7/8 层名分支（stage1.0/stage2.0/stage2.3/stage3.0/stage3.3，5 层）
- [ ] M7/M8 注册 MODEL_REGISTRY（J1Arch 参数化，任务 C）
- [ ] 本地 FAKE 对拍：`BACKEND=numpy` 跑 model5/6（+7/8）确认与训练前向一致（X0 SOP §2.5）
- [ ] 上板：四判据放行 → fresh 校准 → 跑批段 ≤20 min，段间 run_calib 背靠背
- [ ] 校准自检（可选增强）：路径 A 补 SE/留出验证输出，对齐 calibrate_col 的质量门槛
