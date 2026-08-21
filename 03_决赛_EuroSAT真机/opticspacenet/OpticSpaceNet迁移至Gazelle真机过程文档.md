# Optic-SpaceNet（复赛）迁移至 Gazelle 光计算真机 · 完整过程文档

> 队伍：CICC1003564（复旦大学 · 华东赛区）
> 完成日期：2026-08-07
> 状态：✅ **四模型（Model 1a / 2 / 3 / 4）光电分离 HTTP 架构迁移完成，小样本真机验证全部通过**
> 对应初赛文档：`contest-national/mnist/MNIST迁移至Gazelle真机过程文档.md`（本任务大量复用其经验，新增 4 个新坑见 §6）

---

## 1. 背景与目标

复赛项目 Optic-SpaceNet 的三个模型原先只在**容器 osimulator 仿真**上验证。本任务把它们的全部光计算层迁移到**远程 Gazelle 真机**（8×2 光学矩阵乘法器，8a8w12o）。

| 模型 | 名称 | 权重 | 光计算层 | osimulator 全量基准 |
|---|---|---|---|---|
| Model 1 | Baseline VGG（flat+BN） | `baseline_vgg_phase4_v3_int8.pth`（变体 A）/ `_vB`（变体 B） | 7 个（6 Conv + fc1 + fc2，首层电计算） | 全量 ~9 天不可行（q50 抽样 100%） |
| Model 2 | SpaceNet V1（硬件对齐） | `spacenet_v1_phase4_v3_int8.pth` | 5 个（stage1/2/3 + fc1/fc2，stem 电计算） | **90.43%**（5400 张） |
| Model 3 | SpaceNet V2 KD | `spacenet_v2_phase4_v3_int8.pth` | 5 个（同 Model 2 架构） | **90.28%**（5400 张） |

- 量化：全部 int8（input uint8 / weight int8，Gazelle 原生 8a8w12o），bias=False，BN 保留
- 目标：真机光计算路径准确率接近 osimulator 基准，误差可控、有据可查；小样本验证按用户约定（Model 1 因计算量只跑 20 张）

## 2. 总体架构（方案 A：光电分离的 HTTP 远程调用）

```
【本地 Windows / WSL】(torch 电计算 + 客户端)
    │  stem/BN/ReLU/Pool 电计算；光计算层 im2col→量化→HTTP POST
    │  SSH 隧道  ssh -L 8000:10.102.13.37:8000 -J 跳板机 ...
    ▼
【Gazelle 真机】(root)  server_gazelle.py  ──compass_sdk──▶ 光芯片
    POST /matmul {"act":uint8, "weight_id"} → {"data": MAC}
```

**关键决策**：板上无 torch → 光计算 matmul 做成 HTTP 服务（stdlib，参照官方 `compass_server.py`），本地用 torch 完整复用 `optic_layers.py` 的 `OpticConv2d/OpticLinear/build_optical_model`（与基线**同一条前向路径**），仅把 `OpticalEngine._matmul_real / matmul_pre_quantized` 两处 osimulator 调用替换为 2D 后端（compass 语义 `(m,k)@(k,n)`，osimulator 是 3D batch 语义，取 `(b*m,k)` 展平 + 按行切块即可）。

## 3. 硬件事实（实测确认，含与 MNIST 文档不同的新发现）

| 项 | 实测值 | 说明 |
|---|---|---|
| EBR | **9.73 / 9.66** ✅ | 官方 `evb_test_sample.py` 实测，校准状态良好（与 MNIST 2026-08-06 校准一致） |
| tia_gain_scale_factor | **255.0** | SDK `global_var` 读取；SDK 已把 12-bit ADC 读数乘回 MAC 单位 |
| `compass_matmul(vec,w)` 语义 | **≈ (vec@w) 整数 MAC 单位** | ⚠️ 与直觉不同：**不要**再除以 tia_gain（除以 255 会得到 0.4% 的相对值，曾误导初测） |
| 规模 | m>1024（2048 实测可跑）、k=1024、n=256 均可 | 客户端仍按 1024 行切块保守调用 |
| 速度 | fc1(1024×256) ≈ 0.75s/次；Model 1 单图 ~94s（k 达 8192） | 全量 5400 需分段批量 |

## 4. 完整执行过程（时间线）

### 4.1 环境准备与连接（复用 MNIST 经验）
- 本地 `SSH_ASKPASS` 方案（`scratch/sshpass_ask.sh`）+ **`ssh -J` 两跳直达**（`-J huadong3564@140.206.121.211:2036 uisrc@10.102.13.37`），比文档记载的两段式 scp 更简单，`scp -J` 同步可用。
- WSL→Windows 环境变量转发：Windows exe 不继承 WSL env，必须 `WSLENV=KEY1:KEY2 ...`；`KMP_DUPLICATE_LIB_OK=TRUE` 解决 torch+numpy OpenMP 重复加载。
- 隧道用托管后台任务常驻（普通 `&` 进程随工具调用退出而死）。

### 4.2 硬件体检
- `evb_test_sample.py` → **EBR 9.73/9.66**（≥8 ✅），无需重跑 `compass_cali`。
- 诊断脚本确认 SDK 可导入、`compass_init(150)` 正常。

### 4.3 硬件特性探测（三个探针脚本）
1. `diag_gazelle.py`：全量程随机 matmul，初测 rel err ≈0.996 → 误以为差 0.4%；实为**错误地除以了 tia_gain**（SDK 已返回 MAC 单位）。
2. `probe_gazelle.py`：小已知用例（期望 [36,-36] 硬件返回 [1020,765] → 小数值 SNR 差，符合预期）；大随机拟合 → `hw ≈ 1.026×ref + b`。
3. `probe2_gazelle.py`：逐通道拟合 → **增益 a_j∈[0.94,1.06]（逐通道 std 0.03-0.13）、偏移 b_j≈(20~50)·k、重复性差 2-6k**。
- **关键发现**：官方 EBR 测试用零均值数据集，**测不出增益/偏移系统误差**（见 §6 坑①）。

### 4.4 本地基线复现（不动真机，验证数据/权重/前向路径忠实性）
- `eurosat_loader.py`：PIL 复刻 torchvision ImageFolder 排序 + `eurosat_split` test 索引 → 与 90.43% 基线**同 5400 张测试集**（已核对仓库数据 zip 与目录一致，类计数 3000/2500/2000 等为队伍自制数据集）。
- 本地 numpy 干净参考（同一 `_matmul_real` 量化路径、无噪声）：全量 5400 张 **90.78%**（BATCH=1）/ **90.70%**（BATCH=8，批量共享量化代价仅 −0.08pt）→ 与 osimulator 90.43%（含仿真噪声）、QAT 92.20% 自洽，**证明迁移路径忠实于基线**。

### 4.5 服务与客户端实现
- 板上 `server_gazelle.py`：stdlib HTTP，`compass_init(150)` 后常驻 8000 端口，`POST /matmul`（act uint8 + weight_id）→ `{"data": MAC}`；`/health` 上报 tia_gain/调用数。
- 本地 `gazelle_engine.py`：`GazelleOpticalEngine(OpticalEngine)` 重写 `_matmul_real`/`matmul_pre_quantized` 走 2D 后端（`HttpBackend`/`NumpyBackend`），逐通道修正 `y=(y_hw−b_j)/a_j`（按权重 md5 索引）；`MODEL_REGISTRY` 支持四档模型。
- 工程优化：服务端**权重缓存**（md5 上传一次，之后只传 id，请求体 2.5MB→60KB）；客户端**REP 平均**（同 matmul 重复 N 次取平均）；**BATCH=8** 批量（调用数降 6.7×）。

### 4.6 三模型小量真机验证（见 §5 结果表）
- Model 2（200 张）：校准（40 张，REP=4）→ 真机 86.0% vs 干净 89.0%（gap −3.0；另时刻 88.5%，gap −0.5）。
- Model 3（200 张）：校准 → 真机 89.0% vs 干净 89.0%（**gap 0.0**）。
- Model 1a（20 张）：7 光计算层，校准 8 张 → 真机 100% vs 干净 95%（n=20 仅流程验证）。
- 过程 bug 修复：校准脚本原写死 5 层，Model 1 有 7 层导致首次校准报错 → 改为按前向记录数动态识别层数。

## 5. 验证结果汇总（2026-08-07 实测）

### 5.1 四模型真机 vs 干净参考

| 模型 | 样本 | 真机（REP=4+新鲜校准） | 干净参考（同窗口） | gap | 备注 |
|---|---|---|---|---|---|
| Model 2 SpaceNet V1 | 200 | **86.0%** | 89.0% | **−3.0** | 另时刻 88.5%（gap −0.5），随漂移波动 |
| Model 3 SpaceNet V2 KD | 200 | **89.0%** | 89.0% | **0.0** | KD 训练噪声鲁棒性最好 |
| Model 1a Baseline VGG | 20 | **100%** | 95.0% | **+5.0** | n=20（SE±6.7%），仅流程/连通验证 |
| Model 4 MiniVGG-GAP | 50 | **96.0%** | 92.0% | **+4.0** | 本地新训 int8（test 95.50%），3×3+GAP 结构真机最强 |

### 5.2 关键消融（Model 2，200 张窗口）

| 配置 | 真机准确率 | 与干净参考 gap | 结论 |
|---|---|---|---|
| 无逐通道修正 | 61.5% | −27.5 | 修正为**必需项** |
| 修正 + REP=1 | 79.5% | −9.5 | 平均为**必需项** |
| 修正 + REP=2 | 83.5% | −5.5 | |
| **修正 + REP=4** | **86.0~88.5%** | **−0.5 ~ −3.0** | 性价比拐点 |
| 修正 + REP=8 | 84.5% | −4.5 | 长时间运行热漂移抵消增益 |

### 5.3 对照基线

| 口径 | 数值 | 来源 |
|---|---|---|
| QAT int8 干净 test | 92.20%（Model 2） | 复赛记录 |
| osimulator 全量 5400 | 90.43%（Model 2）/ 90.28%（Model 3） | 复赛记录 |
| 本地干净参考（同量化路径，无噪声） | 90.78%（5400 张） | 本任务复现 |
| 真机小样本（gap 0~3pt） | 86-89%（Model 2/3）、96%（Model 4） | 本任务实测 |

## 6. 遇到的问题与解法（相对 MNIST 文档的 4 个新坑）

### 坑① 硬件逐通道增益/偏移系统误差（官方 EBR 测不出）
- 现象：不修正真机仅 61.5%（≈随机偏上）。
- 根因：官方 `compass_evb_test` 数据集**零均值**，`error_mean≈0` 掩盖了增益误差；真实激活非零均值 → 逐通道 `a_j∈[0.94,1.06]`、`b_j≈(20~50)·k`。
- 解法：客户端逐通道仿射修正 `y_corr=(y_hw−b_j)/a_j`，`(a_j,b_j)` 由校准脚本在真实模型激活上逐层拟合（按权重 md5 索引，推理期权重固定可复用）。

### 坑② 残余误差是随机物理噪声（EBR 9.7 → 约 MAC 均值 7%）
- 现象：修正后残余 std≈0.07·|MAC|，同输入重跑差 2-6k，修正无法消除。
- 解法：**REP 平均**（噪声 ∝1/√N）；REP=4 为性价比拐点（再高被漂移抵消且耗时翻倍）。

### 坑③ 硬件逐通道 (a_j,b_j) 随时间漂移（~1h 量级）
- 现象：同设置 200 张先后 86.5%→83.5%，重校准回 88.5%。
- 解法：**分段再校准**（`run_full.sh` 每段开头 30-40 张重新拟合）；`run_small.sh` 也是"先校准后评估"。

### 坑④ HTTP 每次重传固定权重（2.5MB JSON）太慢
- 根因：fc1 权重 (1024,256) 每次调用序列化 ~2.5MB 经隧道传输。
- 解法：服务端按 md5 缓存权重，客户端首次上传、后续只传 `weight_id`（~60KB）。

### 工程细节
- `compass_matmul` 直接返回 MAC 单位（SDK 已乘回 tia_gain），客户端不要再除（初测曾因除以 255 误判）。
- 校准脚本层数需与模型光计算层数匹配（Model 1 为 7 层，Model 2/3 为 5 层），已改为动态识别。
- WSL 环境变量转发 `WSLENV`；`ssh -J` 两跳直达替代两段式 scp。

## 7. 结论

1. **迁移成功**：方案 A（光电分离 HTTP）三模型全部在 Gazelle 真机跑通，与干净参考差距 0~3 点（受硬件漂移影响波动），显著优于无修正/无平均（61.5% / 79.5%）。
2. **三个必备技术**（缺一不可）：
   - 逐通道仿射修正（消除增益/偏移系统误差）；
   - REP=4 平均（把 7% 物理噪声降到 ~3.5%）；
   - 分段再校准（对抗 ~1h 量级硬件漂移）。
3. **Model 3（KD）噪声鲁棒性最好**：200 张 gap = 0.0 点；Model 2 gap −3.0 点。
4. **Model 1 计算量大**（单图 ~94s，7 个光计算层）：20 张流程验证通过（100% vs 95%），全量不现实。
5. **全量预期**：相对 osimulator 基线（Model 2 90.43% / Model 3 90.28%），真机全量 5400 预计 87-89%（gap 0~3pt），分段运行流程 `run_full.sh` 已就绪，待用户择时执行。
6. **文档价值**：本任务证明「EBR 达标 ≠ 可直接用」——真实部署必须做逐通道标定 + 噪声平均 + 抗漂移分段校准。

## 8. 全量 5400 运行方法（待时机执行）

```bash
# ① 确保板上服务在跑（root）：
sudo bash /home/uisrc/opticspacenet/start_server.sh        # 输出 listening on 0.0.0.0:8000

# ② 本地开隧道（WSL，常驻）：
SSH_ASKPASS=scratch/sshpass_ask.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  ssh -L 8000:10.102.13.37:8000 -J huadong3564@140.206.121.211:2036 -N \
  -o StrictHostKeyChecking=no uisrc@10.102.13.37

# ③ 全量分段运行（每段 600 张：40 张再校准 + 600 张 REP=4 评估，约 4.5h）：
cd contest-national/opticspacenet && MODEL=model2 SEG=600 REP=4 BATCH=8 bash run_full.sh
# 结果汇总在 full_results.txt；每段日志 seg_eval_*.log；Model 3 同理（MODEL=model3）
```

预期：9 段，每段 gap −0.5 ~ −3pt，全量 ≈ 87-89%。

## 9. 部署文件清单（`contest-national/opticspacenet/`）

| 文件 | 作用 |
|---|---|
| `server_gazelle.py` | **板上** HTTP 光计算服务（stdlib，root 运行，端口 8000，权重缓存） |
| `start_server.sh` | 板上启停脚本（setsid 常驻，注意 `pkill -f "[s]erver..."` 防自杀） |
| `gazelle_engine.py` | 本地核心：`GazelleOpticalEngine` + `HttpBackend/NumpyBackend` + 五档模型定义（MODEL_REGISTRY: model2/3/1a/1b/4）+ 修正逻辑 |
| `eurosat_loader.py` | PIL 复刻 torchvision ImageFolder 排序 + eurosat_split test 索引（与基线同 5400 张） |
| `run_eval.py` | 评估客户端（BACKEND=numpy\|http，LIMIT/BATCH/REP/OFFSET/CORRECTION/MODEL 环境变量） |
| `analyze_layers.py` | 逐层校准：记录 (x,w,y_exact,y_hw) → 逐通道拟合 (a_j,b_j) → 存 calib.npz（层数动态识别） |
| `run_client.sh` / `run_calib.sh` | WSL→Windows env 转发助手（WSLENV，含 MODEL/REP） |
| `run_full.sh` | 全量分段驱动（每段再校准 + REP=4 评估 + 汇总 full_results.txt） |
| `run_small.sh` | 一键小样本端到端流程（MODEL/N/REP/BATCH 可配） |
| `diag_gazelle.py` / `probe_gazelle.py` / `probe2_gazelle.py` | 硬件诊断/标定探针（tia_gain、逐通道 a/b、可重复性） |
| `small_validation_results.txt` | 三模型真机小量验证结果快照 |

板上部署目录：`/home/uisrc/opticspacenet/`（服务器常驻运行中）。

## 10. 复现步骤（一键小样本，推荐）

```bash
# ① 板上（root）起服务  →  ② 本地隧道（见 §8）
# ③ 一键端到端（校准 + 真机 N 张 + 干净参考对比）：
cd contest-national/opticspacenet
MODEL=model2  N=200 bash run_small.sh   # Model 2 → 预期 gap ≈ −3.0
MODEL=model3  N=200 bash run_small.sh   # Model 3 → 预期 gap ≈ 0
MODEL=model1a N=20 REP=1 BATCH=1 CALIB_IMGS=8 bash run_small.sh  # Model 1 → 流程验证
MODEL=model1b ...                       # 变体 B（conv3_2 电计算）
```

手动分步（与 run_small.sh 等价）：
```bash
bash run_calib.sh MODEL=model2 LIMIT=40 BATCH=8 REP=4 CALIB_OUT=calib.npz
bash run_client.sh MODEL=model2 BACKEND=http LIMIT=200 BATCH=8 REP=4 CORRECTION=calib.npz
```

## 附录 A：三模型校准实测（逐层 a_mean / b_mean / 修正降噪比）

| 模型 | 层 | 形状 (m,k)@(k,n) | exact\|·\|mean | raw_err std | a_mean | b_mean | 修正降噪比 |
|---|---|---|---|---|---|---|---|
| M2 | stage1 | (51200,32)@(32,16) | 22939 | 1515 | 1.015 | 527 | 1.16× |
| M2 | stage2 | (3200,64)@(64,32) | 32588 | 2436 | 1.031 | 1091 | 1.34× |
| M2 | stage3 | (3200,32)@(32,16) | 13052 | 1192 | 1.015 | 481 | 1.11× |
| M2 | fc1 | (50,1024)@(1024,256) | 196934 | 13793 | 1.010 | 11257 | 1.32× |
| M2 | fc2 | (50,256)@(256,10) | 74204 | 5454 | 1.019 | 5387 | 1.36× |
| M3 | stage1 | (40960,32)@(32,16) | 7855 | 1066 | 1.034 | 329 | 1.23× |
| M3 | fc1 | (40,1024)@(1024,256) | 130685 | 11450 | 1.025 | 8659 | 1.47× |
| M1a | conv1_2 | (32768,288)@(288,32) | 29159 | 3092 | 1.018 | 2209 | 1.15× |
| M1a | conv3_2 | (2048,1152)@(1152,128) | 176059 | 7994 | 0.992 | 3624 | 1.19× |
| M1a | fc1 | (8,8192)@(8192,256) | 1225567 | 49408 | 1.034 | 11374 | **2.73×** |
| M1a | fc2 | (8,256)@(256,10) | 89405 | 5023 | 1.032 | 1171 | **2.60×** |

（完整逐层数据见校准日志：`run_small.sh` 会落盘 `small_calib_{MODEL}.log`；Model 1 成功校准日志在 `/tmp/m1_calib2.log`；修正后残余 ≈ 随机噪声，即 EBR 9.7 物理噪声水平。）

## 附录 B：本地环境

- 本地电计算：Windows anaconda（`E:\anaconda3`，torch 2.13 CPU + numpy + PIL；无 torchvision，数据加载用 `eurosat_loader.py` PIL 复刻）
- WSL 环境变量转发：`WSLENV=KMP_DUPLICATE_LIB_OK:BACKEND:...`；`KMP_DUPLICATE_LIB_OK=TRUE` 解决 OpenMP 重复加载
- 板上：Python 3.6.9，compass_sdk 1.0.2，root 运行；部署目录 `/home/uisrc/opticspacenet/`
