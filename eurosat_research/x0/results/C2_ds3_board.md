# C2 上板：w075ds3 / ds3pool3 真机验证（逐列校准）

Round X0 · C2-board-ds3 · 2026-08-09 · 状态：✅ 完成（无抢占、无异常，单窗口背靠背）

## 结论（核心数字）

**同一 fresh compass_cali 窗口，逐列 calib，各 1000 样本（与 C1 同口径同测试集）：**

| 模型 | GPU QAT test | 本地 FAKE | **hw col** | hw−GPU | hw−torch FP32 |
|---|---|---|---|---|
| **ds3pool3**（269K，seed42） | 96.76 | 97.40 | **96.40%** | **−0.36pt** | −0.90pt（torch 97.30） |
| **w075ds3**（152K，seed42） | 95.87 | 96.30 | **94.90%** | −0.97pt | −1.50pt（torch 96.40） |
| 对照 c3d + col（C1，昨日） | — | — | 94.60% | — | — |

- **ds3pool3 96.40% = 真机新 SOTA**，比旧纪录 c3d+col 94.60% **+1.80pt**；达到并超过 95+ 目标。
- w075ds3 94.90% 同样超过旧 SOTA（+0.30pt）。
- ds3pool3 的 hw−GPU gap 仅 **−0.36pt**：qat_v8（probe 实测组分噪声训练）+ 同窗口逐列 calib 基本闭合了真机 gap。hw 错误中 23/36 与 torch FP32 参考同错（模型本身错误），hw 只净引入 9 个错误。
- 跑批轨迹（样本排序效应，非漂移）：w075ds3 97.1@104→94.0@384→94.9 收尾；ds3pool3 100@72→96.1@456→96.4 收尾，中后段单调回升，窗口内无漂移迹象。

## 放行判据（四项全过，17:37–17:41 CST）

| 判据 | 数值 | 基准/要求 | 判定 |
|---|---|---|---|
| EBR | **9.769 / 9.839** | ≥8（C1 w2：9.70/9.77） | ✅ |
| error_std | **4.694 / 4.473** | C1 w2 4.92/4.70，低噪声侧 −4.6%/−4.8% | ✅ |
| MNIST canary 1000 | HW **96.90** vs numpy ref 96.80 | gap ≤0.3pt（C1：0.30） | ✅ gap 0.10 |
| 200 样本 mini-run | HW **96.50** = ref 96.50 | 与参考一致 | ✅ gap 0.00 |

开跑前 ps 检查无 gazelle/compass/server 他队进程；全程 4 次 ps 复查均空闲，无抢占。

## 时间线（CST，单窗口）

- 17:26–17:36 compass_cali（fresh）
- 17:37 evb_test（上表）→ 17:39 canary → 17:41 mini-run → 放行
- 17:44–17:56 probe w075ds3 pairs（7 层）→ 17:56–18:02 probe ds3pool3 pairs（7 层）
- 17:58 calib_col_w075ds3.json → 18:01 calib_col_ds3pool3.json（CPU 拟合）
- 18:03 / 18:07 calib_scalar_{w075ds3,ds3pool3}.json（calibrate_any_ds3，同窗口含 h1/h2）
- 18:11–18:44 **run w075ds3 col → FINAL 94.90%**（elapsed 1980s）
- 18:46–19:40 **run ds3pool3 col → FINAL 96.40%**（elapsed 3230s）

## 部署链路适配（本任务新增，全部在 `x0/scripts/`）

- `export_ds3.py`：ckpt → 板端权重包。conv3s2 下采样层权重 reshape (C_out, 9C) int8 per-channel；stem_pool_mode（max / max3）写入 meta；labels 与 C1 canonical `labels_1000.npy` 逐位一致。
- `run_ds3_gazelle.py`：基于 C1 patched run_j1_gazelle.py。新增 `optical_conv3s2`（k=9C im2col + stride2 + m≤2 tiling 规避 FPGA 行回绕）与 `pool3s2`（torch 口径 MaxPool2d(3,s2,p1)）；逐列 calib 折叠与 C1 patch 同公式；标量/逐列逻辑由公共 `_dequant` 统一。env 全部 DS3_* 前缀。
- `probe_dump_ds3.py`：7 层 pairs（含 s1ds/s2ds，k=9C im2col 与 runner 同口径）。
- `calibrate_any_ds3.py`：同窗口标量 calib **含 h1/h2**（改进 C1 的"h1/h2 沿用昨日"做法）。

**上板前本地 FAKE 对拍**（1000 样本，`check_fake_ds3.py`）：w075ds3 FAKE 96.30 vs torch 96.40（logits corr 0.99988）；ds3pool3 FAKE 97.40 vs torch 97.30（corr 0.99990）；stem 逐位一致（err ~1e-6）。板上 FAKE 200 样本冒烟一致（97.50/98.00）。无 head bias 丢失/stem 不一致。

## probe / calib 细节

resid_std（raw counts，同窗口）：ds3pool3 s1a 1005 / **s1ds 3187** / s2a 951 / s2b 1200 / **s2ds 2698** / s3a 1250 / s3b 1537；w075ds3 762 / **2651** / 1002 / 1017 / **2541** / 1107 / 1265。conv3s2 层（k=9C=216/432）resid 约为 1×1 层 2.5–3×，与 k 放大 ~3× 一致，在 qat_v8 训练噪声口径内（config layer_noise_sigmas 对该两层给的就是同层段值）。

逐列 calib 留出验证（var reduction，全正）：w075ds3 s1a −26.5 / s1ds −16.4 / s2a −13.3 / s2b −5.6 / s2ds −3.6 / s3a −7.9 / s3b −5.3%；ds3pool3 −19.3 / −17.8 / −10.4 / −8.6 / −3.4 / −6.1 / −6.0%。列结构 SNR 3–19 全 >1，与 C1 同量级。

## 错误级分析（hw col vs torch FP32 参考预测，1000 样本）

| 模型 | 一致率 | hw 修正 torch 错误 | hw 新增错误 | 两者同错 |
|---|---|---|---|---|
| w075ds3 | 97.10% | 6 | 21 | 30（torch 36 错 / hw 51 错） |
| ds3pool3 | 98.30% | 4 | 13 | 23（torch 27 错 / hw 36 错） |

ds3pool3 的 hw 错误 64% 是模型固有错误，硬件净代价仅 9 样本。

## 未做 / 备注

- **scalar ABA 对照未跑**：两个 col 主跑完成后窗口已 ~2h15m，优先保 SOTA 数字收窗；C1 已建立 col−scalar = +1.0~+2.0pt 的同窗口结论，本窗口 scalar calib json（含 h1/h2）已留存板上与本地，需要时可补。
- s43 种子未跑（按任务优先级只跑各模型最佳种子 seed42）。
- 板上新增文件（未覆盖任何既有文件）：`run_ds3_gazelle.py`、`probe_dump_ds3.py`、`calibrate_any_ds3.py`、`weights_w075ds3/`、`weights_ds3pool3/`（测试 npy 复用 weights_c3d canonical 副本）、`probe_{w075ds3,ds3pool3}_*`、`calib_{col,scalar}_{w075ds3,ds3pool3}.json`、`logits_{w075ds3,ds3pool3}_col.npy`、`run_c2_*.log`。

## 产物位置（本地 `x0/data/c2_ds3/`）

- 2×run log + 2×hw logits + 2×torch 参考 logits（`logits_*_torch_ref.npy`）+ labels.npy
- 4×calib json（col ×2 + scalar ×2）
- pairs：`pairs_w075ds3/`、`pairs_ds3pool3/`（各 7 层 ideal+hw，xint 留板上未拉）
- 权重包与 ckpt：`x0/weights/`、`x0/ckpts/`；板上参考脚本快照：`x0/board_ref/`

## 判读与下一步

1. **95+ 达成**：ds3pool3 96.40% 为新真机 SOTA（+1.80pt），hw−GPU gap −0.36pt 基本闭合。w075ds3 94.90% 也超旧 SOTA。
2. **ds3 路线（conv3s2 光计算下采样）验证成功**：多加 2 个光计算层（k=9C，resid ~3×）并未放大 hw gap——qat_v8 噪声组分训练对更大 k 层同样有效；stem max3 + ds3 的组合（ds3pool3）是当前光计算架构最优解。
3. 建议：跨日再复现一次 ds3pool3 确认稳定性（列结构占比跨窗口波动，C1 已坐实）；若复现成立，X0 真机结论可定格为 **ds3pool3 + qat_v8 + 同窗口逐列 calib = 96.40%**。
