# C1 逐列校准（per-column calib）板上验证结果

Round X0 · C1-board · 2026-08-09 · 状态：✅ 完成（w2 恢复窗口 ABA 对照）

## 结论（核心数字）

**w2 恢复窗口 ABA 背靠背（c3d 权重，各 1000 样本，同一 fresh compass_cali 窗口）：**

| 段 | 配置 | hw acc |
|---|---|---|
| A | 标量 calib（calib_scalar_c3d_w2.json） | **92.60%** |
| B | **逐列 calib**（calib_col_c3d_w2.json，J1_CALIB_COL） | **94.60%** |
| A′ | 标量 calib（窗口内漂移检测） | **93.60%** |

- **col − mean(scalar) = +1.50pt**（col−A +2.0，col−A′ +1.0）；A′−A = +1.0pt（窗口内漂移，
  与历史 run-to-run ±1.3pt 一致，窗口稳定）。
- **col 94.60% 为真机历史最佳**（旧纪录 c3d 93.80% @ 2026-08-08）。
- 错误级分析（logits 对 labels）：col 修正 scalar A 错误 26 个 / 新增 6 个；修正 A′ 错误 19 个 /
  新增 9 个。修复/新增比 ~3:1，与 v8/v9 家族跨窗口表现同向，不是噪声抖动。
- 判读：逐列 calib 在严格同窗口 ABA 对照下真实收益 **+1.0~+2.0pt**，与设计文档按 C3 噪声
  敏感性外推的"+零点几~1pt"一致（偏上沿）。

## 过程时间线（UTC）

- 05:20–05:33 compass_cali #1（EBR 9.78/9.83，error_std 4.67/4.50）→ MNIST canary **94.70%**（ref 94.40）✓
- 05:34–05:42 probe_c3d pairs w1（s1a 16384 / s2a 4096 / s2b 4096 / s3a 1024 / **s3b 8192 行**）
- 05:44 calib_col_c3d.json（w1）：留出方差改善 s1a −24.4 / s2a −17.3 / s2b −10.8 / s3a −14.7 / s3b −4.1%，
  结构 SNR 5–83 全 >1，与本地 c2c pairs 预测（4.5–25.2%）精确吻合
- **05:44 他队抢占**（server_gazelle.py, HTTP :8000, 来源 10.102.12.4 ssh 隧道）：v1 跑批中止于
  56/1000 @94.64%（device busy）；~06:05 对方自行退出，未动对方任何文件/进程
- 06:46 现场全部拉回本地 `x0/data/incident_20260809/`（含 MANIFEST.md）+ probe_post_s2a 探针
- 06:59 恢复验证第 1 次：error_std 6.09/5.73（+30%/+27%）**不达标** → 等 10 min 重试
- 07:21 恢复验证第 2 次（fresh cali）：EBR 9.70/9.77，error_std 4.92/4.70（+5.3%/+4.5% <10%），
  MNIST canary **94.70%**（Δ0.0），200 样本 97.50（vs 98.00）→ **全部达标放行**
- 07:23–07:33 w2 pairs（probe_c3d_w2_*，同 w1 行数）→ calib_col_c3d_w2.json（留出 var
  −7.8/−14.2/−6.0/−12.3/−3.6%，全正，幅度小于 w1——列结构占比有跨窗口变化）→
  calib_scalar_c3d_w2.json（5 层 w2 pairs 标量拟合 + h1/h2 沿用 calib_c3d.json）
- 07:33–08:50 ABA 跑批（上表）

## 污染窗口数据（⚠️ 仅存档，不采信）

v2 窗口（06:00–06:37，他队 server 抢占后、恢复 cali 前）：scalar（calib_scalar_c3d_fresh.json，
05:42 pairs 拟合 + 昨日 h1/h2）FINAL **85.50%**；col 跑至 392/1000 @92.60% 被停。
轨迹 93.3@104 → 84.0@200 → 85.5 收尾，显著低于同日干净窗口（92.6@208 同位置 93.75）。
窗口状态被抢占事件污染，绝对值与相对比较均不采信。现场与分析见
`x0/data/incident_20260809/` 与 `x0/results/C1_incident_analysis.md`（本地归因 subagent 产出）。

佐证：昨日 stale 标量 calib 与当日 fresh 标量 calib 在当日 pairs 上 resid_std 差异 <3%
（calib 参数本身不是退化原因；退化在 pairs 采集后的 run 窗口硬件状态）。

## 归因

（本任务只做板上执行；抢占影响定量归因见 `x0/results/C1_incident_analysis.md`——
由本地分析 subagent 基于 incident_20260809 现场材料产出。）

## 数据与产物位置

- w2 pairs：`x0/data/probe_pairs_c3d/probe_c3d_w2_*`（s3b 8192 行，补齐 A1 指出的小样本问题）
- w2 跑批：`x0/data/c1_w2/`（3×run log + 3×logits + 2×calib json + labels.npy）
- w1（抢占前）pairs 与全部现场：`x0/data/incident_20260809/`
- 板上新文件（均带 c3d/col/w2/post 后缀，未覆盖任何既有文件）：
  `calibrate_col.py`、`probe_dump_c3d.py`（新增，支持 PROBE_LAYERS/PROBE_OUT_PREFIX/逐层行数 env）、
  `run_j1_gazelle.py`（已 patch，原文件备份 `run_j1_gazelle.py.bak_c1`）、
  `calib_col_c3d{,_w2}.json`、`calib_scalar_c3d_{fresh,w2}.json`、`probe_c3d_*`、`probe_c3d_w2_*`、
  `probe_post_s2a_*`、`run_c1_*.log`、`run_w2_*.log`、`logits_*`

## patch 适配记录（板上 diff vs 仓库 patch）

`run_j1_gazelle_colcalib.patch` 的 hunk 1–3 干净应用；**hunk 4（optical_fc）手动合并**——
板上版本比仓库版多 head bias 支持（`bias=None` 参数 + 反量化后加 bias），逐列折叠代码插入在
bias 相加之前，语义与 patch 一致。patch 后本地数值自检：折叠路径 vs 逐步"先 calib 再反量化"
最大偏差 2.2e-16；板上 FAKE 对拍（patched vs .bak_c1，64 样本）logits 逐位一致。

## 异常与决策记录

1. **他队抢占**（05:44–~06:05）：按纪律不对抗，保存现场等待释放；v2 窗口数据按用户裁决标记不采信。
2. **恢复验证第 1 次不达标**（error_std +30%）：等 10 min + fresh cali 后第 2 次全达标。
   error_std 当日轨迹 4.67→4.95→6.09→4.92（两次 cali 后回落），提示存在小时级状态波动。
3. **标量基线 calib 选择**：发现 calib 参数跨窗口漂移（beta 最大 ~200 counts），按"同窗口"原则
   用当日 pairs 重拟标量基线（h1/h2 无 pairs 沿用昨日），保证 scalar vs col 严格同窗同源。
4. **w1 vs w2 留出改善幅度差异**（如 s1a −24.4% → −7.8%）：列结构占比跨窗口不稳定，
   支持"逐列 calib 须与跑批同窗口采集"的结论（设计文档 §6 风险已坐实）。

## 下一步建议

1. **采信口径**：逐列 calib +1.0~+2.0pt 真实有效，c3d + col calib（94.60%）可列为新真机 SOTA；
   建议跨日再复现一次确认稳定性（列结构占比跨窗口波动，收益幅度可能随之波动）。
2. **部署 SOP 固化**：pairs → calibrate_col → 跑批必须同窗口背靠背；calib_col.json 不入库复用。
3. **h1/h2 无 pairs**：逐列 calib 未覆盖 head（回退标量）。如需补全，head 层 probe 行数少
   （FC 每样本 1 行，1000 样本=1000 行/层），一次 dump 即可。
4. w2 pairs（含 s3b 8192 行）已回本地，可供 A1 残差分解复跑（对比 c2c 旧 pairs 的 s3b 1024 行）。
