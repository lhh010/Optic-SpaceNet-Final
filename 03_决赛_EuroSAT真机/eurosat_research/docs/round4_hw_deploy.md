# Round 4 — J1 真机部署验证（gazelle-crossval 之后）

## 结论

**J1（1.38M MACs）真机精度 90.60%**（EuroSAT 1000 样本）。全链路：FP32 96.65% → QAT 96.30% → FAKE numpy 96.40% → **真机 90.60%**。硬件 gap 5.7pt 来自每层绝对加性硬件噪声（σ_total≈4.5 counts → 反量化后 3-10% 相对误差）× 7 层光计算累积。

## 部署脚本（Gazelle-national/mnist/）

- `export_j1.py`：torch best.pth → int8 权重 + per-channel scale + 每层 BN 参数（`{layer}_bn.npy`）+ test 数据 npy
- `run_j1_gazelle.py`：真机推理，FAKE 模式离线验证
  - 环境变量：`J1_WEIGHTS_DIR` / `J1_FAKE=1` / `J1_LIMIT` / `J1_BATCH` / `J1_CALIB`（compass_sdk 篡改 sys.argv，禁位置参数）
  - `optical_mm` 用 m≤2 tiling（FPGA m≥3 行回绕 bug）
  - 反量化：`y = x_scale·w_scale·y_int − x_scale·zp·w_scale·col_sum`
  - per-layer 校准：`y = (y − beta)/alpha`（仅真机）
- `calibrate_j1.py`（随机输入）→ `calib_j1.json`；`calibrate_real.py`（真实激活分布）→ `calib_j1_real.json`

## 关键 Bug 修复（FAKE 对齐 QAT 过程）

1. **反量化公式符号/缺 scale**：`+x_zp·w_scale·col_sum` → `−x_scale·x_zp·w_scale·col_sum`（x 反量化 x≈scale·(x_int−zp)）。原公式 corr≈0，修正后 corr 0.99997
2. **光计算层后 BN 未处理**：5 层 stage BN（s1a/s2a/s2b/s3a/s3b）需在反量化输出上应用（stem 已有）
3. **pool 位置错误**：stage2 是 [conv,BN,relu,conv,BN,relu,**pool**]，pool 在 stage 末尾不在中间
4. 板上 numpy 旧：`np.pad` 需显式 `mode="constant"`；无 `np.random.default_rng`/`rcond` 参数

修复后 FAKE = 96.40% ≈ QAT 96.30% ✓

## 真机校准（关键步骤）

probe 测量：`compass_matmul` 输出 ≈ **alpha×ideal + beta + eps**（alpha≈1.03-1.06 系统性增益，beta 偏移，resid_std 每层 700-2000 raw）。

| 层 | alpha | beta | resid_std |
|---|---|---|---|
| s1a | 1.0564 | 126 | 684 |
| s2a | 1.0551 | 397 | 995 |
| s2b | 1.0626 | 611 | 1352 |
| s3a | 1.0532 | 741 | 1619 |
| s3b | 1.0651 | 690 | 1297 |
| h1 | 1.0569 | 1186 | 1965 |
| h2 | 1.0589 | 702 | 1457 |

（真实激活分布校准；随机校准 alpha≈1.03 差异因激活幅值分布不同）

- 校准前 54.40% → 校准后 91.80%（500）/ **90.60%（1000）**
- resid_std 不可线性校准（非线性 + 每层噪声底），是硬件 gap 主因
- 随机 vs 真实激活校准结果一致（91.80%）→ 校准饱和

## SOP（有效执行）

1. EBR 检查 ≥ 8（本次 9.81→9.91 校准后）
2. 新鲜 `compass_cali`（~10 min）+ MNIST canary 94.40%（与参考一致）
3. 大跑前校准已生效

## 结论与建议

- J1 真机 90.60% 在 1.38M MACs 下合理；相比 Model 2/3 osim 90.43% 口径更好（真机直接验证）
- 若要进一步缩小 gap：训练噪声需匹配真实 per-layer 噪声累积特性（当前 0.0392 是单层口径）；或深层用更宽位宽/更少光计算层
- 校准 alpha/beta 必须在**每次 compass_cali 后**重做（漂移），已入 SOP
