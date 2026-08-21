# Gazelle 真机 vs osimulator 交叉验证报告（CROSSVAL）

日期：2026-08-07 · 数据：`stats.json`（同目录拷贝）、`fitted_params.json`（同目录拷贝）· 图：`figures/`

> 本文件是从 `gazelle-crossval/` 仓库拷贝的归档副本（数据/图已随附同目录）。§8 复现指南中的命令需在 gazelle-crossval 仓库根目录执行。

## 1. 摘要

Gazelle 板的工作机制已查清：`compass_matmul` 输出 = FPGA FIFO count × 255（`tia_gain_scale_factor`，设计语义，compass_lib.py:166 源码实证），alpha 归一后增益 ≈1.0；此前观察到的"小信号垃圾"全部是 ±4~5 counts 的**绝对加性噪声底**淹没小信号所致，模拟核心本身健康。真机噪声是**与信号幅度无关的绝对加性底噪**（uint8/uint4x16 下 σ_total ≈ 4.49/3.85 counts，与 evb 长窗 4.4 counts 吻合），其中短窗快噪声仅 1.02 counts，σ_static ≈ 4.37 counts 的慢漂+静态分量主导——因此操作规程要求大跑前新鲜 `compass_cali` + MNIST canary。osimulator 则是**每次调用随机**的噪声模型，结构为"底噪 + 信号相关分量"（跨 regime 变化 ~300×），与真机的纯绝对加性噪声**结构性不同**——这是对 QAT 最重要的修正输入。逆向报告 §7 的行为结论（±3 LSB 非线性、可加性、无饱和、零串扰、alpha≈1.0437）在同归一化口径下全部被 sim 侧复测确认。

## 2. 实验设置

**配对协议**：同一组向量（`vectors/exp_vectors.npz`，由 `experiments.py` 定义、`gen_vectors.py` 生成）分别在两侧执行同一 GEMM，逐块对比：

- **hw 侧**：Gazelle 真机（内网 10.102.13.37，两段 SSH 跳板），`compass_sdk` 的 `compass_matmul`，`hw_side/run_hw_experiments.py`。
- **sim 侧**：osimulator 1.3.4（docker context `fdusc-cpu-135`，容器 `gazelle_sim`，conda `moca_llm`，Python 3.9），`sim_side/run_sim_experiments.py`。

**实验规模（N）**：

| 实验 | 内容 | 块数 / 重复 |
|---|---|---|
| E0–E4 | 布局/逐元素/可加性/饱和/串扰小实验 | 46 blocks（`results_hw_small2.npz` / `results_sim_small.npz`） |
| E5 | 大 N 随机基准，3 值域（uint4/uint8/uint4x16）× 100 chunk × 100 向量 | 300 blocks = 每值域 10000 GEMM（`results_hw_big.npz` / `results_sim_big.npz`） |
| E6 | 重复性：100 固定对 × 2 值域 | hw 200 blocks × **100 repeats**；sim 200 blocks × **2 repeats**（`results_hw_e6.npz`） |
| E7 | 噪声-幅度：base × {1,2,4,8,16} | 5 blocks（含于 big） |

**已知硬件约束与绕行**：FPGA block-matmul 在 **m≥3（k=8）时行回绕**（第 r≥2 行返回第 r%2 行的重算结果，debug-1 实证）；hw 引擎按 **m≤2 tiling** 拆分调用规避（commit `cdb6b14`），全部 hw 数据均经此路径采集。

**校准状态与日期**（2026-08-07，本地 00:13–01:57）：

| 时点 | EBR | error_std (counts) | 备注 |
|---|---|---|---|
| Task 7 首跑（00:13，`results/evb_20260807_001353.log`） | [9.7326, 9.6527]，min 9.653 | [4.81, 5.09] | E0 异常，触发 debug |
| Task 7b 校准前（01:12） | [9.69, 9.64] | [4.95, 5.12] | — |
| Task 7b `compass_cali` 后（01:22） | **[9.89, 9.84]** | [4.32, 4.48] | MNIST canary **94.50%**（ref 94.40%） |
| Task 7b 重跑 E0–E4（01:24，small2） | min **9.83** | — | 46 blocks |
| Task 8 校准前 evb（01:33） | [9.886, 9.825] | [4.33, 4.52] | — |
| Task 8 `compass_cali`（01:34→01:43，~9 min，`results/cali_20260807_013410.log`）后 | [9.853, 9.813] | [4.43, 4.55] | canary **94.70%**（ref 94.40%） |
| Task 8 hw big（01:49–01:51） | min **9.811** | — | 305 blocks |
| Task 8 hw e6（01:55–01:56） | min **9.817** | — | 200×100 |

EBR 全程 ≥ 8 安全线。板子存在**跨小时漂移**（Task 7 调试期间观测：8/6 上午 MNIST ~97% 的同形状重放，数小时后未重校准直接重跑时输出误差显著增大，重校准后恢复；具体幅度见 Task 7b 调试记录，非严格定量），故确立 SOP：**大 k 跑批前必须新鲜 `compass_cali` + MNIST canary 验证**。

## 3. E0 布局与增益

- **增益语义（实测+源码实证）**：hw 输出严格为 255 的整数倍。`compass_lib.py:166`：`result = compass_mm(...) * tia_gain_scale_factor`，`tia_gain_scale_factor = 255 if tia_rx==0 else 25.5`（板上 `calibration_settings.yaml` 为 `tia_rx: 0`）→ **输出 = FPGA count × 255 ≈ MAC 值**。官方 `local_matmul_sample.py` 中 `tia_gain_scale_factor = 256` 的写法是过时注释性代码。
- **增益实测（alpha_hw）**：信噪比足够大的块上 alpha_hw ≈ 1.0——E4_u8 两块 1.029/1.003（Task 7b）；Task 9 大 N：uint8 **1.0230**、uint4x16 **0.9899**（stats.json E5）。即 count×255 ≈ MAC 单位，alpha ≈ 1 确认。
- **布局 sanity（count 域，Task 7b small2）**：A 1×8@8×2：hw [1,2] cnt vs ref [0.14,-0.14] cnt（≤2 cnt 误差）；B 2×8@8×2：≤1 cnt；C 4×16@16×4：四行互不相同、**无 row%2 回绕模式**、各行 ≤3 cnt——全部落在 evb 噪声底（~4.3 counts）内。首跑（Task 7）的"fit 乱跳 + diff 巨大"实为信号 ≈0 counts 时对纯噪声拟合，非通路故障。
- 板上 evb 与 matmul 是**同一条数据通路**（evb 即 10000 次 compass_matmul 后在 count 域比误差）；EBR≥8 规格等价于 error_std ≤ 16 counts。

## 4. 逆向结论对照表

对照对象为 `osimulator/GAZELLE_ARCHITECTURE.md` §7 的行为结论。sim 复测 = 本工程 stats.json（E1–E4 用 small 配对，E5 用 big）；hw 实测 = 同表 hw 侧。单位：MAC（= raw）；counts = MAC/255。

| # | 逆向结论 | sim 复测 | hw 实测 | 判定 |
|---|---|---|---|---|
| E1 | 逐元素非线性 ±0~3 LSB（§7.2） | 16 个 u4 块：std 0.56–1.15 LSB，max_dev 1.14–3.05 LSB → 复现 ±3 LSB | std 10.1–1115.3 MAC = **0.04–4.37 counts**，max_dev 最大 2480 MAC（k6_n1）；std 全部 ≤ evb 噪声底 | **确认**（sim 复现；hw 与噪声底一致，无结构性差异——噪声底内无法分辨更细的逐元素非线性） |
| E2 | k 元素可加性，耦合 ±1~3 LSB（§7.3） | max_coupling_dev = **3.35 LSB** | max_coupling_dev = 281.29 MAC = **1.10 counts**（噪声底内） | **确认**（sim 3.35 与 ±1~3 同量级；hw 在噪声底内一致） |
| E3 | 全值域无饱和（§7.4） | ratio_min=0.0 / ratio_max=**3.06** | ratio_min=0.0 / ratio_max=**2.16** | **确认（附注）**：ratio 极值由 sweep 小信号端（ideal≈136，低于/接近噪声底）的噪声主导，**不是饱和证据**；大信号端双侧均无饱和，与 §7.4 一致 |
| E4 | n 通道串扰 ≈0（§7.5） | max_leak = 783.22 MAC（相对 sig_std 18828 约 4%——sim 噪声随激励幅度注入到静通道，见 §5） | max_leak = 1703.19 MAC = **6.68 counts** ≈ 噪声底 max | **确认（hw 侧）**：hw 串扰在噪声底内 ≈0；sim 侧非零 leak 反映 sim 的噪声注入机制，非真串扰 |
| E5 | uint4 残差 mean=-2.44 / std=4.72 / MAE=4.18 LSB（§7.6） | alpha=**1.0431**（逆向同归一复算 1.0437），normed std=**1.088**（95% bootstrap CI [0.955, 1.210]），mean=0.07，MAE=0.834 | uint4：alpha=0.364、std=712.7 MAC=2.79 counts——**信号（±1.5 counts）低于噪声底，fit 无物理意义**；uint8/uint4x16：σ=**4.49/3.85 counts**，alpha=1.023/0.990 | **确认 + 口径修正**：逆向 std=4.72 是**未归一 raw 残差**（含 alpha≈1.0437 增益与 bias），按本工程相同 alpha 归一复算为 std=1.27，本工程 1.088 忠实复现；hw 侧噪声量级相符但**结构不同**（见 §5） |

E5 完整残差表（MAC 单位，alpha 归一后；hw counts = std/255）：

| regime | side | alpha | mean | std | std 95% CI | std (counts) |
|---|---|---|---|---|---|---|
| uint4 | hw | 0.364（无意义） | 99.75 | 712.67 | [656.42, 764.81] | 2.79 |
| uint4 | sim | 1.0431 | 0.07 | 1.088 | [0.955, 1.210] | — |
| uint8 | hw | 1.0230 | -355.02 | 1143.86 | [1042.50, 1244.26] | **4.49** |
| uint8 | sim | 1.0433 | 164.75 | 357.35 | [326.86, 388.27] | — |
| uint4x16 | hw | 0.9899 | 603.34 | 981.74 | [893.20, 1069.63] | **3.85** |
| uint4x16 | sim | 1.0485 | -17.77 | 266.37 | [238.87, 293.53] | — |

图：`figures/e5_uint4_residual.png`（E5 uint4 残差分布 hw vs sim 双侧对比；2026-08-20 重绘——原图标题文字叠印，且原始直方图数据 `results_*_big.npz` 未随档，故按本表统计量以拟合正态 N(μ, σ) 重绘，重绘脚本 `plot_e5_uint4_residual.py` 同目录）。

## 5. 真机独有发现

### 5.1 σ_dynamic：时间尺度依赖的噪声（E6）

| 侧 | σ_dynamic（中位） | max run-to-run diff | 说明 |
|---|---|---|---|
| hw（100 repeats，短窗 <1s） | 260.93 MAC = **1.02 counts** | 5100 MAC（20 counts） | 短窗快噪声仅 ~1 count |
| sim（2 repeats） | 97.83 MAC | **1220**（>0 → 随机性实锤） | 仅 2 repeats，只作随机性指标 |

**关键结论**：
- hw 噪声是**时间尺度依赖**的：短窗（<1s）快噪声 σ≈1.02 counts，但长窗总噪声 σ_total≈4.49 counts（uint8）→ **σ_static = sqrt(σ_total²−σ_dynamic²) ≈ 4.37 counts 主导**，来自慢漂 + 静态 LUT 误差。这解释了"跨小时漂移必须重校准"的 SOP。
- **osimulator 是每次调用随机的**（非静态 LUT）：sim E6 run-to-run 中位 97.83 MAC（输出幅度 ~±40k，**~0.24%**；仅 2 repeats，只作随机性指标），与逆向报告 MAE 4.18 LSB / 相对误差 0.6% 同量级，互为印证。sim 侧比较统计必须按噪声分布处理，不能当确定性真值。

### 5.2 噪声-幅度关系（E7）：绝对 vs 相对

对 5 档缩放（sig2 从 3.8e4 到 2.5e9）拟合 σ² = b·E[ideal²] + a：

| 侧 | a（绝对底噪方差, MAC²） | σ_floor = √a | b（相对斜率） |
|---|---|---|---|
| hw | 144167.58 | ≈380 MAC ≈ **1.49 counts** | 3.82e-4 |
| sim | 70563.43 | ≈266 MAC | 1.25e-5 |

解读：**a 是绝对加性底噪，b 是相对（信号相关）分量**。

- **hw = 纯绝对加性噪声**：E5 大 N 下三个值域 σ_total = 2.79/4.49/3.85 counts 近似恒定（uint4 的 2.79 还是被小信号截断压低后的值），与 evb 长窗 4.4 counts 吻合。E7 的 b=3.82e-4 看似非零，但拟合仅 5 点且最大信号点杠杆极大，仅供量级参考；以 E5 大 N 为准。
- **sim = 底噪 + 信号相关分量**：E5 sim std 跨 regime 从 1.09（uint4）→ 266（uint4x16）→ 357（uint8）MAC，变化 **~300×**；E4 静通道 783 MAC 的"leak"也是同一机制（噪声随激励幅度注入）。
- **双侧噪声结构本质不同**，这是对 QAT 最重要的单一输入：用 sim 的噪声模型做 QAT，学到的是"相对噪声鲁棒性"，而真机要的是"绝对底噪鲁棒性"。

图：`figures/e7_noise_vs_power.png`（σ² vs E[ideal²] 双侧拟合）。

## 6. 拟合噪声模型（`qat_update/fitted_params.json` 全文）

模型：`out = alpha*ideal + beta + eps, eps ~ N(0, σ_static² + σ_dynamic²)`。原始单位 = FPGA counts × 255；`*_counts` 字段 = raw/255。

```json
{
  "model": "out = alpha*ideal + beta + eps, eps~N(0, sigma_static^2+sigma_dynamic^2)",
  "units": "raw = FPGA counts * 255 (alpha-normalized MAC units); *_counts = raw / 255",
  "regimes": {
    "uint4": {
      "alpha": 0.36410201521202046,
      "beta": 99.74855100463678,
      "sigma_total": 712.6679792135493,
      "sigma_dynamic": 260.9276837887086,
      "sigma_static": 663.1835284662789,
      "sigma_total_counts": 2.794776389072742,
      "sigma_static_counts": 2.6007197194756033,
      "sigma_dynamic_short_window_counts": 1.0232458187792492,
      "rms_ideal": 111.13136168516968,
      "noise_structure": "absolute_additive",
      "noise_structure_note": "E5 hw σ_total 在 uint8/uint4x16 下≈恒定绝对加性底噪（4.49/3.85 counts），与 EVB 长窗 4.4 counts 吻合；σ_static = sqrt(σ_total²−σ_dynamic²) 为慢漂+静态 LUT 误差的slow/static 分量，σ_dynamic 仅为短窗快噪声。",
      "warning": "signal below noise floor; alpha/fit not physically meaningful"
    },
    "uint8": {
      "alpha": 1.0229626549137127,
      "beta": -355.0166961785841,
      "sigma_total": 1143.8570163847724,
      "sigma_dynamic": 260.9276837887086,
      "sigma_static": 1113.6990696616986,
      "sigma_total_counts": 4.485713789744206,
      "sigma_static_counts": 4.367447332006661,
      "sigma_dynamic_short_window_counts": 1.0232458187792492,
      "rms_ideal": 29201.75808650825,
      "noise_structure": "absolute_additive",
      "noise_structure_note": "E5 hw σ_total 在 uint8/uint4x16 下≈恒定绝对加性底噪（4.49/3.85 counts），与 EVB 长窗 4.4 counts 吻合；σ_static = sqrt(σ_total²−σ_dynamic²) 为慢漂+静态 LUT 误差的slow/static 分量，σ_dynamic 仅为短窗快噪声。"
    },
    "uint4x16": {
      "alpha": 0.9898616228923051,
      "beta": 603.3404612866501,
      "sigma_total": 981.7418917189545,
      "sigma_dynamic": 260.9276837887086,
      "sigma_static": 946.4321876334147,
      "sigma_total_counts": 3.8499682028194298,
      "sigma_static_counts": 3.711498775032999,
      "sigma_dynamic_short_window_counts": 1.0232458187792492,
      "rms_ideal": 29986.30765420778,
      "noise_structure": "absolute_additive",
      "noise_structure_note": "E5 hw σ_total 在 uint8/uint4x16 下≈恒定绝对加性底噪（4.49/3.85 counts），与 EVB 长窗 4.4 counts 吻合；σ_static = sqrt(σ_total²−σ_dynamic²) 为慢漂+静态 LUT 误差的slow/static 分量，σ_dynamic 仅为短窗快噪声。"
    }
  },
  "noise_vs_power": {
    "hw": {
      "a": 144167.5827301027,
      "b": 0.0003818301451809411
    },
    "sim": {
      "a": 70563.43200118159,
      "b": 1.246965297008512e-05
    }
  },
  "e6_sim_max_run_to_run_diff": 1220.0,
  "sim_characterization": {
    "type": "stochastic_per_call",
    "structure": "floor + signal-dependent",
    "e5_uint4": {
      "alpha": 1.0430606329109733,
      "mean": 0.07215201088353344,
      "std": 1.087772282367014,
      "mae": 0.8338567420692897,
      "max_abs": 3.7615111273790554,
      "std_ci": [
        0.9550208759591742,
        1.210267625583228
      ]
    }
  },
  "qat_suggestion": {
    "tia_noise_std": 0.03917082707815648,
    "noise_std_ratio": 0.00893534159880812,
    "sigma_dynamic_short_window_counts": 1.0232458187792492,
    "note": "tia_noise_std=σ_total/rms_ideal; noise_std_ratio=σ_dynamic/rms_ideal; rms_ideal 为 uint8 值域 E5 ideal 的 RMS。注意：σ_dynamic 来自 E6 短窗（100 repeats < 1s）快噪声测量，长窗总噪声以 σ_total（uint8 ≈4.49 counts）为准，两者不可混用。"
  }
}
```

## 7. 对 QAT 的建议

1. **噪声结构：用绝对加性高斯底噪，不要用纯相对（乘性）噪声。** 真机 σ_total 与信号幅度无关（uint8 4.49 / uint4x16 3.85 counts ≈ evb 长窗 4.4 counts）。QAT 中注入 `eps ~ N(0, σ²)` 的加性噪声，σ 按 count 域 ≈4.4 counts（≈1120 MAC 单位）设定，而非按输出的固定百分比。osimulator 的"底噪+信号相关"结构（~300× 跨 regime 变化）不能代表真机。
2. **tia_noise_std 建议值 = 0.0392**（= σ_total/rms_ideal = 1143.86/29201.76，**uint8 全值域随机 GEMM 基准**，rms_ideal ≈ 114.5 counts）。⚠️ **归一化口径警告**：Ltsimulator-test 当前值 **5.34e-4** 与本值差 **~73×**——两边 RMS 归一化基准不同（本工程是 uint8 全值域随机 GEMM 的输出 RMS；Ltsimulator-test 按每层输出 RMS/激活尺度归一）。**Task 12 打补丁前必须对照 `optic_layers.py` 实际 int8 GEMM 输出幅度核对归一化基准，禁止直接替换数值。**
3. **σ_dynamic 与 σ_total 不可混用**：`noise_std_ratio=0.00894`（σ_dynamic/rms_ideal）来自 E6 短窗（100 repeats <1s）快噪声（1.02 counts），只适用于"单帧内"口径；长窗总噪声以 σ_total（uint8 ≈4.49 counts）为准。σ_static ≈4.37 counts（慢漂+静态误差）主导总噪声，模型应对慢漂鲁棒。
4. **uint4 直接上板不可行**：uint4 信号仅 ±1.5 counts，低于 4.4 counts 噪声底（SNR<1，拟合 alpha 无物理意义）。低位宽激活必须 ×16 放大（uint4x16 路径，rms_ideal ≈ 117.6 counts，SNR 足够）或直接用 uint8。这与 Gazelle-national MNIST 必须 ×16 的经验一致。
5. **操作规程入模型预期**：板子跨小时漂移（调试期观测到大 k 通路误差显著增大，非严格定量，见 §2），靠新鲜 `compass_cali` + canary 恢复。QAT 应把 σ_total 当作"校准后新鲜状态"的噪声；漂移余量靠 SOP 保证，不靠训练吸收。

## 8. 复现指南

前置：SSH 免密（跳板 `huadong3564@140.206.121.211:2036` → `uisrc@10.102.13.37`）、docker context `fdusc-cpu-135` 可用。

```bash
cd gazelle-crossval

# 0. 连通性与代码上传
bash driver/hw_drive.sh verify && bash driver/hw_drive.sh upload
bash driver/sim_drive.sh verify && bash driver/sim_drive.sh upload

# 1. 校准 SOP（大跑前必做）
bash driver/hw_drive.sh evb        # EBR ≥ 8 才继续
bash driver/hw_drive.sh cali       # compass_cali，约 10 分钟
bash driver/hw_drive.sh evb        # 确认 error_std 下降
# 板上跑 MNIST canary（1000 样本，预期 94–95% vs NumPy 参考 94.40%）
# 脚本在 Gazelle-national/mnist/（仓库外，需另行获取），不在本仓库内

# 2. E0–E4 小实验（hw 经 m≤2 tiling 引擎）
bash driver/hw_drive.sh run E1,E2,E3,E4 0 small2 && bash driver/hw_drive.sh fetch small2
bash driver/sim_drive.sh run E1,E2,E3,E4 0 small && bash driver/sim_drive.sh fetch small

# 3. E5–E7 大 N 实验
bash driver/hw_drive.sh run E5,E7 1 big && bash driver/hw_drive.sh fetch big
bash driver/hw_drive.sh run E6 0 e6 && bash driver/hw_drive.sh fetch e6   # hw 100 repeats
bash driver/sim_drive.sh run E5,E6,E7 0 big && bash driver/sim_drive.sh fetch big  # sim E6 自动 2 repeats

# 4. 分析与拟合（本地）
python3 analysis/compare.py          # → results/stats.json + report/figures/*.png
python3 analysis/fit_noise_model.py  # → qat_update/fitted_params.json
python3 -m pytest tests/ -q          # 12 passed
```

注意：osimulator 每次调用随机（无种子），重跑 sim 侧数值会有 ~1% 量级抖动，结论以统计口径为准。

## 9. 局限性

- **单块板**（die N4-D09-1-EVB02）：所有结论来自一块 EVB，板间差异未测。
- **温度/时间漂移未定量**：只定性确立"跨小时漂移 → 新鲜校准 SOP"，未测漂移速率与温度依赖。
- **LUT 文件级对比未做**：逆向 dump 的 LUT（`gazelle_artifacts/`）与板上 `txt_lut.csv`/weight LUT 未逐文件 diff，只做了行为级对比。
- **E6 sim 仅 2 repeats**：只够证明随机性（max diff 1220>0），不能估 sim σ；sim σ 以 E5 大 N 残差为准。
- **E7 仅 5 点且杠杆集中**：b（相对斜率）不确定度大，噪声结构结论以 E5 大 N 为准。
- **uint4 regime 低于噪声底**：其 hw alpha/σ_static 分解无物理意义（JSON 已带 warning），下游禁用。
- **小信号区 hw 增益 ~1.3 未确证**：E2_u8/E3_u8 小信号块 fitted alpha 稳定 ~1.3，但信号低于噪声底，无法区分真实增益偏差与拟合伪影。
- **全部 hw 测量经 m≤2 tiling 绕行**（FPGA m≥3 行回绕 bug，commit `cdb6b14`），未直接验证硬件原生多块路径；k=784 长累积通路未在本框架内独立测量。
- **sim 无种子随机**：复跑数值有抖动，对照表中的 sim 数值为单次运行统计。
- **spec §7 部分统计未交付**：QQ 图 / 正态性检验 / k 位置 ANOVA 未做（异方差已由 E7 覆盖，k 位置效应部分由 E1 逐位置表覆盖）；结论不依赖这些检验。
