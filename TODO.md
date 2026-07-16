# TODO — 后续运行清单

> 背景: test⊂train 泄漏 (Bug #11) 已在代码侧修复 (`eurosat_split.py` 单一数据源, 已 push)。
> 三模型均已在干净 split (train 16200 / val 5400 / 留出 test 5400) 上**重训完成**;
> held-out test 的 QAT 交叉验证也已完成: Model 1/2 拿到干净 **int8** test 数 (test≈val, 无泄漏),
> Model 3 的 `--qat` 是 int4 路径 (非 int8), 其 int8 test 数待 osimulator。
>
> 剩余: 三模型 osim 真机**全量**已完成 — M1 q50 抽样 (98/100%) / **M2 全量 5400 = 90.43%** / **M3 全量 5400 = 90.28%** (2026-07-16); 硬件 gap 已钉死 (两模型均 ~1.6 点, 基本相等)。**"KD 提升噪声鲁棒性" 假设不被全量支持** (q500 的 M2 89% 是抽样噪声)。后续可选: 转入光电混合提速 (H1/H2/H3, 见 `optic_inference_h{1,2,3}.py`)。
>
> 命令在 `E:\LT-Simulator\train-test` 下运行。建议每条加 `2>&1 | tee xxx.log` 存日志。

---

## 进度 (2026-07-16)

| 任务                                     | 状态                                                                               |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| Model 1 重训 A/B (干净 split)              | [x] 完成 — val A **97.87%** / B **98.02%** (日志 `log_model1_baseline_phase4_v3.md`) |
| Model 1 QAT 交叉验证 (int8, held-out test) | [x] 完成 — test A **97.89%** / B **97.96%** (test≈val, 无泄漏 ✓)                      |
| Model 2 重训 (干净 split)                  | [x] 完成 — val **92.06%** (日志 `log_model2_spacenet_v1_phase4_v3.md`)               |
| Model 2 QAT 交叉验证 (int8, held-out test) | [x] 完成 — test **92.20%** ≈ val 92.06% (无泄漏 ✓)                                    |
| Model 3 重训 (干净 split)                  | [x] 完成 — val **91.83%** (日志 `log_model3_spacenet_v2_phase4_v3.md`)               |
| Model 3 QAT 交叉验证 (held-out test)       | [x] 完成 — 但 `--qat` 是 **int4** (84.59%), 非 int8; fp32 test 92.13%≈val (无泄漏)       |
| Model 3 int8 test 数                    | [x] 完成 — osim **全量 90.28%** ≈ val 91.83% (q500 抽样 90.80% 已被取代)                |
| osimulator 真机 (三模型)                    | M1 A/B ✓ q50 (98/100%); **M2 ✓ 全量 90.43%** / **M3 ✓ 全量 90.28%** (2026-07-16)     |

> **Bug #11 修复判据已验证 (M1/M2)**: 干净 test int8 ≈ val (M1 97.89/97.96 vs 97.87/98.02; M2 92.20 vs 92.06), 不再是旧 leaky 虚高 (99.96% / 93.28%) → 修复生效。

---

## 第一优先 — osimulator 小量数据集验证 (容器内)

> 用户计划: 对 Model 1 var A/B、Model 2、Model 3 在 osimulator 上做小量数据集验证。
> Model 1 全量 ~9 天不可行, 用 `--quick N` 抽样; Model 2/3 可全量 (~4-6h) 或抽样。

- [x] Model 1 变体 A osimulator 抽样 — **98.00%** (49/50, 11614s)
  ```bash
  python optic_inference_int8_model1.py --variant A --quick 50
  ```
- [x] Model 1 变体 B osimulator 抽样 — **100.00%** (50/50, 9853s)
  ```bash
  python optic_inference_int8_model1.py --variant B --quick 50
  ```
- [x] Model 2 osimulator (int8) — **全量 5400 = 90.43%** (vs val 92.06%, Δ −1.63%; vs FP32 基准 90.15% **+0.28% 反超**) ✓
  ```bash
  python optic_inference_int8.py   # 全量已跑 (2026-07-16), 日志 log_optic_int8_new.md
  ```
- [x] Model 3 osimulator (int8, **拿到 Model 3 的 int8 test 数**) — **全量 5400 = 90.28%** ≈ val 91.83% (Δ −1.55%) ✓
  ```bash
  python optic_inference_kd.py   # 全量已跑 (2026-07-16), 日志 log_optic_kd_new.md
  ```

**关键判据**: osimulator int8 数应 **≈ 训练 val** (M1 ~97.9-98.0%, M2 ~92%, M3 ~91.8%), 而不是旧 leaky 虚高 (M2 93.28% / M3 93.26%, 作废)。
**已验证 (2026-07-16, 全量 5400)**: M2 osim **90.43%** (Δ vs val 92.06% = −1.63%); M3 osim **90.28%** (Δ vs val 91.83% = −1.55%)。两模型均干净无泄漏 ✓。**两模型真机 gap 基本相等 (~1.6 点)** — q500 时 M2 (89.00%, −3.06%) 看似远大于 M3 (90.80%, −1.03%), 全量证明那是抽样噪声 (q500 的 M2 恰落 ~1 SE 低处); **"KD 提升噪声鲁棒性" 假设不被全量支持** (M2-vs-M3 直接差 0.15%, z≈0.27 不显著)。
- osim ≈ val → 干净 ✓
- osim 明显高于 val → 还有泄漏, 停下查

已回写 EXPERIMENTS.md §11.13/§13 (用干净全量 osim 数替换标注作废的)。

---

## 已完成 (备查)

| 模型 | Int8 val | Int8 test (QAT) | osim test | 备注 |
|---|---|---|---|---|
| Model 1 变体 A | 97.87% | 97.89% | 98.00% (q50) | conv1_1 FP32, 光计算 97.74% |
| Model 1 变体 B | 98.02% | 97.96% | 100.00% (q50) | conv1_1+conv3_2 FP32, 光计算 73.64% |
| Model 2 | 92.06% | 92.20% | **90.43% (全量)** | 光计算 90.65%; 真机反超 FP32 基准 +0.28% |
| Model 3 | 91.83% | — (int4 --qat=84.59%) | **90.28% (全量)** ✓ | int8 test 已得; fp32 test 92.13% |

> M1/M2: int8 test≈val (Δ ≤ 0.14%), Bug #11 修复后无泄漏、泛化良好。
> M3: `optic_inference_kd.py --qat` 是 int4 路径 (非 int8), int8 test 须走 osimulator。
> M2/M3 osim 全量 gap 基本相等 (Δ vs val −1.63% / −1.55%), q500 抽样 (89.00% / 90.80%) 已被全量取代。
> 重训 val 比 21600-train 旧版略低 (M1 −0.28%, M2 −1.05%, M3 −0.52%), 训练集缩小 25% 所致, 非回归。

---

## 跑之前 / 跑的时候注意

1. **三模型权重均已更新**, 旧版在 git 里。
2. **`--qat` 是秒级伪量化、osimulator 是小时级真硬件仿真**: 拿 test int8 数, Model 1/2 可用 `--qat`; **Model 3 的 `--qat` 是 int4**, 其 int8 数只能 osimulator (默认 OPTIC 模式)。
3. **进度打印稀疏**: osimulator 每 10% 一次、单张 ~150s (Model 1), 长时间无输出正常。
4. **串联跑** 别并行 (CPU 抢核):
   ```bash
   python optic_inference_int8_model1.py --variant A --quick 50 2>&1 | tee osim_A.log && \
   python optic_inference_int8_model1.py --variant B --quick 50 2>&1 | tee osim_B.log && \
   python optic_inference_int8.py --quick 50 2>&1 | tee osim_model2.log && \
   python optic_inference_kd.py --quick 50 2>&1 | tee osim_model3.log
   ```
