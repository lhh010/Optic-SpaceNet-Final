# TODO — 后续运行清单

> 背景: test⊂train 泄漏 (Bug #11) 已在代码侧修复 (`eurosat_split.py` 单一数据源, 已 push)。
> 三模型均已在干净 split (train 16200 / val 5400 / 留出 test 5400) 上**重训完成**;
> Model 1 还跑完了 held-out test 的 QAT 交叉验证 (test≈val, **确证无泄漏**)。
>
> 剩余: Model 2/3 的 held-out test QAT 交叉验证, 以及三模型的 osimulator 真机数。
>
> 命令在 `E:\LT-Simulator\train-test` 下运行。建议每条加 `2>&1 | tee xxx.log` 存日志。

---

## 进度 (2026-07-14)

| 任务 | 状态 |
|---|---|
| Model 1 重训 A/B (干净 split) | [x] 完成 — val A **97.87%** / B **98.02%** (日志 `log_model1_baseline_phase4_v3.md`) |
| Model 1 QAT 交叉验证 (held-out test) | [x] 完成 — test A **97.89%** / B **97.96%** (test≈val, 无泄漏 ✓) |
| Model 2 重训 (干净 split) | [x] 完成 — val **92.06%** (日志 `log_model2_spacenet_v1_phase4_v3.md`) |
| Model 3 重训 (干净 split) | [x] 完成 — val **91.83%** (日志 `log_model3_spacenet_v2_phase4_v3.md`) |
| Model 2/3 QAT 交叉验证 (held-out test) | [ ] 待跑 (秒级) |
| osimulator 真机 (三模型) | [ ] 待跑 (容器内) |

> **Bug #11 修复判据已验证**: Model 1 重训后 test 数 (97.89% / 97.96%) **≈ val**, 不再是旧 leaky 版的虚高 99.96% → 修复生效, test 集干净。

---

## 第一优先 — Model 2/3 拿干净 test 数 (秒级)

> Model 2/3 重训已完成, 现只缺 held-out test 的干净 int8 数。

- [ ] Model 2 QAT 交叉验证 (held-out test 干净数) — test int8 应 ≈ val 92.06%, 不再虚高
  ```bash
  python optic_inference_int8.py --qat --batch 256 2>&1 | tee qat_model2_v2.log
  ```
- [ ] Model 3 QAT 交叉验证 (held-out test 干净数) — test int8 应 ≈ val 91.83%
  ```bash
  python optic_inference_kd.py --qat --batch 256 2>&1 | tee qat_model3_v2.log
  ```

**关键判据**: test int8 数应 **≈ val** (M2 ~92%, M3 ~91.8%), 而不是旧 leaky 版的虚高 (旧 M2 osim 93.28% / M3 osim 93.26%, 已作废)。
- test ≈ val → 干净 ✓
- test 明显高于 val → 还有泄漏, 停下查 split

跑完把 log 发回, 回写 EXPERIMENTS.md §11.12 / §13 (用干净 test 数替换标注作废的 osimulator 数)。

---

## 第二优先 — osimulator 真机 (容器/docker, 准备好再做)

> 容器内跑。Model 1 全量 ~9 天不可行, 只抽样; Model 2/3 全量 ~4-6h 可接受。

- [ ] Model 1 变体 A 干净 test 抽样 (~2h)
  ```bash
  python optic_inference_int8_model1.py --variant A --quick 50
  ```
- [ ] Model 1 变体 B 干净 test 抽样 (~2h)
  ```bash
  python optic_inference_int8_model1.py --variant B --quick 50
  ```
- [ ] Model 2 osimulator 全量 (~4-6h)
  ```bash
  python optic_inference_int8.py
  ```
- [ ] Model 3 osimulator 全量 (~4-6h)
  ```bash
  python optic_inference_kd.py
  ```

---

## 已完成 (备查)

| 模型 | Int8 val | Int8 test | 光计算占比 | 备注 |
|---|---|---|---|---|
| Model 1 变体 A | 97.87% | 97.89% | 97.74% | conv1_1 FP32 |
| Model 1 变体 B | 98.02% | 97.96% | 73.64% | conv1_1 + conv3_2 FP32 |
| Model 2 | 92.06% | (待跑) | 90.65% | — |
| Model 3 | 91.83% | (待跑) | 90.65% | int8 + KD |

> Model 1 test≈val (Δ ≤ 0.06%), Bug #11 修复后无泄漏、泛化良好。详见 EXPERIMENTS.md §11.11。
> 重训 val 比 21600-train 旧版略低 (M1 −0.28%, M2 −1.05%, M3 −0.52%), 纯粹训练集缩小 25% 所致, 非回归。

---

## 跑之前 / 跑的时候注意

1. **重训已完成**: 三模型权重均已更新 (`baseline_vgg_phase4_v3_int8{,_vB}.pth` / `spacenet_v{1,2}_phase4_v3_int8.pth`), 旧版在 git 里。
2. **`--qat` 是秒级、osimulator 是小时级**: 拿 test int8 数用 `--qat`; 真硬件仿真去掉 `--qat` (默认 Optic 模式)。
3. **进度打印稀疏**: QAT 每 batch 一次; osimulator 每 10% 一次、单张 ~150s (Model 1), 长时间无输出正常。
4. **串联跑** 别并行 (CPU 抢核):
   ```bash
   python optic_inference_int8.py --qat --batch 256 2>&1 | tee qat_model2_v2.log && \
   python optic_inference_kd.py --qat --batch 256 2>&1 | tee qat_model3_v2.log
   ```
