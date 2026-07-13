# TODO — 后续运行清单

> 背景: test⊂train 泄漏 (Bug #11) 已在代码侧修复 (`eurosat_split.py` 单一数据源, 已 push)。
> 但**现有权重是修复之前训的、见过 test 图**, 所以要拿「真正没碰过的独立 test 集」数,
> 必须用修复后的 split 重训一次。val 数 (Model 1: A 98.15% / B 98.02%) 本身干净, 重训只是
> 把它升级成 held-out test 数 (消除 model-selection peek)。
>
> 命令在 `E:\LT-Simulator\train-test` 下运行。建议每条加 `2>&1 | tee xxx.log` 存日志。

---

## 进度 (2026-07-13 晚)

| 任务 | 状态 |
|---|---|
| Model 2 重训 (干净 split) | [x] 完成 — Int8 val **92.06%** (日志 `log_model2_spacenet_v1_phase4_v3.md`) |
| Model 3 重训 (干净 split) | [x] 完成 — Int8 val **91.83%** (日志 `log_model3_spacenet_v2_phase4_v3.md`) |
| Model 2/3 QAT 交叉验证 (held-out test) | [ ] 待跑 (秒级) |
| Model 1 重训 A/B | [ ] 待跑 (~12h CPU) |
| osimulator 真机 (docker) | [ ] 明天 (容器内) |

> 重训 val 比 21600-train 旧版略低 (M2 92.06 vs 93.11, M3 91.83 vs 92.35), 纯粹训练集缩小所致, 正常。

---

## 第一优先 — Model 1 拿干净 test 数 (~12h CPU)

- [ ] 重训变体 A (~6h, 现在 train=16200 / val=5400 / 留出 test 5400)
  ```bash
  python model1_baseline_phase4_v3.py --variant A 2>&1 | tee train_A_v2.log
  ```
- [ ] 重训变体 B (~6h)
  ```bash
  python model1_baseline_phase4_v3.py --variant B 2>&1 | tee train_B_v2.log
  ```
- [ ] QAT 交叉验证 (秒级) — 现在在干净的 held-out test 上跑
  ```bash
  python optic_inference_int8_model1.py --variant A --qat --batch 256 2>&1 | tee qat_A_v2.log
  python optic_inference_int8_model1.py --variant B --qat --batch 256 2>&1 | tee qat_B_v2.log
  ```

**关键判据**: 重训后 QAT 的 test 数应为 **~98% (接近 val), 而不是 99.96%**
- ~98% → 修复生效, test 集真的干净了
- 仍 ~99.96% → test 还在泄漏, 停下找我查

跑完把 4 个 log 发回, 我回写 EXPERIMENTS.md §11.11 / §13 (用干净 test 数替换之前标注作废的)。

---

## 第二优先 — osimulator 真机抽样 (容器内, 准备好再做)

> 之前说「容器内先不跑」, 准备好了再做。Model 1 全量 ~9 天不可行, 只抽样。

- [ ] 容器内, 变体 A 干净 test 抽样 (~2h)
  ```bash
  python optic_inference_int8_model1.py --variant A --quick 50
  ```
- [ ] 容器内, 变体 B 干净 test 抽样 (~2h)
  ```bash
  python optic_inference_int8_model1.py --variant B --quick 50
  ```

---

## 第三优先 — Model 2/3 清一遍 (重训已完成, test/osimulator 待跑)

> Model 2/3 重训已完成 (2026-07-13, 干净 split train 16200)。日志:
> `log_model2_spacenet_v1_phase4_v3.md` / `log_model3_spacenet_v2_phase4_v3.md`

- [x] Model 2 重训: Int8 val **92.06%** (Float32 91.76%, 量化损失 -0.30%)
- [x] Model 3 重训: Int8 val **91.83%** (Float32 91.65%, 量化损失 -0.19%)
- [ ] Model 2/3 QAT 交叉验证 (秒级, held-out test 干净数) — test int8 应 ≈ val, 不再虚高
  ```bash
  python optic_inference_int8.py --qat --batch 256     # Model 2
  python optic_inference_kd.py --qat --batch 256       # Model 3
  ```
- [ ] Model 2/3 osimulator 全量 (容器/docker, ~4-6h 每个) — 明天做
  ```bash
  python optic_inference_int8.py          # Model 2
  python optic_inference_kd.py            # Model 3
  ```

---

## 最小可行方案 (时间紧时)

只跑**第一优先的变体 A** (重训 + QAT, ~6h), 就能拿到一个干净的 test int8 数,
足够说明 int8 量化无损 + 干净泛化精度。其余按需补。

---

## 跑之前 / 跑的时候注意

1. **train 变小是正常的**: 重训日志开头会显示 `训练: 16200` (不是 21600) —— test 那 5400 张被剔出来了, 这是修复后的正确行为。
2. **旧权重会被覆盖**: 重训会重新生成 `baseline_vgg_phase4_v3_int8.pth` / `..._vB.pth` (旧的已 commit 在 git 里, 不会丢)。
3. **每 5 epoch 才打印一行**, 中途几分钟没输出是正常的。
4. **串联跑** (A 跑完接 B): 一条命令搞定, 别并行 (CPU 会抢核):
   ```bash
   python model1_baseline_phase4_v3.py --variant A 2>&1 | tee train_A_v2.log && \
   python model1_baseline_phase4_v3.py --variant B 2>&1 | tee train_B_v2.log
   ```
