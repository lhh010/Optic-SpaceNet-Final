# INT4 光计算容器推理日志

> 日期: 2026-07-10 (quick 50) → 2026-07-11 (根因调查) → 2026-07-12 (全量 5400)
> 模型: Model 2 SpaceNet V1 Phase4 v2 (INT4, 91.06%)
> 权重: spacenet_v1_phase4_v2_ste.pth
> 文件: optic_inference_int4.py

---

## 全量 5400 张 osimulator 推理 (2026-07-12)

```
python optic_inference_int4.py
```

| 指标 | 值 |
|---|---|
| 光计算准确率 | **87.94%** |
| QAT 参考 (test set, 全量) | 94.57% |
| QAT float32 (test set) | 91.43% |
| 训练参考 (val set) | 91.06% |
| 量化损失 vs QAT test | **-6.63%** |
| 总耗时 | 20316s (~5.6h) |
| 光计算占比 | 90.65% |
| 引擎调用 | 27000 次 |
| 总 MACs | 5.15e+09 |

**进度曲线:**
```
 540/5400 ( 10.0%) acc=85.74%
1080/5400 ( 20.0%) acc=87.22%
1620/5400 ( 30.0%) acc=87.47%
2160/5400 ( 40.0%) acc=87.41%
2700/5400 ( 50.0%) acc=88.00%
3240/5400 ( 60.0%) acc=87.96%
3780/5400 ( 70.0%) acc=88.07%
4320/5400 ( 80.0%) acc=88.17%
4860/5400 ( 90.0%) acc=88.00%
5400/5400 (100.0%) acc=87.94%
```

**QAT 交叉验证 (全量 5400, 2026-07-11):**
```
python optic_inference_int4.py --qat
Float32: 91.43%  |  Int4 QAT: 94.57%  |  Quant Loss: -3.15%
```

---

## 根因调查 (2026-07-11)

### ~6.6% 差距的三个来源

1. **int4→int8 权重量化网格不对齐**: QAT 训练 scale=max/7 (16级), osimulator scale=max/127 (256级).
   同一浮点权重落在不同量化值上, 每个权重 ~0.3% 相对误差.

2. **per-channel→per-tensor 激活量化退化**: QAT 每个输入通道独立 scale,
   osimulator 的 im2col 展平后所有通道共享一个 scale.

3. **stem QAT→FP32 不匹配**: 训练时 stem 参与 int4 QAT, 推理时 stem 保持 FP32 电子.
   BN 统计量基于 QAT 分布训练, 应用于 FP32 输出时有轻微偏移.

### 尝试过的修复 (全部失败或无效)

| 方案 | 结果 | 原因 |
|---|---|---|
| weight_bit=4 | 18% | osimulator 内部 8a8w 编译, int4 值被误解释 |
| BN 重校准 (bs=1) | 40.5% | 单样本 BN 统计量噪声过大 |
| per-channel + shift + correction | 71% | correction 量化误差被 shift×128 放大 |
| per-channel + scale 吸收 (干净) | 88% | 与原始相同, scale 吸收后 weight 重量化抵消收益 |

### Phase4 v4 (stem=FP32 重新训练) 也失败

用 `model2_spacenet_v1_phase4_v3.py --wbits 4` 训练 (first_conv_fp32=True, 匹配推理),
QAT 精度 91.94%, 但 osimulator 仅 84% — Gazelle 噪声使权重过度特化于 int4 网格,
int8 重量化时偏差更大.

### 结论

INT4 QAT 模型在 osimulator (8a8w) 上的实际上限是 ~88%.
推荐: 直接使用 INT8 模型 (93.28%) 或 LSQ+ 模型 (92.76%).

详见 EXPERIMENTS.md §16.

---

## 原始记录

### Quick 验证 (2026-07-10)

```
python optic_inference_int4.py --quick 50
```

| 指标 | 值 |
|---|---|
| 准确率 | ~90% |
| 训练参考 | 91.06% int4 |

### 关键发现 (2026-07-10)

INT4 容器开发过程中发现两个关键问题:

#### Bug A: stem 被转为光计算 → 46%
stem (Conv 3→8, 1×1) 展平=3 补零到 8, int4 仅 16 级精度, 输出几乎是噪声。
**修复**: `keep_first_conv_electronic=True` → 74%

#### Bug B: 激活被压到 int4 → 74%
QAT v3 训练配置是 `weight_bits=4, act_bits=8`, 但容器设了 `input_bit=4` 把激活也压成 int4。
**修复**: `input_bit=8, weight_bit=8` → ~90%

#### osimulator 不支持混合位宽
尝试 `input_bit=8, weight_bit=4` 时 osimulator 内部按 int8 解析 int4 权重值, 精度降到 10%。
**结论**: 必须统一使用 `input_bit=8, weight_bit=8`.
