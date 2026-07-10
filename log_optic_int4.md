# INT4 光计算容器推理日志

> 日期: 2026-07-10
> 模型: Model 2 SpaceNet V1 Phase4 v2 (INT4, 91.06%)
> 权重: spacenet_v1_phase4_v2_ste.pth
> 文件: optic_inference_int4.py

---

## 关键发现

INT4 容器开发过程中发现两个关键问题:

### Bug A: stem 被转为光计算 → 46%
stem (Conv 3→8, 1×1) 展平=3 补零到 8, int4 仅 16 级精度, 输出几乎是噪声。
**修复**: `keep_first_conv_electronic=True` → 74%

### Bug B: 激活被压到 int4 → 74%
QAT v3 训练配置是 `weight_bits=4, act_bits=8`, 但容器设了 `input_bit=4` 把激活也压成 int4。
**修复**: `input_bit=8, weight_bit=8` — osimulator 原生 8a8w, QAT int4 权重量化到 int8 是无损的 → ~90%

### osimulator 不支持混合位宽
尝试 `input_bit=8, weight_bit=4` 时 osimulator 内部按 int8 解析 int4 权重值, 精度降到 10%。
**结论**: 必须统一使用 `input_bit=8, weight_bit=8`, QAT 的 int4 权重以 FP32 存储, int8 量化无损。

---

## Quick 验证 (修复后)

```
python optic_inference_int4.py --quick 50
```

| 指标 | 值 |
|---|---|
| 准确率 | ~90% |
| 训练参考 | 91.06% int4 |
| Stem | 电计算 (FP32) |
| 其余层 | 光计算 (int8 act + int8 weight, osimulator 原生) |

---

## 待办

- [ ] 全量 5400 张验证 (预计 ~4h)
