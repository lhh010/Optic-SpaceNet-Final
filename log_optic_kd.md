# KD+INT4 光计算容器推理日志

> 日期: 2026-07-10
> 模型: Model 3 SpaceNet V2 KD Phase4 v2 (KD + INT4, 91.50%)
> 权重: spacenet_v2_phase4_v2_ste.pth
> 文件: optic_inference_kd.py

---

## Quick 验证 (per-channel 量化修复前)

```
python optic_inference_kd.py --quick 200
```

| 指标 | 值 |
|---|---|
| 准确率 | 83.50% (200 张) |
| QAT 对照 | 96.00% (50 张, 统计波动) / FP32: 88.00% |
| 训练参考 | 91.50% int4 KD |
| 引擎调用 | 1000 次 |
| 总 MACs | 1.91e+08 |

### 根因

KD 训练的权重通道间分布差异大, `_matmul_real` 的 per-tensor 量化浪费了大量精度。
**修复**: 改为 per-channel 权重量化 (`optic_layers.py` 已更新)。

## Quick 验证 (per-channel 修复后, 待更新)

```
python optic_inference_kd.py --quick 50
```

待重新运行验证。

---

## 待办

- [ ] Quick 50 验证 per-channel 修复效果
- [ ] 全量 5400 张验证 (~4h)
