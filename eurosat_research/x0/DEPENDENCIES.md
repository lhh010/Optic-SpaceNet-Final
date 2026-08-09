# Round X0 — 依赖图与进度看板

> 通信协议：每个子任务开工时写 `progress/<task>.status`（首行 IN_PROGRESS / DONE / FAILED + 时间戳），
> 结果写 `results/<task>_*.md`。数据：`data/probe_pairs/`（板上取回，5 层 × {hw, ideal, xint}）。

## 依赖图

```
probe_pairs (data/) ──┬─> A1 δW 结构分解 ──────> 解读 B3 / 设计 v10
                      ├─> A2 非线性残余建模 ───> v10 噪声模型候选参数
                      ├─> A3 权重分布×残差 ────> B2 正则/wd 臂设计
                      └─> (已完成) C3 三组分 ──> qat_v8 现役
c3d logits + labels ──> A4 per-class hw 错误分布 → 是否需新建模
models.py 旋钮 ───────> B1-prep (代码+configs) ──> B1 训练 (GPU) ──> proxy_v8 ──> C5 hw
calibrate_real.py ────> C1-prep (逐列 calib 代码) ──> C1 板上执行 ──> B4 余量重扫
A3 ──> B2 (wd 扫描可独立先行; kurtosis 正则待 A3)
A1 ──> B3 结果解读 (训练本身不阻塞)
```

## 资源

- **GPU**：docker context `fdusc-cpu-135`，容器 `gazelle_sim`，2× A800 80GB 空闲。
  容器内代码 `/workspace/Ltsimulator-test/auto_research/`（= 本地 eurosat_research 旧命名，已到 c3d/r8）。
  启动：`docker -c fdusc-cpu-135 exec -d gazelle_sim bash -c '...'`；PY=`/local/miniconda/envs/moca_llm/bin/python`
- **开发板**：`ssh -J huadong3564@140.206.121.211:2036 uisrc@10.102.13.37`（免密）。
  pairs 在 `/home/uisrc/j1/probe_*`；大跑前 fresh `compass_cali` + canary；calib→跑批背靠背；m≤2 tiling。

## 任务状态（详看 progress/）

| 任务 | 内容 | 资源 | 依赖 | 状态 |
|---|---|---|---|---|
| A1 | δW 低秩/块结构分解 | 本地 | pairs | wave1 |
| A2 | 非线性残余 MLP/查表建模 | 本地 | pairs | wave1 |
| A3 | 权重 kurtosis × per-column 残差 | 本地 | pairs + c3d 权重 | wave1 |
| A4 | c3d per-class hw 错误分布 | 本地 | c3d logits + EuroSAT labels | wave1 |
| B1-prep | models.py 下采样旋钮 + configs | 本地→容器 | — | wave1 |
| C1-prep | 逐列 calib 代码 + BN 折叠部署 | 本地 | — | wave1 |
| B1-train | MaxPool3 / pool4 / BlurPool / conv3x3s2 臂 | GPU | B1-prep | wave2 |
| C1-board | 逐列 calib 上板验证 | 开发板 | C1-prep | wave2+ |
| B3 | rf_s2k3/w200 × v8 重训 | GPU | —（解读待 A1） | wave2 |
