# 决赛真机演示前端 v2 · Runbook（demo-hw）

> 2026-08-23 v2 改版：
> ① **模型换 M10 ds3pool3（默认）/ M9 w075ds3**——本地 numpy 前向（`ds3net.py`，逐行镜像板端 runner 数值语义），光算 matmul 直连板上 `server_gazelle.py`（`OPTC_HOST` 可直接指板 IP，不再假设 SSH 隧道）；
> ② 判据③升级为**真 MNIST canary**（DSQ 三层 MLP，初赛 97.35% 链路）；
> ③ 新增 **MNIST 官方抽样 200 张跑批** 与 **EuroSAT 分段跑批**（按 30min 窗口预算设计）。

## 架构

```
浏览器 demo_hw/web/index.html
  └─ HTTP :8100 ─▶ 本地 FastAPI (demo_hw/server.py)
        └─ ds3net.forward (M10/M9 numpy 前向, 镜像 run_ds3_gazelle.py)
              └─ 7 光算层 matmul → HTTP OPTC_HOST:OPTC_PORT ─▶ 板上 server_gazelle.py
                   (m≤2 tiling = M10 全量 95.33% canonical 链路, HW_CHUNK 可调)
```

## 0. 前置条件

1. 板上已起服务：`sudo env OPTC_PORT=8000 python3 server_gazelle.py`（root，环境变量传参）；
2. 本地能直达板上 8000 端口（直连网段或隧道均可——只要求 `OPTC_HOST:OPTC_PORT` 可达）；
3. 已完成 fresh `compass_cali`（~10min）+ 四项放行判据全过；
4. 本地依赖：`pip install fastapi uvicorn pillow numpy torch torchvision`。

## 1. 启动

```bash
cd 决赛提交包/04_决赛演示_真机前端
uvicorn demo_hw.server:app --port 8100
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `HW_MODEL` | `model10` | `model10`（ds3pool3, SOTA 95.33%）/ `model9`（w075ds3, 94.43%） |
| `HW_BACKEND` | `http` | `numpy` = 离线干净参考（不占板） |
| `OPTC_HOST/PORT` | `127.0.0.1:8000` | 板上 server_gazelle 地址，**可直接指板 IP** |
| `HW_CHUNK` | `2` | 光算 matmul 行分块；2=FPGA 行回绕规避（canonical），板上 server 实测支持大 m 时可调大加速 |
| `HW_CALIB_COL` | 空 | 逐列校准 json（calibrate_col.py 产物，同窗口！） |
| `DS3_HEAD_ELEC` | `0` | `1`=head FC 走电算 |
| `HW_CHECK_N` | `10` | 判据④ mini-run 张数 |
| `HW_MNIST_DIR` | 空 | 主办方抽样图目录（png + 可选 `labels.txt` 每行 `name,label`）；空=内置官方测试集前 N 张 |

状态灯：绿=真机探针真实可达（5s 主动探测，无假绿）；「真机不可达」=配置 http 但板不通；numpy=离线模式。

## 2. 30 分钟窗口预算（C2 实测）

| 项 | 耗时 |
|---|---|
| fresh compass_cali | ~10 min |
| 四项放行判据 | ~5 min |
| **跑批段** | **~15 min** |

- 实测吞吐（板端 runner 口径）：**M10 3.23s/张、M9 1.98s/张**；
- 15 min ≈ M10 280 张 / M9 450 张 → **默认段 [0,200)**（M10 ~11min，留裕量）；
- 全量 5400（M10 ~4.8h）单窗口不可行——现场只跑抽样段，全量数字用日志/图表证据链；
- ⚠️ 本前端走 HTTP 直连，m≤2 tiling 下单图 ~250 次往返，**慢于板端 runner**——首次联调实测单图延迟后再定段长；板上 server 实测接受大 m 时可 `HW_CHUNK=1024` 提速；
- MNIST 200 张：秒级~分钟级，无压力。

## 3. 演示流程（建议话术）

1. 指状态灯：「直连 Gazelle 真机，每次矩阵乘在光学核心执行」；
2. 点「运行②③④」讲判据设计 → 板端读 EBR 录入 → 四灯全绿；
3. 单图推理（备好不同类别 EuroSAT 图；错误样本呼应错分分析页）；
4. MNIST 官方抽样 200 张一键跑（DSQ 初赛链路，gap ≤0.1pt 口径）；
5. 有余力再跑 EuroSAT [0,200) 段（或 M9 切换演示）。

## 4. 故障排查

| 现象 | 处置 |
|---|---|
| 「真机不可达」 | 检查板上 server_gazelle / 网络 / OPTC_HOST；探针 5s 超时即报，无假绿 |
| 推理 503 | 板被占用/热崩溃 → 冷却≥1h + fresh 校准重开窗 |
| 跑批精度偏低 | 校准 stale（calib 不可跨窗口）→ 重新 probe+calib_col 背靠背 |
| 只验页面逻辑 | `HW_BACKEND=numpy` 启动 |

## 5. 诚实边界（答辩口径）

- 页面演示真机单图/抽样段链路（连接、判据、逐层光算），**不代表全量数字**；全量以 02_验证报告口径表为准（M10 95.33% / M9 94.43%）；
- MNIST 200 为官方抽样现场验证；canary/mini-run 为放行判据演示；
- 状态灯不是绿色时不得声称在跑真机。

## 6. 已知限制 / 待联调项（板恢复后第一优先）

- **真机侧未实测**（板子 2026-08-21 失联，预计 08-23 下午恢复）；联调重点：单图延迟（HTTP m≤2 vs 大 chunk）、判据②读数 ±2% 内、canary gap、MNIST 200 全流程；
- `HW_MNIST_DIR` 官方抽样图就位后需现场导入并核对 labels.txt 格式；
- 逐列校准 json 必须同窗口生成（`HW_CALIB_COL`），跨窗口必失效（stale −12.5pt 实证）。