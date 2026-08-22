# 决赛真机演示前端 · Runbook（demo-hw）

> 目标：浏览器页面上传一张遥感图 → **Gazelle 真机光计算**逐层推理 → 返回 Top-5 预测。
> 与 02_复赛 demo 的区别：那条链路连的是 **osim 容器（仿真）**；本目录连的是**真机**（路径 A：板上 server_gazelle.py :8000）。
> 链路复用 03_决赛_EuroSAT真机/opticspacenet 的 gazelle_engine（与全量 94.19%（M4）同一条推理路径）。

## 架构

```
浏览器 demo_hw/web/index.html
  └─ HTTP :8100 ─▶ 本地 FastAPI (demo_hw/server.py)
        └─ gazelle_engine.build_model(model2, HttpBackend)
              └─ HTTP 127.0.0.1:8000 (SSH 隧道) ─▶ 板上 server_gazelle.py ─▶ compass_matmul 真机光计算
```

## 0. 前置条件

1. 板上已起 server（root，环境变量传参）：`sudo env OPTC_PORT=8000 python3 server_gazelle.py`
2. SSH 隧道（本地 8000 → 板 8000），经跳板机两跳；
3. **已做 compass_cali + 四项放行判据全过**（流程同 `03_决赛.../mnist/MNIST_现场演示Runbook.md` §2–3）；
4. 本地依赖：`pip install fastapi uvicorn httpx pillow numpy torch`（Python 3.9+）。


## 1. 启动

```bash
cd 决赛提交包/04_决赛演示_真机前端

# 默认 model2 + 真机后端
uvicorn demo_hw.server:app --port 8100

# 可选配置（环境变量）：
#   HW_MODEL=model2|model3|model1a|model1b   架构（权重取自 02_复赛 weights/）
#   HW_WEIGHT=path/to/xxx.pth                指定权重文件
#   HW_BACKEND=numpy                         离线干净参考（不占板）
#   CORRECTION=calib.npz                     校准修正（可选）
```

浏览器打开 <http://127.0.0.1:8100> —— 状态灯绿色且显示「真机已连接（Gazelle 光计算）」才代表真机链路；
显示 numpy 参考模式时是离线参考，**演示时必须口头说明**。

## 2. 演示流程（建议话术）

1. 指状态灯：「现在连接的是 Gazelle 真机，光计算层的每一次矩阵乘都在板上的光学核心执行」；
2. 上传一张 EuroSAT 测试图（提前备好几张不同类别的图）；
3. 等待数秒（单图 = stem 电算 + 5 次光计算层 HTTP 往返），页面出 Top-5 概率条；
4. 对照真值标签讲正确/错误样本都可以——错误样本正好呼应 PPT「错分分析」页。

## 3. 故障排查

| 现象 | 原因 | 处置 |
|---|---|---|
| /api/health remote=down | 隧道断 / 板上 server 未起 | 重连隧道；板上重启 server_gazelle.py |
| 推理 503 超时 | 板被占用 / 热崩溃 | 按 Runbook §2 判据重查；冷却后 fresh 校准 |
| 精度明显偏低 | 校准 stale | 重新 compass_cali（20 分钟纪律），重跑 |
| 只想验证页面逻辑 | — | `HW_BACKEND=numpy` 启动，走干净参考 |

## 4. 诚实边界（答辩口径）

- 本前端默认模型为**复赛 Model 2**（int8，osim 全量 90.43%；其真机数据见 03 目录各模型全量表）；
- 页面演示的是**真机单图推理链路**（连接、校准、逐层光计算往返），不代表全量精度数字；
全量数字以 03_决赛.../文档/02_验证报告.md 为准；
- 状态灯不是绿色时不得声称在跑真机。

## 5. 已知限制 / 待联调项

- **本目录代码未在真机上实测过**（组包时按路径 A 现成组件搭建）；演示前务必按 §0–§1 完整联调一次；
- 当前 build_model registry 覆盖 model2/model3/model1a/model1b（复赛架构 + M4 所需）；
M9/M10（路径 B 板端 runner）不经此链路，如需演示 M10 请用 03 目录脚本；
- 单图延迟取决于隧道带宽（base64 已优化 ~3×），现场先测一图记个大概秒数。