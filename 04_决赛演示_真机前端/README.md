# 决赛真机演示前端 v3 · Runbook（demo-hw）

> 2026-08-23 v3 改版：**demo 从 pathA(HTTP matmul) 切到 pathB(板端 runner 直调 compass)**。
> 实测 pathA 每次光算 matmul 约 10s(与规模无关, 单图 90s+)，无法做实时演示；
> pathB 板上 runner 直调 compass ~3.2s/张(与全量运行同链路)。

## 架构

```
浏览器 demo_hw/web/index.html
  └─ HTTP :8100 ─▶ 本地 FastAPI (demo_hw/server.py)
        └─ board.py (paramiko SSH + sudo) 触发板上 runner
              └─ run_ds3_gazelle.py (M10, ~3.2s/张, 95.33% canonical 链)
              └─ run_mnist_gazelle.py / run_mnist_official.py (官方200)
```

## 0. 前置

1. 板子内网可达(当前 `192.168.31.158`, 连「小米主路由器」wifi test1234, uisrc/5182)；
2. 板上有运行脚本：`~/j1/run_ds3_gazelle.py`、`~/j1/weights_m10_5400`、`~/mnist/run_mnist_official.py` + 官方200 npy；
3. **已完成 fresh compass_cali + 标量校准**（见「校准」节）；
4. 本地依赖：`pip install fastapi uvicorn paramiko pillow numpy torch torchvision`。

## 1. 启动

```bash
cd 决赛提交包/04_决赛演示_真机前端
uvicorn demo_hw.server:app --port 8100
```
| 环境变量 | 默认 | 说明 |
|---|---|---|
| `HW_MODEL` | `model10` | model10→weights_m10_5400, model9→weights_w075ds3 |
| `HW_CALIB` | 空 | 板上标量 calib json 文件名(如 `calib_scalar_m10_0823.json`) |
| `BOARD_HOST` | `192.168.31.158` | 板上 SSH 地址 |
| `HW_CHECK_N` | `8` | 判据④ mini-run 张数 |

状态灯：绿=SSH 通 + runner 可用(pathB)；「真机不可达」=SSH 不通。

## 2. 校准（15 min 方案）

板上执行(需 root)：
```bash
compass_cali --mode-local                          # fresh bringup ~10min
cd ~/j1 && DS3_WEIGHTS_DIR=weights_m10_5400 \
  DS3_CALIB_OUT=calib_scalar_m10_0823.json \
  python3 calibrate_any_ds3.py                     # 标量 per-layer ~3min
```
注：`calibrate_any_ds3.py` 必须在 `compass_cali` 完成后运行(否则 device busy)，且需 root(api.log)。
Demo 通过 `HW_CALIB=calib_scalar_m10_0823.json` 引用。

## 3. 演示流程

1. 状态灯绿；
2. 点「运行 ③④ 判据」：③ MNIST canary、④ EuroSAT mini-run(默认 8 张)；
3. 「EuroSAT 真机跑批」：默认 [0,8) 张(~26s)，可调 1 张/10 张；显示 hw acc + 参考 + gap；
4. 「MNIST 官方抽样」：跑官方 200 张(或小数张)；显示 acc/ref/gap。

## 4. 已知限制 / 待联调

- 路径 B 跑的是**板上测试集 npy**(`weights_m10_5400/test_images_j1.npy`)，非任意上传图——演示为「测试集批量跑批」形态；
- `calibrate_any_ds3.py` 标量校准精度略低于逐列(col 96.4%)，约 94-95%；需要 96%+ 则用逐列(probe+calibrate_col, ~25min)；
- MNIST 官方200 已上板(`~/mnist/test_images_official200.npy` + `run_mnist_official.py`)；
- 校准跨窗口必失效，演示前同窗口重新校准。