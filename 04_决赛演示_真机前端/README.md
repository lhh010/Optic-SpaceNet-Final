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
| `HW_CHECK_N` | `100` | 判据④ mini-run 张数(默认100, 约6-7min, 可调8~500) |

状态灯：绿=SSH 通 + runner 可用(pathB)；「真机不可达」=SSH 不通。

## 1.5 离线/仅局域网运行（现场无外网时）

> demo **运行本身不依赖外网**：权重/数据/官方 MNIST 全在本地，板子走内网(192.168.31.158)。唯一需要外网的是**首次 `pip install` 依赖**——提前装好或带离线 wheelhouse。

**开跑前务必确认（缺一不可）**：
1. **依赖已装**：`fastapi uvicorn paramiko pillow numpy torch torchvision`（`pip list` 确认；无外网时用 `pip install --no-index --find-links <wheelhouse>` 装或带 `python` 环境整包）；
2. **本地材料就位**：`03_决赛_EuroSAT真机/eurosat_research/weights/m10_ds3pool3_v8probe15.pth`、`02_复赛_EuroSAT仿真/data/EuroSAT_RGB`（gitignore，本地已在）、`04_.../CICC_2026_MNIST_TEST_DATASET/`（官方200，已入库）；
3. **板子内网可达**：连「小米主路由器」WiFi(test1234)，`Test-NetConnection 192.168.31.158 -Port 22` 通过；板上 `run_ds3_gazelle.py`/`weights_m10_5400`/`run_mnist_official.py`+官方200 已就位；
4. **已完成 fresh `compass_cali` + 标量校准**（否则精度低；`/api/calibrate` 可在线重校准，但需约10-12min）。

**离线启动序列（无网络依赖）**：
```powershell
# 1) 连内网 WiFi 小米主路由器(test1234) → 确认板子可达
Test-NetConnection 192.168.31.158 -Port 22   # True
# 2) 起 demo（指向板子 + 校准）
cd E:\LT-Simulator\决赛提交包\04_决赛演示_真机前端
$env:BOARD_HOST="192.168.31.158"; $env:HW_MODEL="model10"; $env:HW_CALIB="calib_scalar_m10_0823b.json"
python -m uvicorn demo_hw.server:app --port 8100
# 3) 浏览器打开 http://127.0.0.1:8100（无需外网）
```
- 全程只需局域网：浏览器→demo(本地)→SSH(板子)→光算。无任何外网请求。
- 若现场要用另一台机器当浏览器访问，uvicorn 加 `--host 0.0.0.0`，且那台机器连同一局域网即可。
- 校准/判据/跑批/逐图对比/MNIST 官方 200 全部离线可用。

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
2. 点「运行 ③④ 判据」：③ MNIST canary(n=1000)、④ EuroSAT mini-run(页面「④ 张数」可调, 默认100) ；
3. 「EuroSAT 真机跑批」：默认 [0,8) 张，可调 1~200；显示 hw acc + 参考 + gap + 逐图真机/参考对比；
4. 「MNIST 官方抽样」：跑官方 200 张(或小数张)；显示 acc/ref/gap。

## 4. 已知限制 / 待联调

- 路径 B 跑的是**板上测试集 npy**(`weights_m10_5400/test_images_j1.npy`)，非任意上传图——演示为「测试集批量跑批」形态；
- `calibrate_any_ds3.py` 标量校准精度略低于逐列(col 96.4%)，约 94-95%；需要 96%+ 则用逐列(probe+calibrate_col, ~25min)；
- MNIST 官方200 已上板(`~/mnist/test_images_official200.npy` + `run_mnist_official.py`)；
- 校准跨窗口必失效，演示前同窗口重新校准。