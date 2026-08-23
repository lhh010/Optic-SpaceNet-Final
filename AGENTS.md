# Optic-SpaceNet 决赛提交包 · 项目记忆（AGENTS）

> 本文件 = 提交包仓库的长期记忆，记录板子访问、demo 经验、校准要点。与 `contest-national/AGENTS.md`、`train-test/AGENTS.md` 保持同步。

## 板子访问（2026-08-23 起内网直连）

- **板子内网直连 `192.168.31.158`**；演示机连 WiFi「小米主路由器」密码 `test1234`；板内 `uisrc`/`5182`，sudo 密码 5182。
- **不再走跳板机/隧道**——旧的 `-J huadong3564@140.206.121.211:2036 uisrc@10.102.13.37`（免密 key `~/.ssh/id_ed25519_gazelle`）公网路径已失效（板子迁内网后那张网不通）。
- 演示机需与板子同网（USB 网络共享 / 连小米路由器）。自动化用 `paramiko`（已装）：`echo 5182 | sudo -S <cmd>` 或 PTY `sudo -i`。
- **禁位置参数**（compass_sdk 篡改 sys.argv），一律环境变量；`sudo env VAR=...`。
- FPGA m≥3 行回绕 bug → **路径 B 板端 runner 用 m≤2 tiling**；路径 A 的 `server_gazelle.py` 不做 m≤2（历史大 m 正常）。

## demo 前端（`04_决赛演示_真机前端/demo_hw/`）经验

1. **路径 B 才适合演示**：路径 A（本地 forward + HTTP 逐 matmul 到板上 server）**每次光算 matmul ~10s（与规模无关）**，单图 90s+ 无法实时；路径 B（板上 runner 直调 compass）**~3.2-5s/张**（M10 canonical 链）。demo 用 `board.py`（paramiko SSH+sudo 触发 runner + 解析）。
2. **标量校准够演示（90-95%），逐列冲 96%**（probe+`calibrate_col.py` ~25min，C2 96.40%）。`calibrate_any_ds3.py` 快但略低。
3. **fresh `compass_cali` 必做且能大修板子**：离线 2 天后 EBR 4.19→9.70（≥8）、error_std 224→4.9（基线）。
4. **校准顺序**：`compass_cali --mode-local` 先完整跑完，再单独 `calibrate_any_ds3.py`；不同时（device busy / Aborted 134）；都要 root（写 api.log）。
5. **compass_sdk 篡改 sys.argv** → 自定义板端脚本用环境变量传参（如 `run_mnist_official.py` 用 `MNIST_LIMIT`），禁位置参数。
6. **demo 端点**：`/api/run/eurosat`（逐图真机/参考 Top5+一致性）、`/api/mnist/run`（官方200）、`/api/checks/ebr`（自动测）、`/api/checks/canary`(n=1000)、`/api/calibrate`（后台重校准+轮询）。
7. **判据②**：error_std 与**基线**对比（健康~4.7 counts），非绝对<2%。
8. **交互**：判据① EBR 独立容器（不被③④覆盖）；`Cache-Control: no-cache` 必须加（否则浏览器缓存旧 JS→`rowCard` 内层双引号截断致整段 script 解析失败、状态卡「检测中」）；`_read_npy` 参数名 `remote_path`。
9. **清板**：跑批/校准进程可残留（`sudo pkill -9 -f run_ds3_gazelle.py` 等）；`/tmp/ds3_logits_*.npy` 累积用 `sudo rm -f` 清。

## 校准 15 分钟方案

```bash
compass_cali --mode-local                          # ~10min (root)
cd ~/j1 && DS3_WEIGHTS_DIR=weights_m10_5400 DS3_CALIB_OUT=calib_scalar_<ts>.json \
  python3 calibrate_any_ds3.py                     # ~3min (root)
```
demo 用 `HW_CALIB=calib_scalar_<ts>.json` 引用；也可用前端顶部「重新校准」按钮（`/api/calibrate`，后台线程，完成后自动切换）。

## 离线/仅局域网运行

demo 运行**不依赖外网**（本地权重/数据/官方 MNIST + 板子内网）；只需提前装好依赖。见 `04_决赛演示_真机前端/README.md` §1.5。