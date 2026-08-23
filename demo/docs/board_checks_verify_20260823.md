# 上板判据验证记录（2026-08-23 · 离线链路验证）

> 验证环境：Docker 容器 `gazelle-demo:1.0`（torch 2.13 CPU + fastapi），
> `GAZELLE_FAKE=1`（numpy 参考模式），**未连接 Gazelle 真机**。
> 真机判据需按 SOP 开窗执行（fresh compass_cali → 四判据背靠背）。

## 一、验证内容与结果

| 项 | 结果 | 说明 |
|---|---|---|
| 镜像构建 `gazelle-demo:1.0` | ✅ 60.7s | python:3.10 + torch CPU + fastapi + picocom/sshpass/pyserial |
| `demo.server.app` import | ✅ 11 路由 | 依赖齐全；optic_layers 检测打印已静默 |
| `/api/checks/all`（FAKE） | ✅ all_pass=True | 探针 0.00% / canary·mini-run 报 numpy 参考模式 |
| `/api/checks/ebr`（录入） | ✅ ① EBR 9.7→pass | ② error_std 4.55（低于基线，改善）→pass |
| `/api/checks/ebr`（恶化） | ✅ 5.10（+8.51%）→fail | SOP：仅恶化超阈值判 fail |
| `/` 与 `/checks.html` | ✅ 200 | 演示页 + 上板检查页 |
| test200 数据生成 | ✅ 200 张 | eurosat_split seed42，与 canonical 同源 |

## 二、上板判据（SOP，global/AGENTS.md）

四项**全部达标**才开窗：

1. **EBR ≥ 8**：板上 `compass_evb_test` 读数 → 前端「上板检查」录入；
2. **error_std 对基线 <+2%**：evb error_std 录入（低于基线视为改善自动通过）；
   自动快速探针：真机已知矩阵 vs numpy 参考 rel<2%；
3. **MNIST canary gap <0.5pt**：DSQ 三层 ×16 scale，官方抽样 200 张
   （权重/数据自包含 `demo/server/mnist_res/`），真机 vs 同量化 numpy；
4. **EuroSAT mini-run 正常**：当前模型（model3/9/10）真机 vs numpy 干净参考，
   逐图一致率≥80% 且 acc 正常（n=200）。

开窗纪律：who/ps 侦测他队占用 + 读板上台账 BOARD_USAGE.md 尾部；
fresh compass_cali 后判据背靠背；calib json 同窗口 20 分钟内使用（stale −12.5pt）；
他人使用后 ~40min 物理瞬态 → fresh 校准 → 判据重过。

## 三、真机上板流程（tools/）

```bash
bash tools/start.sh        # 菜单: 环境自动就绪
  [4] 连接/检查硬件         # SSH 状态 / 启停 server_gazelle / EBR / 串口(picocom)
  [1] 校准                  # compass_cali → probe → 标量/逐列 calib → 拉回 json
  前端「上板检查」四判据全过 → 开窗
  [3] M9/M10 200 张抽样验证 # 或前端 /checks.html mini-run
```

## 四、遗留（真机窗口执行项）

- [ ] 板上 192.168.31.158 工具链/权重包存在性确认（calib_board.sh 自动检查/上传脚本）
- [ ] server_gazelle.py 运行确认 + 台账尾部 + 占用侦测
- [ ] fresh compass_cali + EBR/error_std 读数录入
- [ ] canary/mini-run 真机实测（GAZELLE_FAKE=1 时跳过，仅链路验证）
