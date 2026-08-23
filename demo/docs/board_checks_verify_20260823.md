# 上板判据离线审计记录（2026-08-24 修订）

> 当前没有连接 Gazelle。本文件只记录代码/离线验证，**不构成真机放行证据**。
> 2026-08-23 版本曾允许 `GAZELLE_FAKE=1` 返回 `all_pass=true`，该行为不安全，
> 已删除。

## 与 Optic-SpaceNet-Final/main 对齐后的口径

1. EBR：板端自动运行 `compass_evb_test`，两通道均 ≥8；
2. error_std：从同一次输出读取两通道，对健康基线 4.694/4.473 的恶化均 <2%；
3. MNIST canary：板端 `run_mnist_gazelle.py`，n=1000，hw/ref gap <0.5pt；
4. EuroSAT mini-run：M9/M10 板端 `run_ds3_gazelle.py` path B，默认 n=100，
   SCP 拉回 logits；真机与本地同量化参考准确率 gap <2pt。

前置条件也会阻断放行：人工确认 `who`/`ps`/`BOARD_USAGE.md` 无人占用且
冷却≥5min；对应模型校准文件身份可核验、年龄≤20min。未确认时服务端不会
发送板端工作负载。

## 已完成的离线验证

- Python/JavaScript/Bash 静态检查；
- 两通道 EBR/error_std 解析与阈值单元测试；
- 未确认占用时不触板、任一 EVB 项失败即阻止后续重负载的单元测试；
- SSH 探针失败 TTL、HTTP 权重上传失败不污染缓存的代码审计；
- `/checks.html` 不再提供手填 EBR/error_std，也不再提供 Model 3 path-B 选项。

## 真机窗口仍必须验证

- [ ] `uisrc@192.168.31.158`、sudo 5182、`~/j1`/`~/mnist` 文件名与权限；
- [ ] M9 板端目录 `weights_w075ds3`、M10 `weights_m10_5400`；
- [ ] `compass_evb_test` 与两个 runner 的实际 stdout 是否仍符合解析格式；
- [ ] SCP 拉回 `/tmp/ds3_gate_*.npy` 的权限与 logits shape；
- [ ] fresh calibration → 四项判据背靠背实测，记录时间与台账。
