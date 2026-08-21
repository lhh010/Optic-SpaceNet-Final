# C1 板上取证报告 — 2026-08-09 抢占事件（操作记录侧）

取证人：C1-forensics 子任务｜取证时间：2026-08-09 06:51–07:10 UTC｜方式：全程只读
原始证据：`x0/data/incident_20260809/forensics/`（basic.txt / logins.txt / sshd.txt / authlog_sudo_accept.txt / auth_stats.txt / server_code.txt / server_gazelle_src_and_logs.txt / tmp_mtimes.txt / histories.txt / compass_cali_script.txt）

## 0. 关键前提：双方身份如何区分

板上只有一个用户 **uisrc**（root 无直接登录记录，`lastlog` 全 Never）。两队均从 **10.102.12.4**（内网跳板/隧道源）接入，IP 无法区分双方。可区分特征（证据：`auth_stats.txt`）：

- **我方**：唯一 ED25519 公钥指纹 `SHA256:hDpObY8qfbRZIB7CA+LWyAoTWhCZLH/9I1CoZRPNOv4`（`Accepted publickey`，Aug 6-9 共 309 次全部同一指纹）
- **他队**：`Accepted password`（密码认证，Aug 5 首次出现）
- 所有 root 操作无论谁做都会留 `/var/log/auth.log` 的 `sudo: COMMAND=` 记录——这是本次取证的主证据源

## 1. 事件时间线（UTC，对齐到分钟）

| 时间 | 事件 | 证据 |
|---|---|---|
| 08-08 04:44:56 | 我方 sudo compass_cali | authlog L77 |
| 08-08 04:54–07:29 | 我方 c2c/c3d/c3h/c3f 标定+跑批（calibrate_any.py / run_j1_gazelle.py） | authlog L78-102 |
| 08-08 08:22:16–08:26:05 | **他队 6 次 password 登录**，无任何 sudo 命令；其中 pts/0（08:23:36）挂起至今未退出 | authlog L106-111；`who` |
| 08-09 05:11–05:16 | 我方登录，05:16:07 查看 cali.log | authlog L116-124 |
| **08-09 05:20:55** | **我方 sudo compass_cali（fresh cali）** | authlog L133 |
| 08-09 05:32:03 | 我方 compass_evb_test | authlog L137 |
| 08-09 05:32:56 | 我方 MNIST canary（scale, LIMIT=1000） | authlog L139 |
| 08-09 05:33:58 | 我方 probe_dump_c3d（calib_col_c3d.json 于 05:43 写完） | authlog L141；文件 mtime |
| **08-09 05:38:35–05:44:18** | **他队 19 次 password 登录爆发** | authlog L142-164 |
| 08-09 05:39:44 | 他队 sudo start_server.sh —— **1 次密码错误** | authlog L146 |
| 08-09 05:42:45 | 他队 sudo `timeout 25 python3 server_gazelle.py` —— **2 次密码错误**（25s 超时自杀） | authlog L155 |
| 08-09 05:43:05 | 他队 sudo 无密码尝试（"a password is required"） | authlog L158 |
| **08-09 05:44:18** | **他队 sudo start_server.sh 成功** → server_gazelle.py 以 root 启动（pid 14371），compass_init(150) 重写器件配置，listen :8000 | authlog L165；server.log；server_info.txt |
| 08-09 05:44:42 / 05:46:31 | 我方 run_j1_gazelle 两次启动 → 与他队 server **器件争用**（server.log 末尾 "can't open device : Device or resource busy"） | authlog L167-168；server.log |
| 08-09 05:50–05:59 | 我方应急处置：查进程/ss/读 server.log；05:56:53 compass_matmul 自检 OK；05:58:47 MNIST canary 200；05:59:29 run_j1（fresh calib） | authlog L171-190 |
| **08-09 ~06:00–06:05** | **他队 server 退出**（server.log 最后写入 06:00；我方 06:41 快照记录 "observed gone by ~06:05"）；他队 06:00:57–06:01:16、06:10:11 又有 4 次 password 登录 | server_info.txt；authlog L191-199 |
| 08-09 06:37:19 | kill 16481 16482（紧邻 publickey 会话，判定为我方清理残留）；06:37:36 evb_test | authlog L208-210 |
| 08-09 06:41:33 | 我方拷 server.log/api.log 到 j1/incident_tmp/ 保全现场 | authlog L215-217 |
| 08-09 06:42:21 | 我方 probe_post（post-incident 探针） | authlog L220 |
| **08-09 06:47:35** | **我方 sudo compass_cali（恢复验证）** | authlog L229 |
| 08-09 06:44:40 / 06:48:39 / **07:01:10×2** | **他队仍在登录**（与我方 06:47 恢复 cali 并发！） | authlog L224, L230；auth_stats |

EBR 前后对比（server_info.txt，我方主 agent 快照）：
- baseline 05:33：error_mean [0.6726, -0.1269]，error_std [4.671, 4.497]，ebr [9.776, 9.831]
- post-incident 06:37：error_mean [1.0708, 0.4646]，error_std [4.952, 4.710]，ebr [9.692, 9.764]
- → EBR 基本没变，但 ch0 error_mean 系统性偏移 +0.4（bias 型污染，非增益退化）

## 2. 他队行为画像

1. **没有跑 compass_cali**。Aug 8-9 全部 sudo 日志中 compass_cali 仅 3 次（08-08 04:44:56、08-09 05:20:55、08-09 06:47:35），均紧邻我方 publickey 会话；他队 password 会话期间零 cali 记录。其 server 代码（`server_gazelle.py`，已存档）也只调 `compass_init(150)` + `compass_matmul`，不含任何 calibrate 调用。
2. **server 干了什么**：`/home/uisrc/opticspacenet/server_gazelle.py`（他队 08-06 11:14–14:55 创建）是一个 HTTP :8000 JSON matmul 服务（POST /matmul → compass_matmul），root 常驻、单线程、全局持有光器件。启动脚本 `start_server.sh` 带 `pkill` 自重启逻辑。
3. **器件状态被改写（未校准）**：server.log 显示其 `compass_init(150)` 过程中——设激光电流 83.48mA、写入 tx calibration LUT 值（sw_val/dac_val 数组）、配置增益引脚；且出现 **`ltc2265 verify communication failed, read reg value is:135`** 的硬件通信失败，以及与我方进程争用导致的 **"Device or resource busy"**。**未跑 cali 即重写器件配置 + 通信校验失败 + 争用中写入**，是"他队退出后器件系统性偏差"的头号机制候选（最终归因以数据侧 agent 为准）。
4. **sudo 密码是猜/试出来的**：05:39:44（1 次错误）、05:42:45（2 次错误）、05:43:05（不知道要密码），05:44:18 才成功——说明他队此前不知道 sudo 密码，当天现场试出（5182 为弱共享密码）。
5. 他队不改器件配置文件的直接证据未见（无可编辑的配置文件路径痕迹），污染途径是 compass_init 的寄存器写入。

## 3. 登录/使用模式统计（auth.log Aug 4–9）

- password（他队）：Aug 5 ×2（首次）→ **Aug 6 ×96（全天重度 06:45–16:12，建 opticspacenet）** → Aug 7 ×0 → Aug 8 ×6（早间试探）→ **Aug 9 ×26（事件爆发 05:38–05:44 + 后续零散，至 07:01 仍在登录）**
- publickey（我方）：Aug 6 ×37 / Aug 7 ×115 / Aug 8 ×85 / Aug 9 ×73（截至 06:53）
- 结论：**他队是常态使用者（Aug 5 起），非偶发**；两队无任何错峰/排队机制，08-09 05:38 他队上线时完全不检查器件是否被占用（我方 canary 05:33 刚跑完）→ SOP 必须上排队/锁协议（如器件锁文件 + 使用前 `ss -tlnp`/`ps` 检查 + 错峰时间窗）。
- 定时任务：uisrc/root 均无 crontab；systemd timers 全为系统默认（apt/motd/fstrim/tmpfiles）——**无自动任务干扰项**。

## 4. 不可达/缺失项

- `uisrc/.bash_history`：root:root 600，uisrc 无权读；内容为 2026-01-28 的陈旧文件（装机期），交互 shell 历史从未落盘 → 他队交互命令细节不可考，但 sudo 日志已覆盖其全部 root 操作。
- `root/.bash_history`（794B，sudo cat 已读）：仅装机期命令（装 SDK、setmac），无窗口内内容。
- server 无 HTTP access log（代码里 `log_message` 被禁用）→ :8000 收到过哪些请求不可考；api.log（低层寄存器 trace）最后写入 05:44，推测 server 启动后未服务多少请求即陷入争用。
- `journalctl _COMM=sshd` 只有 Disconnect 记录，Accepted 记录全在 auth.log（已完整提取）。

## 5. 风险提示（给恢复工作流）

他队在 **06:44:40 / 06:48:39 / 07:01:10 仍有 password 登录**，与我方 06:47:35 启动的恢复 compass_cali 并发。他队会话 pts/0（08-08 08:23 起）仍挂起。若其再次 start_server，将重演争用。建议恢复验证完成后尽快确认 :8000 无监听、无 server_gazelle 残留。
