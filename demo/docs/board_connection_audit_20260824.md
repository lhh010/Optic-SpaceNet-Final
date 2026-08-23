# Gazelle 连接与上板判据严格审计（2026-08-24）

## 结论

代码已按 `Optic-SpaceNet-Final/main`（审计提交
`c18bdfc5bb1f0ab1ffa980f6d297fbb0a41674c7`）区分两条真机链路：

- 任意上传图片的逐层可视化：path A，SSH 隧道/串口发现 IP 后调用板端
  `server_gazelle.py /matmul`；激活在本地 forward 中形成，因此无需逐层 SCP；
- 正式上板放行：path B，SSH 启动板端 canonical runner，EuroSAT logits
  通过 SCP 拉回并与本地同量化 NumPy 参考比较。

当前没有连接 Gazelle，不能声明真机通过。实测离线探针：path-B SSH 首次约
5.018s 返回 timeout，10s TTL 内再次查询约 0.00003s；path-A SSH 首次约
8.137s，缓存查询约 0.00002s，前端不会每次轮询都重复卡住。

## 连接历史核对

- 当前有效设计：演示机与板卡同连「小米主路由器」，直连
  `uisrc@192.168.31.158`，SSH/sudo 口令由环境变量提供（默认仍为 5182）。
- 旧记录：`ssh -J huadong3564@140.206.121.211:2036
  uisrc@10.102.13.37`，私钥 `~/.ssh/id_ed25519_gazelle`。`final/main/AGENTS.md`
  明确该公网跳板路径在板卡迁入内网后已失效，不应再回退。
- 用户最初给出的桌面文件 `AI交接文档_20280823.md` 当前已不存在；旧跳板机
  记录仍保存在本仓库 `AGENTS.md`，本次以该文件和 final/main 原始文档/代码
  交叉核对。

## 放行判据对齐

| 项 | 当前实现 | 放行条件 |
|---|---|---|
| 人工占用确认 | `/api/checks/usage` 只读 who/ps/台账，前端要求人工勾选 | 无人占用、冷却≥5min |
| 进程互斥 | 自动查 server_gazelle/runner/校准进程 | path B 前必须为空 |
| 校准 | 模型名可核验的 scalar json + mtime | 对应 M9/M10，年龄≤20min |
| ① EBR | 自动运行一次 EVB，解析两通道 | 两通道均≥8 |
| ② error_std | 与同一次 EVB 共用采样 | 相对 4.694/4.473 的恶化均<2% |
| ③ MNIST | 板端 `run_mnist_gazelle.py` | n=1000，hw/ref gap<0.5pt |
| ④ EuroSAT | 板端 `run_ds3_gazelle.py` + SCP logits | 默认n=100，hw/ref acc gap<2pt |

说明：验证报告仍把 200 张称为正式 SOP；final/main 当前演示代码默认 100、允许
1–500。本页跟随当前 main 默认 100，现场可把输入改为 200。历史文档提到
error_std “2%（10%宽限）”，当前默认执行严格 2%；如比赛现场明确批准宽限，
可显式设置 `HW_ERROR_STD_TOL=10`，不应静默放宽。

## 已修复的连接/安全问题

1. EBR/error_std 从“可选手填”改为板端自动双通道解析；缺项不再可能
   `all_pass=true`。
2. `GAZELLE_FAKE=1`/NumPy 离线模式不再产生上板 PASS。
3. health 增加 10s 成功/失败 TTL；SSH 错误包含实际 stderr。
4. SSH 本地端口 18080 被陌生进程占用时明确失败，不再误把任意监听者当隧道。
5. HTTP 权重只有成功上传后才写本地缓存；板端重启/淘汰缓存后遇到
   `unknown weight_id` 会自动重传。
6. 新版 base64 协议失败时会尝试旧版 JSON-list `server_gazelle` 合约。
7. HTTP matmul 响应校验 data、元素数和 finite；错误响应保留板端正文。
8. 串口只从命令 marker 之间解析合法 IPv4，优先 `192.168.31.*`；串口仍只是
   console/IP bootstrap，矩阵数据随后走 HTTP，未伪装成串口 RPC。
9. 校准文件名加入 M9/M10 标签；旧版无法判断模型身份的通用 json 会阻断放行。
10. 校准、EVB、启动/停止 server 前增加占用/冷却确认；密码不再通过
    `sshpass -p` 暴露在本机进程参数中。
11. path-B 放行接口加进程内互斥；放行运行时 Gazelle path-A 推理返回 409。

## 未连接真机时最可能遇到的问题（按风险排序）

1. **网络不在同一网段**：未连小米路由器、WiFi 密码/网段变化、主机路由或
   Docker bridge 无法到达 192.168.31.158，会表现为 5–8 秒 SSH timeout。
2. **path A/path B 冲突**：桌面 Gazelle 启动器会为逐层展示启动
   `server_gazelle.py`；正式 path-B 判据会检测到它并阻断。需要分时使用，不能
   一边逐层推理一边跑 canonical gate。
3. **板端文件名/目录与假设不同**：代码按 main 假设 `~/j1`、`~/mnist`、
   M9 `weights_w075ds3`、M10 `weights_m10_5400`。任一缺失会直接失败。
4. **校准不可用**：当前本地未发现新鲜、带模型标签的校准文件；超过20min、
   M9/M10 混用、scalar/col 格式误用都会阻断或显著掉点。
5. **“三模型并行”耗时误解**：浏览器并发发请求，但板端 HTTP server 是单线程，
   光器件不会并行执行三模型。path A 已知约90s+/图/模型，三模型排队可能逼近
   300s HTTP 超时。
6. **账号/权限变化**：uisrc 密码、sudo 密码、sudo policy、root 写 api.log、
   SCP 读取 root 生成 logits 的权限均未实机复核。
7. **runner 输出格式变化**：MNIST accuracy 正则、EuroSAT `FINAL:` 和 logits
   `(n,10)` 契约来自当前 main；板端脚本若不是同一版本会解析失败。
8. **串口现场差异**：设备名、dialout 权限、console 已被占用、登录提示延迟、
   `uisrc/root` console 口令或网卡 IP 变化都可能导致发现失败；即便串口登录成功，
   Docker 仍必须能路由到发现的 IP:8000/22。
9. **板端 path-A 服务版本**：如果板上是更旧的 server，base64 会回退 JSON-list；
   若连 weight_id 缓存也没有，则当前逐层客户端仍不兼容，需要先同步仓库版本。
10. **物理窗口本身不健康**：即使 SSH、EBR 和 canary 正常，共享板残余瞬态、
    stale 校准和热漂移仍可能让 EuroSAT mini-run 失败；这正是第四判据不可省略的原因。

## 真机窗口建议顺序

1. 连接内网，读取 who/ps/BOARD_USAGE，确认冷却；
2. 停止自己启动的 path-A server（不得误杀他队进程）；
3. fresh `compass_cali --mode-local`，再做对应模型 scalar/col 校准；
4. 在 `/checks.html` 运行四项（建议最终用 n=200 留档）；
5. 记录校准/判据时间与台账；
6. 若要演示任意图片逐层结果，再启动 path-A server；由于启动会重新初始化器件，
   应把 path-A 结果单独标为“逐层展示链路”，不得沿用为 path-B 正式精度证据。
