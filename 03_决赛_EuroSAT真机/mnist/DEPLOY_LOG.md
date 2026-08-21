# MNIST 真机部署 · 排查与进展记录

> 日期：2026-08-06 ｜ 状态：✅ **已跑通（1000 样本 94.80%，校准后）**
> 目标：初赛 MNIST（DSQ 3 层 MLP）在远程 Gazelle 真机上跑通并达到合理精度

---

## 1. 已完成

### 1.1 远程连接链路（已打通）
```
本地 WSL ──ssh -p 2036 huadong3564@140.206.121.211──▶ 跳板机 sh-xizhi
          └─ssh uisrc@10.102.13.37──▶ Gazelle（sudo 免密可用，`sudo -n` 需先缓存凭据）
```
- 非交互密码方案：`SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force`（本机脚本 `scratch/sshpass_ask.sh`）。
- `scp -J` 直连在该网络不可用（banner 超时），采用官方"两段式"：本地↔跳板机↔Gazelle。
- 执行：`echo 5182 | sudo -S python3 ...`（uisrc 密码即 5182）。

### 1.2 MNIST 推理管线已在真机跑通
- 部署目录：`/home/uisrc/mnist/`（脚本 `run_mnist_gazelle.py` + 权重 + 数据，全部上传）。
- 1000 样本 × 3 层光计算 **24 秒跑完**（真机速度无问题）。
- 脚本关键点：**不能有位置参数**（见 §2.1 的 sys.argv 陷阱）。

### 1.3 排查结论（重要）
| 问题 | 根因 | 修复 |
|---|---|---|
| 脚本报 `usage: run_mnist_gazelle.py [--mode-local]...` | **compass_sdk 导入时篡改 sys.argv**：`system_setup.py` 模块级执行 `sys.argv.extend(['--proj',...])`，`compass_init()` 内部 `parse_args(sys.argv[1:])` → 位置参数全被它拦截 | 脚本禁用位置参数，改用环境变量（`MNIST_LIMIT`/`MNIST_BATCH`/`MNIST_MODE`） |
| 硬件准确率 12%（≈随机） | 板子 EBR 仅 **5.08/4.93**（规格 ≥8）；输入/权重值域太小（uint4/int4 只占满量程 6%），信号淹没在 DAC/ADC 噪声里，负权重臂甚至符号反 | **运行官方校准 `compass_cali`**；MNIST 侧改用 `MNIST_MODE=scale`（×16 放大到满 8bit 再算，÷256 还原）或 `advance` |
| `compass_cali` 报 scipy 错误 | 板上 `scipy/_lib/decorator.py` 是**悬空符号链接**（指向丢失的 apt `decorator.py`） | `ln -sf /usr/local/lib/python3.6/dist-packages/decorator.py /usr/lib/python3/dist-packages/scipy/_lib/decorator.py`，`scipy.signal OK` |

### 1.4 关键事实（读 SDK 源码确认）
- `compass_init(150)` 的 150 是 **ad_delay**（ADC 读取延时），不是"TIA 参数"。
- `compass_matmul(vec, weight)`：vec uint8、weight int8；返回 = 原始 MAC × `tia_gain_scale_factor`（255 或 25.5，由板卡配置决定）；argmax 对该均匀缩放不敏感。
- `compass_matmul_advance(A, B)`：按 8 一组自动缩放 A→0..255、B→-127..127，**专门解决小数值信噪比问题**。
- 硬件真实精度以 `compass_evb_test`（10000 组 1×8@8×2）为准：`error_mean/std` + `ebr`。

---

## 2. 校准结果（✅ 成功）

`compass_cali` 校准（约 10 分钟）后，EBR 从 5.08/4.93 提升到 **9.74/9.68**（≥8 达标）：

| 指标 | 校准前 | 校准后 |
|---|---|---|
| error_mean | [66.5, 84.0] | [1.32, 0.34] |
| error_std | [121.4, 134.0] | [4.79, 4.98] |
| EBR | [5.08, 4.93] | [9.74, 9.68] |

## 3. MNIST 真机结果（✅ 跑通）

| 配置 | 准确率 |
|---|---|
| **真机硬件**（完整 10000 测试集，scale 模式） | **94.85%** |
| NumPy 参考（同量化路径） | 94.80% |
| 硬件 vs 参考 | **+0.05 点（几乎一致）** |
| 真机硬件（1000 样本，两次） | 94.80% / 94.70% |

> 结论：校准后真机精度与软件理想路径几乎完全一致（gap 0.05 点），MNIST DSQ 全流程在 Gazelle 真机验证通过。
> scale 模式 = 输入 ×16（uint4→uint8 满量程）、权重 ×16（int4→int8 满量程），结果 ÷256 还原——解决小数值信噪比问题。
> 10000 样本全流程约 4 分钟。

## 4. 待办（后续）
1. （可选）跑全量 10000 测试集确认最终数字。
2. 复赛 OpticSpaceNet 迁移（更重：CNN + 光电混合 + 容器 osimulator → 真机 compass_sdk）。

## 3. 板载环境备忘
- Python 3.6.9；numpy 1.13.3；scipy 0.19.1（已修 decorator 链接）。
- compass_sdk 1.0.2 装于 `/usr/local/lib/python3.6/dist-packages/compass_sdk`（**编译产物 .so**，源码参考本地 `contest-national/sdk_src/compass_python_v1p02/`）。
- 板有外网（archive.ubuntu.com / pypi 可达）。
- 校准产物：`compass_cali` 会生成 LUT 等文件（写回 SDK config 目录），日志 `/home/uisrc/cali.log`。
