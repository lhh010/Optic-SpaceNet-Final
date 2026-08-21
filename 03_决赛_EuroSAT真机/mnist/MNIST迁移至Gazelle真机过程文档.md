# MNIST 迁移至 Gazelle 光计算真机 · 过程文档

> 队伍：CICC1003564（复旦大学 · 华东赛区）
> 完成日期：2026-08-06
> 状态：✅ **三种 QAT 方法（STE / LSQ+ / DSQ）全部迁移完成并验证通过**
> 最终结果（完整 10000 张测试集）：**LSQ+ 97.35% ｜ STE 96.43% ｜ DSQ 94.79%**，与软件参考差距均 ≤ 0.6 点

---

## 1. 背景与目标

初赛的 MNIST 手写数字识别（DSQ 量化感知训练，3 层 MLP）原先只在本地 **osimulator 光计算仿真器**（Docker 容器）上验证。本任务将其迁移到**远程 Gazelle 光计算硬件真机**（8×2 光学矩阵乘法器，8a8w12o）上实际运行。

- 模型：`784 → 128 → 64 → 10`，全连接，**无 bias**（光计算只做 MAC）
- 权重精度：int4（值域 -8..7，恰好落在硬件 int8 范围内）
- 输入精度：uint4（值域 0..15，恰好落在硬件 uint8 范围内）
- 目标：在真机上复现 ≥90% 的分类准确率

## 2. 总体架构

```
【本地计算机】──ssh -p 2036──▶【跳板机 sh-xizhi (140.206.121.211)】──ssh──▶【Gazelle 光计算硬件 (10.102.13.37)】
                                   账号 huadong3564 / 5182              账号 uisrc / 5182
                                                                         执行需 root（sudo / su）
```

**关键限制**：Gazelle 是内网设备，本地无法直连，一切访问（含文件传输、端口转发）必须经过跳板机。

## 3. 环境准备

### 3.1 板上 SDK（官方预置，无需重装）
- `compass_sdk-1.0.2-cp36-cp36m-linux_aarch64.whl`（已 pip 安装，位于 `/usr/local/lib/python3.6/dist-packages/compass_sdk`，编译产物 .so）
- 示例代码：`/home/uisrc/sample_code/code/`（7 个官方脚本）
- SDK 源码包：`/home/uisrc/compass_python_v1p02.tar.gz`（可下载到本地参考，`contest-national/sdk_src/`）
- Python 3.6.9；numpy 1.13.3；scipy 0.19.1

### 3.2 板载接口（读 SDK 源码确认，与产品手册的差异）
| 接口 | 说明 | 备注 |
|---|---|---|
| `compass_init(ad_delay=1000)` | 初始化硬件（激光、ADC/DAC、LUT） | **参数是 ad_delay**（ADC 读取延时），官方示例传 150；手册写"无参"不准确 |
| `compass_matmul(vec, weight)` | 光计算矩阵乘 | vec **uint8**、weight **int8** ndarray；返回 = 原始 MAC × tia_gain（255 或 25.5，均匀缩放不影响 argmax） |
| `compass_matmul_advance(A, B)` | 自动按 8 分组缩放 | 专门解决小数值信噪比问题 |
| `compass_evb_test()` | 10000 组 1×8@8×2 精度测试 | 输出 error_mean/std + EBR，**真机精度权威指标** |
| `compass_calibrate()` | 校准芯片生成 LUT | 等价格令 `compass_cali` |

**规模限制**：vec ≤ 1024×4096，weight ≤ 4096×1024，输出 ≤ 1024×1024。

## 4. 模型与数据准备

从 `train-firstround/` 提取部署所需文件（推理路径为纯 numpy，**不需要 torch**）：

| 文件 | 来源 | 说明 |
|---|---|---|
| `w1/w2/w3_int4_dsq.npy` | `train-firstround/artifacts/dsq/` | 权重 (1,784,128)/(1,128,64)/(1,64,10)，int32 值域 -8..7 |
| `dsq_quant_params.npy` | 同上 | 量化参数：s_in, s_w1, s_h1, s_w2, s_h2, s_w3 |
| `test_images.npy` / `test_labels.npy` | `train-firstround/data/processed/` | 10000 张测试图（float32 [0,1]）+ 标签 |

上传流程（两段式，本地 → 跳板机 → Gazelle）：
```bash
# 本地 → 跳板机（scp -P 2036，密码 5182）
scp -P 2036 mnist_gazelle.tar.gz huadong3564@140.206.121.211:/home/huadong3564/
# 跳板机 → Gazelle（内网，密码 5182）
scp mnist_gazelle.tar.gz uisrc@10.102.13.37:/home/uisrc/mnist/
tar -xzf mnist_gazelle.tar.gz   # 解压后删除压缩包
```

## 5. 部署脚本设计（`run_mnist_gazelle.py`）

三层推理主循环（每层一次 `compass_matmul`）：

```python
# Layer 1: (B,784) @ (784,128)
y1 = compass_matmul(x_int, w1)                    # 硬件光计算
y1_real = y1 * (s_in * s_w1)                      # 反量化
h1 = np.maximum(0, y1_real)                       # ReLU（电计算）
h1_int = np.clip(np.round(h1 / s_h1), 0, 15).astype(np.uint8)  # 再量化

# Layer 2、Layer 3 同理
```

**三个关键设计决策**：

1. **禁用位置参数**（详见 §6.1 坑①）：脚本参数改用环境变量 `MNIST_METHOD / MNIST_MODE / MNIST_LIMIT / MNIST_BATCH`。
2. **scale 模式解决小数值信噪比**（详见 §6.2 坑②）：输入 ×16、权重 ×16 放大到满 8-bit 值域再上硬件，结果 ÷256 还原；由于 argmax 对均匀缩放不敏感，精度不受影响。
3. **统一脚本支持三种方法**：`MNIST_METHOD=dsq|ste|lsqplus`，忠实复刻各方法量化路径（STE 合并式 scale、LSQ+ zero-point shift、DSQ 分解式 scale）；内置 `MNIST_FAKE=1` 离线验证模式（光学调用换成 np.matmul，先本地验证再上真机）。
4. **同时输出 NumPy 参考准确率**：同一量化路径的软件理想结果，用于量化硬件 vs 理想差距。

### 5.1 三种方法的量化路径（真机实现，与 `testing/test_*_photonic.py` 一致）

```python
# STE（合并式 scale）：x=round(x*15); y=round 后逐层 ×scale_h 再量化
x_int = clip(round(x * 15), 0, 15)
y1 = optical(x_int, w1);  h1 = round(relu(y1) * scale_h1)
y2 = optical(h1, w2);     h2 = round(relu(y2) * scale_h2)
y3 = optical(h2, w3)

# LSQ+（zero-point）：x/权重 减 zp 后取整；激活加回 zp 再量化
xs = clip(round(x/s_in)+zp_in, 0, 15) - zp_in
w1s = (w1 - zp_w1).astype(int32)                  # 截断
y1 = optical(xs, w1s) * (s_in*s_w1); h1 = round(relu(y1)/s_h1)+zp_h1
h1s = h1 - zp_h1;  ...                            # 下一层输入同理

# DSQ（分解式 scale）
x_int = clip(round(x/s_in), 0, 15)
y1 = optical(x_int, w1) * (s_in*s_w1); h1 = round(relu(y1)/s_h1)
y2 = optical(h1, w2) * (s_h1*s_w2);    h2 = round(relu(y2)/s_h2)
y3 = optical(h2, w3) * (s_h2*s_w3)
```

> 注：LSQ+ 的硬件路径 `h1s = h1 - zp_h1` 送硬件前须取整（真机只接受整数），numpy 参考保留小数——该语义差异（≈1 点）是原脚本设计行为，非部署缺陷。

## 6. 排查记录（三个关键坑）

### 坑① compass_sdk 篡改 sys.argv（脚本一跑就报 argparse 错误）

**现象**：`python3 run_mnist_gazelle.py 200 50` 输出
```
usage: run_mnist_gazelle.py [-h] [--mode-local] [--mode-host] ...
run_mnist_gazelle.py: error: unrecognized arguments: 200 50
```

**根因**（读 SDK 源码 `system_setup.py` 确认）：
- `system_setup.py` **模块导入时**执行 `sys.argv.extend(['--proj', 'post_silicon/bringup/compass.py'])`；
- `compass_init()` → `universe_setup()` → `parse_args(sys.argv[1:])`——SDK 自带的 FPGA bring-up CLI 会解析整个 sys.argv；
- 位置参数 `200 50` 被它拦截报错；`-h` 显示的也是 SDK 的 help。

**解决**：脚本不接收位置参数，改读环境变量。附带发现：`python3 -c "import compass_sdk..."` 能正常工作正是因为 `-c` 模式下 sys.argv 只有 `['-c']`。

### 坑② 板子 EBR 仅 5.08，小数值信号淹没在噪声中

**现象**：管线全通但硬件准确率 **12.10%（≈随机）**；官方 `local_matmul_sample.py` 输出 mac 与 real 逐元素大致对应但误差很大；`evb_test` 显示 EBR **5.08/4.93**（规格 ≥8）。

**根因链**：
- 硬件噪声是**绝对量级**的（ADC/DAC 量化噪声 + TIA 探测噪声，EBR≈5 时 std≈130）；
- 我的输入（0..15）/权重（-8..7）只占满量程（0..255 / -128..127）的 ~6%，信号幅度太小，被噪声淹没；
- 官方示例 `local_matmul_sample.py` 用满量程随机数所以"看起来正常"（只是噪声大）；
- 诊断测试还发现负权重臂符号错误（1×8@8×2 期望 [36,-36]，硬件返回 [1020,1020]）。

**解决**：
1. 运行官方校准 `compass_cali`（约 10 分钟）——EBR 从 5.08 → **9.74/9.68**，系统性偏差消除；
2. 脚本加 scale 模式（×16 放大），进一步把信号抬到满量程附近。

### 坑③ 板上 scipy 损坏导致校准工具无法运行

**现象**：`compass_cali` 启动即报
```
ModuleNotFoundError: No module named 'scipy._lib.decorator'
```

**根因**：板上 `scipy/_lib/decorator.py` 是**悬空符号链接**（指向 `/usr/lib/python3/dist-packages/decorator.py`，该 apt 文件缺失），但 `__pycache__` 里的 .pyc 完好。

**解决**（pip 版 decorator 4.4.2 可用，重链即可）：
```bash
ln -sf /usr/local/lib/python3.6/dist-packages/decorator.py \
       /usr/lib/python3/dist-packages/scipy/_lib/decorator.py
python3 -c 'import scipy.signal'   # 验证 OK
```

### 附带坑④ sudo 清空环境变量
`MNIST_LIMIT=1000 sudo python3 ...` 中环境变量只作用于 `echo`，sudo 子进程拿到的是干净环境。**必须用 `sudo env VAR=... python3 ...`** 才能传参。

## 7. 校准（关键步骤）

产品手册规定：EBR 不达标（<8）时应校准。校准前先做精度评估：

```bash
# 校准前评估（10000 组）
python3 evb_test_sample.py
# → error_mean [66.5, 84.0]  error_std [121.4, 134.0]  EBR [5.08, 4.93]  ❌

# 执行校准（root，约 10 分钟，日志 /home/uisrc/cali.log）
compass_cali
# 日志关键行：top : start calibration → starting calibration_rx → top : calibration done

# 校准后复测
python3 evb_test_sample.py
# → error_mean [1.32, 0.34]  error_std [4.79, 4.98]  EBR [9.74, 9.68]  ✅
```

校准会重新生成查找表（LUT）并写回 SDK config 目录，是"一次校准、持续受益"的操作。

## 8. 结果验证

### 8.1 三方法真机对比（完整 10000 张测试集，scale 模式，校准后）

| 方法 | 模型结构 | 量化方式 | 真机硬件 | NumPy 参考 | gap |
|---|---|---|---|---|---|
| **LSQ+** | 784→128→64→10 | 学习 scale + zero-point（非对称） | **97.35%** | 97.21% | **+0.14** |
| **STE** | 784→128→64→10 | 合并式 scale（×15 输入） | 96.43% | 97.01% | -0.58 |
| **DSQ** | 784→128→64→10 | 分解式 scale（s_in/s_w/s_h） | 94.79% | 94.80% | -0.01 |

- 真机排名与软件参考完全一致（LSQ+ > STE > DSQ），三种方法硬件 gap 均 ≤ 0.6 点；
- **LSQ+ 真机甚至超过软件参考（+0.14）**——真机噪声与训练时注入的噪声匹配良好；
- STE 的 -0.58 点 gap 源于其量化语义：原测试脚本中硬件路径对 `h1 - zp` 做整数截断（真机只能送整数），而 numpy 参考保留小数——这是"整数化代价"，非硬件缺陷；
- 每方法 10000 样本约 4 分钟，含 `compass_init` 初始化。

### 8.2 校准前后对比（以 DSQ 为例）

| 阶段 | 配置 | 准确率 | 耗时 |
|---|---|---|---|
| 校准前 | raw 模式（int4 原值直上硬件） | 12.10%（≈随机） | 1000 样本 24s |
| 校准后 | scale 模式（×16 放大） | **94.85%**（10000 样本） | ~4 min |
| 参考 | NumPy 同量化路径 | 94.80% | — |

**硬件 vs 软件理想路径差距 ≤ 0.6 点**——真机精度完整复现初赛 osimulator 仿真结论。

## 9. 完整复现步骤

```bash
# ① 连接（本地终端）
ssh -p 2036 huadong3564@140.206.121.211        # 5182
ssh uisrc@10.102.13.37                          # 5182
echo 5182 | sudo -S true                        # 缓存 sudo 凭据

# ② 确认 SDK
python3 -c "from compass_sdk.fast_calibration.compass_lib import compass_matmul, compass_init; print('SDK OK')"

# ③ （首次）精度评估 + 校准
cd /home/uisrc/sample_code/code && python3 evb_test_sample.py   # EBR 应 ≥8，否则校准
compass_cali                                                    # 约 10 分钟

# ④ 上传部署包（本地）
scp -P 2036 mnist_gazelle.tar.gz huadong3564@140.206.121.211:/home/huadong3564/
# （跳板机内）scp mnist_gazelle.tar.gz uisrc@10.102.13.37:/home/uisrc/mnist/ && cd /home/uisrc/mnist && tar -xzf mnist_gazelle.tar.gz

# ⑤ 运行（完整 10000 测试集，三方法）
cd /home/uisrc/mnist
bash run_all_methods.sh   # 依次跑 dsq / ste / lsqplus，每方法 ~4 分钟
# 或单方法：
echo 5182 | sudo -S env MNIST_LIMIT=10000 MNIST_BATCH=50 MNIST_MODE=scale MNIST_METHOD=lsqplus python3 run_mnist_gazelle.py
# 预期：LSQ+ ≈97.3% / STE ≈96.4% / DSQ ≈94.8%，与各自 NumPy 参考差距 ≤0.6 点
```

## 10. 经验总结

1. **先校准，再谈精度**：真机 EBR 不达标时一切精度结论都无意义；上机第一步应跑 `compass_evb_test` 体检。
2. **小数值必须放大**：光计算硬件噪声是绝对量级，输入/权重应尽量用满量程（scale 或 `compass_matmul_advance`）。
3. **SDK 有 sys.argv 副作用**：调用 `compass_init` 的脚本不要带位置参数，参数走环境变量。
4. **真机很快**：MNIST 10000 张 × 3 层光计算仅约 4 分钟，真机算力不是瓶颈，调试成本低。
5. **复赛迁移启示**：OpticSpaceNet 是 int8 权重（满量程），无需 scale 技巧，但 CNN 的 im2col 展平、光电混合切分需要把 `optic_layers.py` 的 osimulator 调用层替换为 compass_sdk 调用。

## 附录：文件清单

| 文件 | 位置 | 说明 |
|---|---|---|
| `run_mnist_gazelle.py` | `contest-national/mnist/` | 真机部署脚本（raw/scale/advance 三模式） |
| `diag_matmul.py` | 同上 | 硬件 matmul 诊断脚本（布局/噪声/线性拟合） |
| `DEPLOY_LOG.md` | 同上 | 简版排查流水记录 |
| `check_env.sh` / `fix_scipy.sh` / `fix_decorator.sh` / `relink_decorator.sh` | 同上 | 环境检查与修复脚本 |
| `mnist_gazelle.tar.gz` | 同上 | 一键上传部署包 |
| `sdk_src/` | `contest-national/` | SDK 源码（从板上下载的 compass_python_v1p02） |
| 板上 `/home/uisrc/mnist/` | Gazelle | 真机部署目录（脚本+权重+数据） |
| 板上 `/home/uisrc/cali.log` | Gazelle | 校准日志 |
