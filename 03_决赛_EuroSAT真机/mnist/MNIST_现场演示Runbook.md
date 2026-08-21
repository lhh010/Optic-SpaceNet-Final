# MNIST 真机现场演示 Runbook（决赛现场速查）

> 目标：在 Gazelle 真机上现场跑通 MNIST 三层 MLP 光计算推理（DSQ/STE/LSQ+ 三方法可选）。
> 详细过程记录见同目录 `MNIST迁移至Gazelle真机过程文档.md` 与 `DEPLOY_LOG.md`；本文件只留现场操作。

## 0. 连接链路

```
本地 ──ssh -p 2036 huadong3564@140.206.121.211──▶ 跳板机 sh-xizhi
     └─ssh uisrc@10.102.13.37──▶ Gazelle 真机
```

板上部署目录：`/home/uisrc/mnist/`（脚本 + 权重 + 数据）。

## 1. 环境自检（30 秒）

```bash
cd /home/uisrc/mnist && bash check_env.sh
```

## 2. 开窗放行判据（每次演示前必查，四项全过才跑）

1. **EBR ≥ 8**：`compass_cali` 后跑 `compass_evb_test` 看输出（实测历史：校准前 5.08 → 校准后 9.74）
2. error_std 对基线偏差 < ±2%
3. MNIST canary gap < 0.5pt
4. 小批量 mini-run 正常（如 200 张精度正常）

> EBR < 8 时**禁止开跑**——输入/权重只占满量程 6%，EBR 低时信号全淹没在底噪里，结果≈随机（历史实测 12%）。

## 3. 校准（每次开窗必做，约 10 分钟）

```bash
echo 5182 | sudo -S python3 -m compass_sdk.fast_calibration.compass_cali   # 以官方入口为准
```

- 校准后复测 EBR，应 ≥8；
- 若报 scipy 错误：`ln -sf /usr/local/lib/python3.6/dist-packages/decorator.py /usr/lib/python3/dist-packages/scipy/_lib/decorator.py`（悬空符号链接修复，见 DEPLOY_LOG §1.3）。

## 4. 运行推理

**脚本禁位置参数（compass_sdk 篡改 sys.argv）——一律环境变量传参，经 `sudo env`：**

```bash
# DSQ · scale 模式（默认，推荐）：10000 张全量约 4 分钟
echo 5182 | sudo -S env MNIST_METHOD=dsq MNIST_MODE=scale MNIST_LIMIT=10000 MNIST_BATCH=50 python3 run_mnist_gazelle.py

# LSQ+ / STE 换方法只需改 MNIST_METHOD：
echo 5182 | sudo -S env MNIST_METHOD=lsqplus MNIST_MODE=scale MNIST_LIMIT=10000 MNIST_BATCH=50 python3 run_mnist_gazelle.py
echo 5182 | sudo -S env MNIST_METHOD=ste MNIST_MODE=scale MNIST_LIMIT=10000 MNIST_BATCH=50 python3 run_mnist_gazelle.py

# 快速冒烟（200 张，几十秒）：
echo 5182 | sudo -S env MNIST_METHOD=dsq MNIST_MODE=scale MNIST_LIMIT=200 python3 run_mnist_gazelle.py

# 离线 FAKE 自检（不占板，验证脚本自身无误）：
MNIST_FAKE=1 MNIST_LIMIT=200 python3 run_mnist_gazelle.py
```

### 环境变量一览

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| MNIST_METHOD | dsq / ste / lsqplus | dsq | 量化方法 |
| MNIST_MODE | raw / scale / advance | scale | 喂硬件方式；scale=×16 放大后除回，推荐 |
| MNIST_LIMIT | 整数 | 1000 | 测试样本数 |
| MNIST_BATCH | 整数 | 50 | batch |
| MNIST_FAKE | 1 | 0 | 离线 numpy 验证，不调光计算 |

## 5. 预期结果

| 方法 | 真机精度（10000 张） | 软件参考 | gap |
|---|---|---|---|
| LSQ+ | **97.35%** | ~97.4% | ≤0.1pt |
| STE | 96.43% | ~96.4% | ≤0.1pt |
| DSQ | 94.79% | ~94.8% | ≤0.1pt |

速度：1000 样本 × 3 层光计算约 24 秒；10000 张全量约 4 分钟。

## 6. 官方测试数据接入

脚本从**脚本同目录**加载 `test_images.npy`（uint8, N×784, 已展平）与 `test_labels.npy`（int64, N,）（`run_mnist_gazelle.py` L189-190）。官方提供新测试数据时：

1. **同为 npy 格式**：直接覆盖 `/home/uisrc/mnist/test_images.npy` 与 `test_labels.npy`（先备份原文件）；
2. **标准 MNIST idx 格式**（t10k-images-idx3-ubyte 等）：先转 npy——
   ```python
   import numpy as np, struct
   def read_idx(p):
       with open(p,'rb') as f:
           magic = struct.unpack('>I', f.read(4))[0]; nd = magic & 0xFF
           dims = struct.unpack('>'+'I'*nd, f.read(4*nd))
           return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
   imgs = read_idx('t10k-images-idx3-ubyte').reshape(-1, 784)
   labs = read_idx('t10k-labels-idx1-ubyte').astype(np.int64)
   np.save('test_images.npy', imgs); np.save('test_labels.npy', labs)
   ```
3. **图片文件**（png 等）：读入 → 灰度 28×28 → `np.array.flatten().astype(np.uint8)` → 存 npy。

> 注意：权重为 MNIST 手写数字训练所得；官方数据若非手写数字域，精度语义需现场确认。

## 7. 故障速查（详见 DEPLOY_LOG.md §1.3）

| 现象 | 根因 | 处置 |
|---|---|---|
| 脚本报 unrecognized arguments | compass_sdk 导入即篡改 sys.argv | 禁位置参数，全部用环境变量 |
| 精度 ≈12%（随机水平） | EBR 低 / 值域太小信号被底噪淹没 | 先 compass_cali；确认 MNIST_MODE=scale |
| compass_cali 报 scipy 错 | 板上 scipy decorator.py 悬空链接 | 见 §3 的 ln -sf 修复 |
| 板子被其他队占用 / 输出异常 | 共享板物理瞬态 | 等 ~40min 或 fresh 校准后判据重查再开窗 |

## 8. 演示后收尾

- 使用登记：在 `/home/uisrc/BOARD_USAGE.md` 追加本次使用记录（勿删他人条目）；
- 关闭会话，板子留冷却。