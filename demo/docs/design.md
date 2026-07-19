# Model 3 光计算演示 — 设计文档

> 2026-07-17 · 状态: 后端与光计算 server 已完成并真机验证; 前端美学版已实现 (frontend-design.md)。
> 目标: 给评委现场演示 Model 3 (SpaceNet V2 + KD, int8) 的**真机光计算推理**,
> 并与本地 FP32 电计算**逐层对比**, 体现「系统能跑、有用、为光计算做了优化」。

## 演示叙事

1. 从干净 test 集 (seed=42, 5400 张) 抽图, 或评委上传图片。
2. 同一份权重 (`weights/spacenet_v2_phase4_v3_int8.pth`) 走两条路径:
   - **FP32 电计算** (本地, 毫秒级) — baseline;
   - **int8 光计算** (远程 `gazelle_sim` 容器, osimulator 真机仿真, ~2.5–4s/张) — 主角。
3. 逐层对比 6 个计算层的 feature map (光 | 电 并排), 展示量化误差很小;
   预测结果 + top-k 概率对比; 指标面板 (光计算占比 90.65%, MOPs 1.05M,
   osim 全量 90.28%, 硬件对齐率 99.6%)。

> 同一份权重做对比是关键设计: 两条路径的差异**完全来自**光计算的量化与噪声,
> 没有训练差异混入 — 这正是「为光计算优化」的证据链。

## 架构

```
浏览器单页前端 (demo/web/, 占位版 → 美学版另案)
   │ HTTP
本地后端 demo/server/ (FastAPI + uvicorn, 演示笔记本)
   ├─ FP32 路径: 本地 PyTorch + forward_traced() 逐层激活/耗时
   ├─ 光计算路径: remote_client → SSH 隧道 (ssh -L 8765)
   │     → 容器内常驻 optic_server (demo/remote/, 纯 stdlib http.server)
   │     启动一次: load_gazelle_model + build_optical_model(int8, keep_first_conv_electronic)
   │     每请求: logits + 逐层激活 (float16 b64) + 逐层耗时
   └─ 降级链: 远程超时/失败 → 本地 FakeOpticalEngine (同 int8 伪量化)
             → 仍失败则 503; meta.degraded 标记, 前端提示
```

关键决策:

- **容器内纯 stdlib**: `gazelle_sim` 无 flask/fastapi, optic_server 只用
  `http.server` + `json` + `numpy` + `torch` + `osimulator`。
- **激活原样回传, 渲染在本地**: 每层激活以 float16 + base64(npy 序列化) 返回,
  feature map 网格渲染由本地/前端完成 (容器无需 PIL/matplotlib);
  光/电两条路径同一层**共享归一化区间**, 对比才公平。
- **逐层捕获统一走 `forward_traced()`** (`demo/server/model_trace.py`):
  按 stem → stage1 → stage2 → stage3 → fc1 → fc2 分段执行,
  FP32 模型与光计算模型 (OpticConv2d/OpticLinear 替换后) 用同一函数,
  保证 hook 点严格对齐。fc1 取 classifier 中 Linear(1024→256)+ReLU 输出, fc2 取最终 logits。
- **模型只加载一次**: 远程常驻进程启动时完成 gazelle 引擎初始化与权重加载,
  每请求纯推理; 本地 FP32/fake 同理。
- **不动现有实验代码**: 全部新增在 `demo/`; 仅 import 复用
  `src/core/optic_layers.py`、`src/data/eurosat_split.py`。
- **OPTIC_FAKE=1**: optic_server 的环境变量开关, 强制 Fake 引擎 —
  使远程服务可以在本地 (无 osim) 完整跑通集成测试。

## 目录

```
demo/
├── docs/            design.md(本文), api.md(接口约定)
├── remote/          optic_server.py        # 容器内常驻光计算推理服务
├── server/          app.py, inference_local.py, remote_client.py,
│                    model_trace.py, metrics.py
├── web/             index.html(占位, 美学版另案)
├── tests/           pytest, 本地全量可跑 (OPTIC_FAKE=1 覆盖远程服务)
└── deploy.sh        同步到容器 + 启动常驻服务 + 建立 SSH 隧道
```

## 部署链路 (deploy.sh)

1. `tar | ssh fdusc-cpu-135 "docker exec -i gazelle_sim tar -x -C /workspace/demo"`
   同步 `optic_server.py` + `optic_layers.py` + 权重 (容器内 repo 是旧扁平版, 独立目录最干净);
2. 容器内 `nohup python optic_server.py --port 8765 &` (用 `/local/miniconda/envs/moca_llm/bin/python`);
3. 本地 `ssh -N -L 8765:localhost:8765 fdusc-cpu-135` 建立隧道;
4. `uvicorn demo.server.app:app --port 8000` 起本地后端。

## 风险与降级

| 风险 | 应对 |
|---|---|
| 现场网络不通 fdusc-cpu-135 | remote_client 30s 超时 + health 探测 → Fake 引擎降级 (meta.degraded), 演示不中断 |
| 远程服务崩溃 | deploy.sh 支持重启; 降级链兜底 |
| 评委上传任意图片 OOD | 上传图只显示预测, 不显示对错; 服务端 resize/center-crop 到 64×64 |
| `--qat` int4 路径混淆 | demo 完全不走 `--qat`; 只用 v3 int8 权重 |

## 测试策略

- `forward_traced`: 6 层名称/形状/耗时正确, FP32 与光计算模型 hook 点对齐;
- `optic_server`: OPTIC_FAKE=1 本地起服务, /health + /infer 全请求 roundtrip;
- `inference_local`: FP32 与 fake 光计算两条路径产出契约一致的 PathResult;
- `app`: FastAPI TestClient 全接口 (remote mock 与真实 fake 各一遍);
- 烟测 (手动): deploy.sh 真机单图推理, 核对 pred/耗时/激活形状。
