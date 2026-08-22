# 复赛可视化演示 demo（osimulator 仿真链路）

> ⚠️ **口径说明（决赛阶段）**：本 demo 的光计算路径连接的是**复赛的 osimulator 仿真容器**（HTTP :8765 SSH 隧道 → gazelle_sim 容器内 optic_server）。
> 决赛现场如需**真机**前端演示，请使用 `../04_决赛演示_真机前端/`（连接 Gazelle 真机 :8000）。
> 远程不可用时本 demo 自动降级 FakeOpticalEngine（页面亮黄灯）——**降级状态不得称为真机/仿真实时推理**。

## 启动（复赛口径）

```bash
# 1) 容器内起 optic_server（demo/remote/）
# 2) SSH 隧道: ssh -L 8765:容器:8765 ...
# 3) 本地后端:
uvicorn demo.server.app:app --port 8000   # 浏览器 http://127.0.0.1:8000
```

详见 `docs/design.md` / `docs/api.md`。