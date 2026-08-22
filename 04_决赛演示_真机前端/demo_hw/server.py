# -*- coding: utf-8 -*-
"""决赛真机演示后端 — 浏览器前端 × Gazelle 真机光计算（demo-hw）。

复用 03_决赛_EuroSAT真机/opticspacenet 的完整路径 A 推理链
(gazelle_engine.build_model + HttpBackend -> 板上 server_gazelle.py)。
与 02_复赛 demo 的区别: optical 路径连接的是【真机】而非 osim 容器。

运行(从 04_决赛演示_真机前端 目录):
  uvicorn demo_hw.server:app --port 8100
前置: ① 板上已起 server_gazelle.py(:8000) ② SSH 隧道 127.0.0.1:8000 -> 板:8000
      ③ 已完成 compass_cali + 四项放行判据(见 03_决赛/mnist/MNIST_现场演示Runbook.md §2)
"""
import base64
import io
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_OPTIC = os.path.join(_PKG, "03_决赛_EuroSAT真机", "opticspacenet")
_SRC = os.path.join(_PKG, "02_复赛_EuroSAT仿真", "src")
for _p in (_OPTIC, _SRC, os.path.join(_SRC, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from gazelle_engine import NumpyBackend, HttpBackend, build_model  # noqa: E402

# ---- 配置(环境变量, 禁位置参数) ----
MODEL_NAME = os.environ.get("HW_MODEL", "model2")   # model2|model3|model1a|model1b
WEIGHT = os.environ.get("HW_WEIGHT", "")             # 空=registry 默认权重
BACKEND = os.environ.get("HW_BACKEND", "http")       # http=真机 numpy=干净参考
OPTC_HOST = os.environ.get("OPTC_HOST", "127.0.0.1")
OPTC_PORT = int(os.environ.get("OPTC_PORT", "8000"))
CORRECTION = os.environ.get("CORRECTION", "")        # 可选 calib npz
WEB_DIR = os.path.join(_HERE, "web")

# ---- 模型(懒加载, 进程级单例) ----
_model = None
_engine = None


def get_model():
    global _model, _engine
    if _model is None:
        backend = (HttpBackend(host=OPTC_HOST, port=OPTC_PORT)
                   if BACKEND == "http" else NumpyBackend())
        corr = CORRECTION or None
        _model, _engine = build_model(WEIGHT or None, backend,
                                     correction=corr,
                                     model_name=MODEL_NAME)
    return _model, _engine


CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
           "Industrial", "Pasture", "PermanentCrop", "Residential",
           "River", "SeaLake"]

MEAN = np.array([0.3434, 0.3807, 0.3253], dtype=np.float32)
STD = np.array([0.1874, 0.1865, 0.1781], dtype=np.float32)


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None])


class InferRequest(BaseModel):
    image_b64: str
    label: str | None = None


app = FastAPI(title="决赛真机光计算演示")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    try:
        model, engine = get_model()
        local = "ok"
    except Exception as e:
        return {"local": "error", "remote": "down", "detail": str(e)[:200]}
    try:
        backend = engine.backend
        remote = ("gazelle-hw:" + backend.name) if backend.name == "http" else "numpy-ref"
    except Exception:
        remote = "unknown"
    return {"local": local, "remote": remote,
            "model": MODEL_NAME, "backend": BACKEND}


@app.post("/api/infer")
def infer(req: InferRequest):
    try:
        image_bytes = base64.b64decode(req.image_b64)
        x = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(400, "image decode failed: %s" % e)
    model, engine = get_model()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits[0], dim=-1).numpy()
    except Exception as e:
        raise HTTPException(503, "真机推理失败(检查隧道/板上 server/校准): %s" % e)
    latency = time.perf_counter() - t0
    top = int(np.argmax(probs))
    order = np.argsort(-probs)[:5]
    return {
        "pred": CLASSES[top],
        "probs": {CLASSES[i]: round(float(probs[i]), 4) for i in order},
        "topk": [{"cls": CLASSES[i], "p": round(float(probs[i]), 4)} for i in order],
        "latency_s": round(latency, 3),
        "engine": "gazelle-hw" if BACKEND == "http" else "numpy-ref",
        "correct": (req.label == CLASSES[top]) if req.label else None,
    }


# ============================================================
# 上板放行判据检查（PPT P17 四项判据的可视化演示）
#   判据1 EBR>=8        — 需板端 compass_evb_test，评委现场看板端读数后手动输入验证
#   判据2 error_std<±2% — 自动：已知探针矩阵乘 vs numpy 精确值
#   判据3 canary<0.5pt  — 自动：10 图 mini-run 的 hw acc vs numpy 干净参考 acc 之差
#                          （真机判据为 MNIST canary；此处用 EuroSAT 10 图等价演示，页面如实标注）
#   判据4 mini-run 对齐 — 自动：同一次 10 图 run 的逐图预测与参考一致性 + 精度正常
# 时间控制：探针(秒级) + 10图×5光算层往返，总计 <2 分钟
# ============================================================

CHECK_N = int(os.environ.get("HW_CHECK_N", "10"))


@app.get("/api/checks/probe")
def checks_probe():
    """判据2：探针矩阵乘误差（与 numpy 精确参考对比）。"""
    rng = np.random.RandomState(42)
    x = rng.randint(1, 200, size=(16, 8)).astype(np.uint8)   # (m,k)
    w = rng.randint(-100, 100, size=(8, 2)).astype(np.int8)  # (k,n)
    exact = x.astype(np.float64) @ w.astype(np.float64)      # (16,2)
    model, engine = get_model()
    backend = engine.backend
    try:
        got = backend.matmul_2d(x.astype(np.int64), w.astype(np.int64))
    except Exception as e:
        return {"name": "error_std 偏差", "pass": False,
                "detail": "真机 matmul 失败: %s" % str(e)[:150]}
    err = np.abs(got - exact).ravel()
    rms = float(np.sqrt(np.mean(err ** 2)))
    ref_scale = float(np.sqrt(np.mean(np.abs(exact) ** 2))) + 1e-9
    rel = rms / ref_scale * 100.0
    ok = rel < 2.0
    return {"name": "② error_std 偏差 < ±2%", "pass": bool(ok),
            "value": "%.2f%%" % rel,
            "detail": "16 组已知探针 vs numpy 精确参考，相对 RMS 误差"}


@app.post("/api/checks/minirun")
def checks_minirun():
    """判据3+4：10 图 mini-run（hw vs numpy 干净参考）。"""
    # 数据：优先包内 02_复赛 data（.gitignore 本地就位）
    data_dir = os.environ.get("HW_DATA",
                              os.path.join(_PKG, "02_复赛_EuroSAT仿真", "data", "EuroSAT_RGB"))
    if not os.path.isdir(data_dir):
        raise HTTPException(503, "EuroSAT 数据未找到: %s（见 README §0）" % data_dir)
    sys.path.insert(0, os.path.join(_SRC, "data"))
    from eurosat_split import split_indices  # noqa
    import torchvision.datasets as datasets  # noqa
    ds = datasets.ImageFolder(data_dir)
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    rng = np.random.RandomState(7)
    picks = sorted(rng.choice(test_idx, size=min(CHECK_N, len(test_idx)), replace=False).tolist())

    # 双引擎：numpy 干净参考 + 真机
    m_np, _ = build_model(None, NumpyBackend(), model_name=MODEL_NAME)
    m_hw, _ = build_model(WEIGHT or None, HttpBackend(host=OPTC_HOST, port=OPTC_PORT),
                          model_name=MODEL_NAME)
    agree = 0
    acc_np = acc_hw = 0
    for i in picks:
        path, target = ds.samples[i]
        img = Image.open(path).convert("RGB").resize((64, 64))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        t = torch.from_numpy(arr.transpose(2, 0, 1)[None])
        with torch.no_grad():
            p_np = int(m_np(t).argmax())
            p_hw = int(m_hw(t).argmax())
        acc_np += int(p_np == target)
        acc_hw += int(p_hw == target)
        agree += int(p_np == p_hw)
    n = len(picks)
    gap = abs(acc_np - acc_hw) / n * 100.0
    return {
        "n": n,
        "acc_ref": round(acc_np / n * 100, 1),
        "acc_hw": round(acc_hw / n * 100, 1),
        "canary": {"name": "③ Canary gap < 0.5pt（EuroSAT-10 等价演示）",
                   "pass": bool(gap <= 0.5),
                   "value": "%.1fpt" % gap,
                   "detail": "hw 与干净参考 acc 差（真机判据为 MNIST canary）"},
        "minirun": {"name": "④ Mini-run 采样对齐",
                    "pass": bool(agree / n >= 0.8 and acc_hw / n >= 0.5),
                    "value": "%d/%d 图与参考一致，hw acc %.1f%%" % (agree, n, acc_hw / n * 100),
                    "detail": "逐图预测一致率与精度正常性"},
    }


@app.post("/api/checks/ebr")
def checks_ebr(req: dict):
    """判据1：EBR 手动录入（板端 compass_evb_test 读数）。"""
    try:
        v = float(req.get("ebr"))
    except (TypeError, ValueError):
        raise HTTPException(400, "ebr 字段缺失/非数值")
    return {"name": "① EBR ≥ 8", "pass": bool(v >= 8), "value": str(v),
            "detail": "板端 compass_evb_test 实测值（现场读数录入）"}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")