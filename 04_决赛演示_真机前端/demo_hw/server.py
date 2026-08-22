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


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")