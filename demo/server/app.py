"""FastAPI backend for the Model 3 optics demo (demo/docs/design.md).

Run from the repo root:  uvicorn demo.server.app:app --port 8000

Endpoints (contract: demo/docs/api.md):
  GET  /api/health   local model status + remote engine probe
  GET  /api/sample   random image from the clean test split (seed=42)
  POST /api/infer    fp32-local + remote optical (fake-optical fallback)
  GET  /api/metrics  static exhibition-board numbers
  GET  /             demo/web static page

CORS is wide open on purpose — this is a local demo server.
"""
import base64
import io
import os
import random
import sys
import time

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))          # demo/server
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))           # repo root
for _p in (_HERE, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _pathsetup  # noqa: E402,F401  (adds src/core, src/data, ...)

from eurosat_split import split_indices  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from torchvision import datasets  # noqa: E402

from demo.server import compare, compare_models, inference_local, render  # noqa: E402
from demo.server import board_checks  # noqa: E402  (上板四项放行判据)
# 数据来源: OPTIC_OSIM=1 → 容器 osimulator (optic_server :8765, 原 remote_client);
#           否则 → Gazelle 真机 HTTP (gazelle_client) / GAZELLE_FAKE=1 离线 numpy。
_OSIM = os.environ.get("OPTIC_OSIM", "0") == "1"
if _OSIM:
    from demo.server import remote_client  # noqa: E402  (osimulator HTTP 客户端)
    from demo.server import gazelle_client as _gazelle_rc  # noqa: E402  (M9/M10 干净参考)
    from demo.server.remote_client import RemoteUnavailable  # noqa: E402
else:
    from demo.server import gazelle_client as remote_client  # noqa: E402  (Gazelle 真机)
    from demo.server.gazelle_client import RemoteUnavailable  # noqa: E402
from demo.server.compare_models import COMPARE_METRICS  # noqa: E402
from demo.server.inference_local import CLASSES, DATA_DIR  # noqa: E402
from demo.server.metrics import METRICS  # noqa: E402

WEB_DIR = os.path.join(REPO_ROOT, "demo", "web")

app = FastAPI(title="Model 3 Optics Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_dataset = None
_test_indices = None


def _get_test_set():
    """Lazy: ImageFolder (no transform) + clean test indices (seed=42)."""
    global _dataset, _test_indices
    if _dataset is None:
        _dataset = datasets.ImageFolder(DATA_DIR)
        _, _, _test_indices = split_indices(len(_dataset), seed=42,
                                            val_ratio=0.2, test_ratio=0.2)
    return _dataset, _test_indices


class InferRequest(BaseModel):
    image_b64: str
    label: str | None = None
    model: str = "model3"   # model3 | model9 | model10 (前端模型选择)


@app.get("/api/health")
def health():
    try:
        inference_local.get_fp32_model()
        local = "ok"
    except Exception:
        local = "error"
    try:
        remote = remote_client.health()["engine"]
    except RemoteUnavailable:
        remote = "down"
    return {"local": local, "remote": remote,
            "gazelle_host": getattr(remote_client, "GAZELLE_HOST", "")}


@app.get("/api/sample")
def sample(class_: str | None = Query(None, alias="class"),
           random_: bool = Query(False, alias="random")):
    ds, test_idx = _get_test_set()
    pool = test_idx
    if class_ is not None and class_ != "random":
        if class_ not in ds.classes:
            raise HTTPException(404, f"unknown class {class_!r}")
        target = ds.class_to_idx[class_]
        pool = [i for i in test_idx if ds.targets[i] == target]
    idx = random.choice(pool)
    path, target = ds.samples[idx]
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
    return {
        "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "label": ds.classes[target],
        "index": int(idx),
        "classes": ds.classes,
    }


@app.post("/api/infer")
def infer(req: InferRequest):
    model = req.model if req.model in ("model3", "model9", "model10") else "model3"
    try:
        image_bytes = base64.b64decode(req.image_b64)
        img_tensor = inference_local.preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"image decode failed: {e}")

    if _OSIM and model != "model3":
        # osimulator 后端仅挂载 Model 1/2/3; M9/M10 用 numpy 干净参考
        # (前端结构展示验证: s1a..h2 九层, 数据为同模型干净参考, 不连真机)
        fp32 = _gazelle_rc.infer(req.image_b64, model_name=model, clean=True)
        optical = _gazelle_rc.infer(req.image_b64, model_name=model, clean=True)
        degraded = False
        remote_latency = 0.0
    else:
        try:
            if model == "model3":
                fp32 = inference_local.infer_fp32(img_tensor)
            else:
                # M9/M10 干净参考: 同模型 numpy 路径 (NumpyBackend), 与光学路径同层链
                fp32 = remote_client.infer(req.image_b64, model_name=model, clean=True)
        except Exception as e:
            raise HTTPException(503, f"local fp32 reference failed: {e}")

    def _rc_infer(**kw):
        """兼容 osimulator(remote_client: model_id) 与 Gazelle(gazelle_client: model_name)。"""
        if os.environ.get("OPTIC_OSIM", "0") == "1":
            return remote_client.infer(req.image_b64,
                                       model_id={"model3": 3}.get(model, 3))
        return remote_client.infer(req.image_b64, **kw)

    if not (_OSIM and model != "model3"):
        t0 = time.perf_counter()
        try:
            optical = _rc_infer(model_name=model)
            degraded = False
        except RemoteUnavailable:
            if model == "model3":
                optical = inference_local.infer_fake(img_tensor)
            else:
                optical = _rc_infer(model_name=model, clean=True)
            degraded = True
        remote_latency = time.perf_counter() - t0

    correct = None
    if req.label is not None:
        correct = optical["pred"] == req.label

    render.inject_grids(fp32, optical)
    compare.inject_comparison(fp32, optical)

    return {
        "fp32": fp32,
        "optical": optical,
        "meta": {
            "degraded": degraded,
            "remote_latency_s": round(remote_latency, 6),
            "label": req.label,
            "correct": correct,
            "model": model,
        },
    }


# ---------------- 上板四项放行判据 (SOP: global/AGENTS.md) ----------------

class EbrRequest(BaseModel):
    ebr: float
    error_std: float | None = None


@app.post("/api/checks/ebr")
def checks_ebr(req: EbrRequest):
    """判据①/②: 板上 compass_evb_test 读数手动录入 (board_connect.sh [4] 可查)。"""
    out = [board_checks.check_ebr(req.ebr)]
    if req.error_std is not None:
        out.append(board_checks.check_evb_std(req.error_std))
    return {"all_pass": all(c["pass"] for c in out), "checks": out}


@app.get("/api/checks/probe")
def checks_probe():
    """判据②自动: 已知探针真机 vs numpy 参考。"""
    return board_checks.check_probe()


@app.get("/api/checks/canary")
def checks_canary():
    """判据③: MNIST DSQ canary gap < 0.5pt。"""
    return board_checks.check_canary()


@app.get("/api/checks/minirun")
def checks_minirun(model: str = Query("model10", alias="model"),
                   n: int = Query(200)):
    """判据④: EuroSAT mini-run (真机 vs numpy 干净参考)。"""
    imgs = labels = None
    for cand in ("/workspace/out/test200_images.npy",
                 os.path.join(REPO_ROOT, "tools", "out", "test200_images.npy")):
        if os.path.isfile(cand):
            imgs = np.load(cand)
            labels = np.load(cand.replace("_images", "_labels"))
            break
    return board_checks.check_minirun(model_name=model, n=n,
                                      images=imgs, labels=labels)


@app.get("/api/checks/all")
def checks_all(model: str = Query("model10", alias="model"),
               ebr: float | None = None, error_std: float | None = None,
               n: int = Query(200)):
    """四判据一键汇总 (EBR/error_std 可选录入; 未录入则只跑自动三项)。"""
    imgs = labels = None
    for cand in ("/workspace/out/test200_images.npy",
                 os.path.join(REPO_ROOT, "tools", "out", "test200_images.npy")):
        if os.path.isfile(cand):
            imgs = np.load(cand)
            labels = np.load(cand.replace("_images", "_labels"))
            break
    return board_checks.all_checks(ebr=ebr, evb_std=error_std,
                                   model_name=model, images=imgs,
                                   labels=labels, n=n)


@app.get("/api/metrics")
def metrics(model: str = Query("model3", alias="model")):
    if model == "model9":
        from demo.server.metrics import METRICS_M9 as M
        return M
    if model == "model10":
        from demo.server.metrics import METRICS_M10 as M
        return M
    return METRICS


# ============================================================
#  Multi-model comparison endpoints (for /compare.html)
# ============================================================

class CompareInferRequest(BaseModel):
    image_b64: str
    model_id: int


@app.post("/api/compare-infer")
def compare_infer(req: CompareInferRequest):
    if req.model_id not in (1, 2, 3):
        raise HTTPException(400, f"model_id must be 1, 2, or 3, got {req.model_id}")
    try:
        result = compare_models.infer_model(req.model_id, req.image_b64)
    except Exception as e:
        raise HTTPException(503, f"inference failed for model {req.model_id}: {e}")
    return result


@app.get("/api/compare-metrics")
def compare_metrics_endpoint():
    return COMPARE_METRICS


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
