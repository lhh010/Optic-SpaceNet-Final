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
from demo.server import gazelle_client as remote_client  # noqa: E402  (Gazelle 真机, 替代容器 osimulator)
from demo.server.compare_models import COMPARE_METRICS  # noqa: E402
from demo.server.inference_local import CLASSES, DATA_DIR  # noqa: E402
from demo.server.metrics import METRICS  # noqa: E402
from demo.server.gazelle_client import RemoteUnavailable  # noqa: E402

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

    try:
        if model == "model3":
            fp32 = inference_local.infer_fp32(img_tensor)
        else:
            # M9/M10 干净参考: 同模型 numpy 路径 (NumpyBackend), 与光学路径同层链
            fp32 = remote_client.infer(req.image_b64, model_name=model, clean=True)
    except Exception as e:
        raise HTTPException(503, f"local fp32 reference failed: {e}")

    t0 = time.perf_counter()
    try:
        optical = remote_client.infer(req.image_b64, model_name=model)
        degraded = False
    except RemoteUnavailable:
        if model == "model3":
            optical = inference_local.infer_fake(img_tensor)
        else:
            optical = remote_client.infer(req.image_b64, model_name=model, clean=True)
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
