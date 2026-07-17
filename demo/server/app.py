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

from demo.server import inference_local, remote_client  # noqa: E402
from demo.server.inference_local import CLASSES, DATA_DIR  # noqa: E402
from demo.server.metrics import METRICS  # noqa: E402
from demo.server.remote_client import RemoteUnavailable  # noqa: E402

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
    return {"local": local, "remote": remote}


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
    try:
        image_bytes = base64.b64decode(req.image_b64)
        img_tensor = inference_local.preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"image decode failed: {e}")

    try:
        fp32 = inference_local.infer_fp32(img_tensor)
    except Exception as e:
        raise HTTPException(503, f"local fp32 inference failed: {e}")

    t0 = time.perf_counter()
    try:
        optical = remote_client.infer(req.image_b64)
        degraded = False
    except RemoteUnavailable:
        optical = inference_local.infer_fake(img_tensor)
        degraded = True
    remote_latency = time.perf_counter() - t0

    correct = None
    if req.label is not None:
        correct = optical["pred"] == req.label

    return {
        "fp32": fp32,
        "optical": optical,
        "meta": {
            "degraded": degraded,
            "remote_latency_s": round(remote_latency, 6),
            "label": req.label,
            "correct": correct,
        },
    }


@app.get("/api/metrics")
def metrics():
    return METRICS


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
