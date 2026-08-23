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
import copy
import io
import os
import random
import sys
import threading
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
from demo.server import board_checks, board_runner  # noqa: E402  (上板放行判据)
from demo.server import gazelle_client  # noqa: E402  (M9/M10: Gazelle 真机)
from demo.server import remote_client as osim_client  # noqa: E402  (Model 3: osimulator)
from demo.server.gazelle_client import (  # noqa: E402
    RemoteUnavailable as GazelleUnavailable,
)
from demo.server.remote_client import (  # noqa: E402
    RemoteUnavailable as OsimUnavailable,
)

# 按模型路由，而不是用一个全局后端替换全部模型：
#   OPTIC_OSIM=1: Model 3 -> osimulator；M9/M10 -> Gazelle。
#   OPTIC_OSIM=0: 三者均可走 Gazelle（保留原单模型真机模式）。
_OSIM = os.environ.get("OPTIC_OSIM", "0") == "1"
from demo.server.compare_models import COMPARE_METRICS  # noqa: E402
from demo.server.inference_local import CLASSES, DATA_DIR  # noqa: E402
from demo.server.metrics import METRICS  # noqa: E402

WEB_DIR = os.path.join(REPO_ROOT, "demo", "web")

app = FastAPI(title="Model 3 Optics Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Prevent stale demo HTML/JS from surviving frontend revisions."""
    response = await call_next(request)
    if request.url.path.endswith((".html", ".js", ".css")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

_dataset = None
_test_indices = None
_board_gate_lock = threading.Lock()


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
    backend: str = "osimulator"


@app.get("/api/health")
def health(backend: str = Query("osimulator")):
    try:
        inference_local.get_fp32_model()
        local = "ok"
    except Exception:
        local = "error"

    allowed = {"electronic", "osimulator", "gazelle_ssh", "gazelle_serial"}
    if backend not in allowed:
        raise HTTPException(400, "unknown backend %r" % backend)
    if backend == "electronic":
        remote = "fp32-local" if local == "ok" else "down"
        detail = "本地浮点电计算"
    else:
        try:
            probe = gazelle_client.health(backend_mode=backend)
            remote = probe["engine"]
            detail = probe.get("detail")
        except GazelleUnavailable as exc:
            remote = "down"
            detail = str(exc)[:180]
    return {
        "local": local, "remote": remote, "backend": backend,
        "detail": detail,
        "backends": ["electronic", "osimulator",
                     "gazelle_ssh", "gazelle_serial"],
        "gazelle_host": gazelle_client.GAZELLE_HOST,
        "gazelle_port": gazelle_client.GAZELLE_PORT,
    }


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
    model = req.model if req.model in (
        "model3", "model9", "model10") else "model3"
    allowed = {"electronic", "osimulator", "gazelle_ssh", "gazelle_serial"}
    if req.backend not in allowed:
        raise HTTPException(400, "unknown backend %r" % req.backend)
    if (req.backend in ("gazelle_ssh", "gazelle_serial") and
            _board_gate_lock.locked()):
        raise HTTPException(409, "path-B release check is using the board")
    try:
        base64.b64decode(req.image_b64, validate=True)
    except Exception as exc:
        raise HTTPException(400, "image decode failed: %s" % exc)

    try:
        fp32 = gazelle_client.infer_fp32(
            req.image_b64, model_name=model)
    except GazelleUnavailable as exc:
        raise HTTPException(503, "local FP32 reference failed: %s" % exc)

    targets = {
        "electronic": "本地 FP32 电计算",
        "osimulator": "本地 osimulator",
        "gazelle_ssh": "Gazelle · SSH 隧道",
        "gazelle_serial": "Gazelle · 串口引导",
    }
    t0 = time.perf_counter()
    degraded = False
    degraded_reason = None
    if req.backend == "electronic":
        optical = copy.deepcopy(fp32)
    else:
        try:
            optical = gazelle_client.infer(
                req.image_b64, model_name=model,
                backend_mode=req.backend)
        except GazelleUnavailable as exc:
            optical = gazelle_client.infer(
                req.image_b64, model_name=model, clean=True)
            optical["engine"] = "numpy-quantized-fallback"
            degraded = True
            degraded_reason = str(exc)[:220]
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
            "degraded_reason": degraded_reason,
            "remote_latency_s": round(remote_latency, 6),
            "label": req.label,
            "correct": correct,
            "model": model,
            "backend": req.backend,
            "target": targets[req.backend],
            "reference_engine": fp32["engine"],
            "selected_engine": optical["engine"],
        },
    }


# ---------------- 上板四项放行判据 (final/main canonical path B) ---------


@app.get("/api/checks/usage")
def checks_usage():
    """只读 who/ps/BOARD_USAGE 证据；进程归属仍须人工确认。"""
    try:
        evidence = board_runner.inspect_usage()
        return {"ok": True, "host": board_runner.board_host(),
                "evidence": evidence}
    except Exception as exc:
        return {"ok": False, "host": board_runner.board_host(),
                "evidence": str(exc)[:500]}


@app.get("/api/checks/ebr")
def checks_ebr(confirmed_idle: bool = Query(False)):
    """判据①/②: 自动执行一次 EVB 并解析两通道 EBR/error_std。"""
    if not confirmed_idle:
        reason = "先查看 who/ps/BOARD_USAGE 并确认无人占用、已冷却≥5min"
        return {"all_pass": False, "checks": [
            {"name": "① EBR ≥ 8", "pass": False, "blocked": True,
             "value": "未执行", "detail": reason},
            {"name": "② error_std 对健康基线", "pass": False,
             "blocked": True, "value": "未执行", "detail": reason},
        ]}
    if not _board_gate_lock.acquire(blocking=False):
        raise HTTPException(409, "another path-B release check is running")
    try:
        out = list(board_checks.check_evb())
    finally:
        _board_gate_lock.release()
    return {"all_pass": all(c["pass"] for c in out), "checks": out}


@app.get("/api/checks/probe")
def checks_probe():
    """短 SSH 连通性诊断（不是四项放行判据之一）。"""
    ok, detail = board_runner.probe_board()
    return {"name": "Gazelle SSH path-B probe", "pass": ok,
            "value": board_runner.board_host(), "detail": detail}


@app.get("/api/checks/canary")
def checks_canary(confirmed_idle: bool = Query(False)):
    """判据③: MNIST DSQ canary gap < 0.5pt。"""
    if not confirmed_idle:
        return {"name": "③ MNIST canary gap < 0.5pt", "pass": False,
                "blocked": True, "value": "未执行",
                "detail": "未确认无人占用与冷却状态"}
    if not _board_gate_lock.acquire(blocking=False):
        raise HTTPException(409, "another path-B release check is running")
    try:
        return board_checks.check_canary()
    finally:
        _board_gate_lock.release()


@app.get("/api/checks/minirun")
def checks_minirun(model: str = Query("model10", alias="model"),
                   n: int = Query(100, ge=1, le=500),
                   confirmed_idle: bool = Query(False)):
    """判据④: path-B 真机准确率 vs 本地参考，偏差 <2pt。"""
    if not confirmed_idle:
        return {"name": "④ EuroSAT mini-run 偏差 < 2pt", "pass": False,
                "blocked": True, "value": "未执行",
                "detail": "未确认无人占用与冷却状态"}
    imgs = labels = None
    for cand in ("/workspace/out/test200_images.npy",
                 os.path.join(REPO_ROOT, "tools", "out", "test200_images.npy")):
        if os.path.isfile(cand):
            imgs = np.load(cand)
            labels = np.load(cand.replace("_images", "_labels"))
            break
    if not _board_gate_lock.acquire(blocking=False):
        raise HTTPException(409, "another path-B release check is running")
    try:
        prereqs, calib = board_checks.preflight(model, confirmed_idle=True)
        if not all(item["pass"] for item in prereqs):
            failed = next(item for item in prereqs if not item["pass"])
            return {"name": "④ EuroSAT mini-run 偏差 < 2pt", "pass": False,
                    "blocked": True, "value": "未执行",
                    "detail": failed["detail"]}
        return board_checks.check_minirun(
            model_name=model, n=n, images=imgs, labels=labels,
            calib_json=calib)
    finally:
        _board_gate_lock.release()


@app.get("/api/checks/all")
def checks_all(model: str = Query("model10", alias="model"),
               n: int = Query(100, ge=1, le=500),
               confirmed_idle: bool = Query(False)):
    """先检查前置条件，再按 ①→④ 顺序执行；缺项永不放行。"""
    if model not in ("model9", "model10"):
        raise HTTPException(400, "path-B release checks support model9/model10")
    imgs = labels = None
    for cand in ("/workspace/out/test200_images.npy",
                 os.path.join(REPO_ROOT, "tools", "out", "test200_images.npy")):
        if os.path.isfile(cand):
            imgs = np.load(cand)
            labels = np.load(cand.replace("_images", "_labels"))
            break
    if not _board_gate_lock.acquire(blocking=False):
        raise HTTPException(409, "another path-B release check is running")
    try:
        return board_checks.all_checks(
            model_name=model, images=imgs, labels=labels, n=n,
            confirmed_idle=confirmed_idle)
    finally:
        _board_gate_lock.release()


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
    backend: str = "osimulator"


@app.post("/api/compare-infer")
def compare_infer(req: CompareInferRequest):
    if req.model_id not in (3, 9, 10):
        raise HTTPException(400, f"model_id must be 3, 9, or 10, got {req.model_id}")
    if req.backend not in compare_models.ALLOWED_BACKENDS:
        raise HTTPException(400, "backend must be osimulator, gazelle_ssh, or gazelle_serial")
    if (req.backend in ("gazelle_ssh", "gazelle_serial") and
            _board_gate_lock.locked()):
        raise HTTPException(409, "path-B release check is using the board")
    try:
        result = compare_models.infer_model(
            req.model_id, req.image_b64, backend_mode=req.backend)
    except Exception as e:
        raise HTTPException(503, f"inference failed for model {req.model_id}: {e}")
    return result


@app.get("/api/compare-metrics")
def compare_metrics_endpoint():
    return COMPARE_METRICS


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
