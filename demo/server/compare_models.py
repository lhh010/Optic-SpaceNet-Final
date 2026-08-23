"""Multi-model comparison API for the three-model demo page.

Provides:
  - infer_model(model_id, image_b64) → result dict (tries remote, falls back to fake)
  - COMPARE_METRICS → static per-model metrics for the comparison panel
"""
import base64
import time

from demo.server import inference_local_multi, gazelle_client as remote_client
from demo.server.inference_local import preprocess
from demo.server.gazelle_client import RemoteUnavailable


# ============================================================
#  Static metrics for the comparison panel
# ============================================================

COMPARE_METRICS = {
    1: {
        "name": "Model 1",
        "arch": "Baseline VGG",
        "desc": "6×Conv 3×3 + 2×Linear, Flat VGG",
        "params": 2_393_098,
        "macs_per_image": 156_600_000,
        "optic_ratio": 0.977,
        "osim_acc": 0.9396,
        "qat_acc": 0.9526,
        "per_image_s": 150.0,
        "weight": "baseline_vgg_phase4_v3_int8.pth",
    },
    2: {
        "name": "Model 2",
        "arch": "SpaceNet V1",
        "desc": "stem + 3 stages + classifier, 轻量 CNN",
        "params": 267_944,
        "macs_per_image": 1_051_136,
        "optic_ratio": 0.9065,
        "osim_acc": 0.8980,
        "qat_acc": 0.9185,
        "per_image_s": 2.5,
        "weight": "spacenet_v1_phase4_v3_int8.pth",
    },
    3: {
        "name": "Model 3",
        "arch": "SpaceNet V2 + KD",
        "desc": "stem + 3 stages + classifier, 知识蒸馏",
        "params": 267_944,
        "macs_per_image": 1_051_136,
        "optic_ratio": 0.9065,
        "osim_acc": 0.9028,
        "qat_acc": 0.9183,
        "per_image_s": 2.5,
        "weight": "spacenet_v2_phase4_v3_int8.pth",
    },
}


# ============================================================
#  Inference (remote with fallback)
# ============================================================

def infer_model(model_id, image_b64):
    """Run inference on a single model. Tries remote osimulator first,
    falls back to local fake-optical.

    Returns dict with: engine, model_id, pred, probs, latency_total_s, layers, degraded
    """
    # Try remote optical first
    t0 = time.perf_counter()
    try:
        result = remote_client.infer(image_b64, model_id=model_id)
        result["degraded"] = False
        result["remote_latency_s"] = round(time.perf_counter() - t0, 6)
        return result
    except RemoteUnavailable:
        pass

    # Fallback to local fake-optical
    image_bytes = base64.b64decode(image_b64)
    img_tensor = preprocess(image_bytes)
    result = inference_local_multi.infer_fake(model_id, img_tensor)
    result["degraded"] = True
    result["remote_latency_s"] = round(time.perf_counter() - t0, 6)
    return result
