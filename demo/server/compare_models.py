"""Model 3 / M9 / M10 parallel inference on one user-selected backend.

The browser fires all three requests together.  Each request explicitly carries
``osimulator``, ``gazelle_ssh`` or ``gazelle_serial`` so the cards cannot silently
use different targets.  An unavailable target degrades only that model to the
local quantized NumPy reference and is always marked as degraded.
"""
import time

from demo.server import gazelle_client
from demo.server.gazelle_client import RemoteUnavailable as GazelleUnavailable

ALLOWED_BACKENDS = {"osimulator", "gazelle_ssh", "gazelle_serial"}
BACKEND_TARGETS = {
    "osimulator": "osimulator",
    "gazelle_ssh": "Gazelle SSH",
    "gazelle_serial": "Gazelle 串口",
}

# M9/M10 use the authoritative full 5400-image hardware figures. Model 3 has
# no matching full Gazelle result in the handoff, so that cell remains empty.
COMPARE_METRICS = {
    3: {
        "name": "Model 3",
        "arch": "SpaceNet V2 + KD",
        "desc": "stem + 3 stages + classifier, knowledge distillation",
        "params": 267_944,
        "macs_per_image": 1_051_136,
        "optic_ratio": 0.9065,
        "reference_acc": 0.9183,
        "osim_acc": 0.9028,
        "hardware_acc": None,
        "hardware_n": None,
        "hardware_gap_pt": None,
        "source": "osimulator",
    },
    9: {
        "name": "M9",
        "arch": "w075ds3",
        "desc": "0.75x width + learned 3x3/s2 optical downsampling",
        "params": 54_900,
        "macs_per_image": 1_520_000,
        "optic_ratio": 0.90,
        "reference_acc": 0.9587,
        "osim_acc": None,
        "hardware_acc": 0.9443,
        "hardware_n": 5_400,
        "hardware_gap_pt": -1.44,
        "source": "Gazelle",
    },
    10: {
        "name": "M10",
        "arch": "ds3pool3",
        "desc": "wider channels + stem MaxPool3 + 3x3/s2 optical downsampling",
        "params": 96_600,
        "macs_per_image": 2_560_000,
        "optic_ratio": 0.90,
        "reference_acc": 0.9676,
        "osim_acc": None,
        "hardware_acc": 0.9533,
        "hardware_n": 5_400,
        "hardware_gap_pt": -1.43,
        "source": "Gazelle",
    },
}


def _finish(result, *, degraded, target, started, reason=None):
    result["degraded"] = degraded
    result["target"] = target
    result["remote_latency_s"] = round(time.perf_counter() - started, 6)
    if reason:
        result["degraded_reason"] = reason
    return result


def infer_model(model_id, image_b64, backend_mode="osimulator"):
    """Run Model 3, M9 or M10 on the same explicitly selected backend."""
    if model_id not in (3, 9, 10):
        raise ValueError("model_id must be 3, 9, or 10, got %s" % model_id)
    if backend_mode not in ALLOWED_BACKENDS:
        raise ValueError("backend must be osimulator, gazelle_ssh, or gazelle_serial")

    model_name = "model%d" % model_id
    target = BACKEND_TARGETS[backend_mode]
    t0 = time.perf_counter()
    try:
        result = gazelle_client.infer(
            image_b64, model_name=model_name, backend_mode=backend_mode)
        result["backend"] = backend_mode
        return _finish(
            result, degraded=False, target=target, started=t0)
    except GazelleUnavailable as exc:
        # Keep all three cards displayable, but never label a local reference
        # as osimulator or Gazelle hardware.
        result = gazelle_client.infer(
            image_b64, model_name=model_name, clean=True)
        result["engine"] = "numpy-clean"
        result["backend"] = backend_mode
        return _finish(
            result, degraded=True, target=target, started=t0,
            reason="%s 不可用: %s" % (target, str(exc)[:160]))
