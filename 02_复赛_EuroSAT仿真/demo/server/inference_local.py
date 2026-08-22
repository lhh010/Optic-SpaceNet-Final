"""Local inference paths for the optics demo: FP32 baseline + fake-optical.

Both models are lazy singletons loaded once from
``weights/spacenet_v2_phase4_v3_int8.pth`` (shape-filtered, strict=False):

- FP32: plain OpticSpaceNetStudent — the electronic baseline.
- fake-optical: same weights through
  ``build_optical_model(input_bit=8, weight_bit=8, keep_first_conv_electronic=True)``
  on ``OpticalEngine(use_real=False)`` (int8 pseudo-quantized matmul) — the
  local stand-in for the remote gazelle engine when it is unreachable.

Image preprocessing: PIL Image → aspect-preserving resize so the short side is
64 → center-crop 64×64 → ToTensor → ImageNet normalize (matches training).
"""
import base64
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))          # demo/server
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))           # repo root
for _p in (_HERE, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _pathsetup  # noqa: E402,F401  (adds src/core, src/data, ...)

from optic_layers import OpticalEngine, build_optical_model  # noqa: E402

from demo.server import model_trace  # noqa: E402
from demo.server.model_trace import CLASSES, build_student, forward_traced  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data", "EuroSAT_RGB")
WEIGHT_PATH = os.path.join(REPO_ROOT, "weights", "spacenet_v2_phase4_v3_int8.pth")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_fp32_model = None
_fake_model = None
_fake_engine = None


def get_fp32_model():
    """Lazy singleton: plain FP32 student with the trained int8-v3 weights."""
    global _fp32_model
    if _fp32_model is None:
        _fp32_model = build_student(WEIGHT_PATH)
    return _fp32_model


def get_fake_model():
    """Lazy singleton: fake-optical model (int8 pseudo-quantized matmul)."""
    global _fake_model, _fake_engine
    if _fake_model is None:
        _fake_engine = OpticalEngine(use_real=False, verbose=False)
        _fake_model = build_optical_model(
            build_student(WEIGHT_PATH), _fake_engine,
            pad_to_8=True, input_bit=8, weight_bit=8,
            keep_first_conv_electronic=True)
        _fake_model.eval()
    return _fake_model


def preprocess(image_bytes):
    """jpeg/png bytes (any size) → normalized tensor (1, 3, 64, 64)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = 64.0 / min(w, h)
    if scale != 1.0:
        img = img.resize((max(64, round(w * scale)), max(64, round(h * scale))),
                         Image.BILINEAR)
    w, h = img.size
    left, top = (w - 64) // 2, (h - 64) // 2
    img = img.crop((left, top, left + 64, top + 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def encode_act_b64(act_tensor):
    """(1, ...) tensor → base64(np.savez(float16 act)), batch dim squeezed."""
    arr = act_tensor.detach().cpu().numpy()[0].astype(np.float16)
    buf = io.BytesIO()
    np.savez(buf, act=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _path_result(engine_label, model, img_tensor):
    """Run forward_traced and assemble the api.md PathResult dict."""
    traced = forward_traced(model, img_tensor)
    logits = traced["logits"][0]
    probs = torch.softmax(logits, dim=0)
    order = torch.argsort(probs, descending=True)
    prob_dict = {CLASSES[i]: round(float(probs[i]), 6) for i in order.tolist()}

    layers = [{
        "name": l["name"],
        "where": l["where"],
        "spec": l["spec"],
        "shape": list(l["shape"]),
        "latency_s": round(l["latency_s"], 6),
        "act_b64": encode_act_b64(l["act"]),
    } for l in traced["layers"]]

    return {
        "engine": engine_label,
        "pred": CLASSES[int(logits.argmax())],
        "probs": prob_dict,
        "latency_total_s": round(sum(l["latency_s"] for l in layers), 6),
        "layers": layers,
    }


def infer_fp32(img_tensor):
    """FP32 electronic baseline path."""
    return _path_result("fp32-local", get_fp32_model(), img_tensor)


def infer_fake(img_tensor):
    """Fake-optical path (local stand-in for the remote gazelle engine)."""
    return _path_result("fake-optical", get_fake_model(), img_tensor)
