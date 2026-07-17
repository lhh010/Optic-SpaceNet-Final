#!/usr/bin/env python3
"""optical inference server for the gazelle_sim container (stdlib-only HTTP).

Runs inside the remote Docker container where only stdlib + numpy + torch +
osimulator are available — the HTTP layer is pure ``http.server``, no
flask/fastapi/PIL dependency.  The model definition is inlined (training
scripts have import side effects) and the only local import is
``optic_layers`` from the same directory (deployed flat by demo/deploy.sh).

Endpoints:
  GET  /health -> {"status", "engine", "weight", "uptime_s"}
  POST /infer  {"image_b64": jpeg/png b64} -> PathResult (see demo/docs/api.md)
               400 on undecodable image, 500 on inference failure.

Env:
  OPTIC_FAKE=1    force the FakeOpticalEngine (local integration tests)
  OPTIC_WEIGHT    override the weight file path
  --port          listen port (default 8765)

The module is import-safe: nothing is loaded or bound until create_server()
or main() is called.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
# optic_layers.py sits next to this file when deployed (demo/deploy.sh packs
# them flat); in the repo tree it lives at src/core/.
for _cand in (_HERE, os.path.join(_HERE, os.pardir, os.pardir, "src", "core")):
    if os.path.isfile(os.path.join(_cand, "optic_layers.py")):
        _cand = os.path.abspath(_cand)
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from optic_layers import OpticalEngine, build_optical_model

WEIGHT_NAME = "spacenet_v2_phase4_v3_int8.pth"

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

LAYER_WHERE = {
    "stem": "electronic", "stage1": "optical", "stage2": "optical",
    "stage3": "optical", "fc1": "optical", "fc2": "optical",
}
LAYER_SPECS = {
    "stem": "Conv2d 3→8 1×1 + BN + ReLU",
    "stage1": "Conv2d 8→16 2×2/s2 + BN + ReLU + MaxPool2d",
    "stage2": "Conv2d 16→32 2×2/s2 + BN + ReLU",
    "stage3": "Conv2d 32→16 1×1 + BN + ReLU",
    "fc1": "Linear 1024→256 + ReLU",
    "fc2": "Linear 256→10",
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ============================================================
#  Model (copy of OpticSpaceNetStudent from src/scripts/optic_inference_kd.py)
# ============================================================
class OpticSpaceNetStudent(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False), nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


def _find_weight(weight_path=None):
    """Weight search order: explicit arg > $OPTIC_WEIGHT > alongside this file
    (deployed layout) > repo weights/ directory."""
    candidates = [
        weight_path,
        os.environ.get("OPTIC_WEIGHT"),
        os.path.join(_HERE, WEIGHT_NAME),
        os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir,
                                     "weights", WEIGHT_NAME)),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"weight file not found, tried: {candidates}")


def _load_weight_into(model, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    own = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(filtered, strict=False)


class ModelContext:
    """Engine + optical model, loaded once at server startup."""

    def __init__(self, weight_path=None, fake=None):
        if fake is None:
            fake = os.environ.get("OPTIC_FAKE") == "1"
        self.weight_path = _find_weight(weight_path)
        self.engine = OpticalEngine(use_real=not fake, verbose=False)
        model = OpticSpaceNetStudent()
        _load_weight_into(model, self.weight_path)
        self.model = build_optical_model(
            model, self.engine, pad_to_8=True, input_bit=8, weight_bit=8,
            keep_first_conv_electronic=True)
        self.model.eval()
        self.engine_label = "gazelle-osim" if self.engine.use_real else "fake-optical"
        self.started_at = time.time()


# ============================================================
#  Image decode + preprocessing (no PIL required: torchvision.io first)
# ============================================================
def decode_image(image_bytes):
    """jpeg/png bytes → normalized tensor (1, 3, 64, 64).

    Decode chain: torchvision.io.decode_image (container has torchvision) →
    PIL Image (local fallback).  Raises ValueError if neither works.
    """
    img_chw = None
    try:
        from torchvision.io import decode_image as _tv_decode
        img_chw = _tv_decode(
            torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8).clone())
    except ImportError:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            img_chw = torch.from_numpy(arr.transpose(2, 0, 1))
        except ImportError:
            raise ValueError("no image decoder available "
                             "(need torchvision or PIL)")
    except Exception as e:
        raise ValueError(f"image decode failed: {e}")

    if img_chw.dim() != 3 or img_chw.shape[0] not in (1, 3, 4):
        raise ValueError(f"unexpected decoded shape {tuple(img_chw.shape)}")

    img = img_chw[:3].float() / 255.0                       # (3, H, W) in [0,1]
    _, h, w = img.shape
    scale = 64.0 / min(h, w)
    if abs(scale - 1.0) > 1e-6:
        new_h, new_w = max(64, round(h * scale)), max(64, round(w * scale))
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                            mode="bilinear", align_corners=False)[0]
    _, h, w = img.shape
    top, left = (h - 64) // 2, (w - 64) // 2
    img = img[:, top:top + 64, left:left + 64]
    return ((img - _IMAGENET_MEAN) / _IMAGENET_STD).unsqueeze(0)


# ============================================================
#  Traced forward → PathResult
# ============================================================
@torch.no_grad()
def forward_traced(model, x):
    """Segmented forward (stem..fc2) with per-segment latency — identical
    segmentation to demo/server/model_trace.py (kept inline: self-contained)."""
    layers = []

    def _trace(name, fn, inp):
        t0 = time.perf_counter()
        out = fn(inp)
        layers.append({"name": name, "act": out,
                       "latency_s": time.perf_counter() - t0})
        return out

    h = _trace("stem", model.stem, x)
    h = _trace("stage1", model.stage1, h)
    h = _trace("stage2", model.stage2, h)
    h = _trace("stage3", model.stage3, h)
    cls = list(model.classifier.children())
    h = _trace("fc1", lambda t: cls[2](cls[1](cls[0](t))), h)
    logits = _trace("fc2", lambda t: cls[4](cls[3](t)), h)
    return {"logits": logits, "layers": layers}


def _encode_act(act_tensor):
    arr = act_tensor.detach().cpu().numpy()[0].astype(np.float16)
    buf = io.BytesIO()
    np.savez(buf, act=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_inference(ctx, image_bytes):
    """Full request path: decode → traced forward → PathResult dict."""
    img_tensor = decode_image(image_bytes)
    traced = forward_traced(ctx.model, img_tensor)
    logits = traced["logits"][0]
    probs = torch.softmax(logits, dim=0)
    order = torch.argsort(probs, descending=True)
    prob_dict = {CLASSES[i]: round(float(probs[i]), 6) for i in order.tolist()}

    layers = [{
        "name": l["name"],
        "where": LAYER_WHERE[l["name"]],
        "spec": LAYER_SPECS[l["name"]],
        "shape": list(l["act"].shape[1:]),
        "latency_s": round(l["latency_s"], 6),
        "act_b64": _encode_act(l["act"]),
    } for l in traced["layers"]]

    return {
        "engine": ctx.engine_label,
        "pred": CLASSES[int(logits.argmax())],
        "probs": prob_dict,
        "latency_total_s": round(sum(l["latency_s"] for l in layers), 6),
        "layers": layers,
    }


# ============================================================
#  HTTP layer (stdlib only)
# ============================================================
class OpticHandler(BaseHTTPRequestHandler):
    server_version = "OpticServer/1.0"
    protocol_version = "HTTP/1.1"

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ctx = self.server.ctx
            self._send_json(200, {
                "status": "ok",
                "engine": ctx.engine_label,
                "weight": os.path.basename(ctx.weight_path),
                "uptime_s": round(time.time() - ctx.started_at, 3),
            })
        else:
            self._send_json(404, {"error": f"unknown route {self.path}"})

    def do_POST(self):
        if self.path != "/infer":
            self._send_json(404, {"error": f"unknown route {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            image_bytes = base64.b64decode(payload["image_b64"])
        except Exception as e:
            self._send_json(400, {"error": f"bad request body: {e}"})
            return
        try:
            result = run_inference(self.server.ctx, image_bytes)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"inference failed: {e}"})
        else:
            self._send_json(200, result)

    def log_message(self, fmt, *args):  # keep container logs quiet
        pass


def create_server(host="0.0.0.0", port=8765, weight_path=None, fake=None):
    """Build a ThreadingHTTPServer with the model loaded exactly once."""
    ctx = ModelContext(weight_path=weight_path, fake=fake)
    server = ThreadingHTTPServer((host, port), OpticHandler)
    server.ctx = ctx
    print(f"[optic_server] engine={ctx.engine_label} "
          f"weight={os.path.basename(ctx.weight_path)} "
          f"listening on {host}:{server.server_address[1]}", flush=True)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="optical inference server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--weight", default=None)
    args = parser.parse_args(argv)
    server = create_server(host=args.host, port=args.port,
                           weight_path=args.weight)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
