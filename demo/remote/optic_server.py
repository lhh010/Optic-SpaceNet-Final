#!/usr/bin/env python3
"""optical inference server for the gazelle_sim container (stdlib-only HTTP).

Runs inside the remote Docker container where only stdlib + numpy + torch +
osimulator are available — the HTTP layer is pure ``http.server``, no
flask/fastapi/PIL dependency.  The model definition is inlined (training
scripts have import side effects) and the only local import is
``optic_layers`` from the same directory (deployed flat by demo/deploy.sh).

Endpoints:
  GET  /health -> {"status", "engine", "weight", "uptime_s", "models": [...]}
  POST /infer  {"image_b64": jpeg/png b64} -> PathResult (see demo/docs/api.md)
               Query param ?model=1|2|3 selects model (default: 3 for backward compat)
               400 on undecodable image, 500 on inference failure.

Env:
  OPTIC_FAKE=1    force the FakeOpticalEngine (local integration tests)
  OPTIC_WEIGHT    override the weight file path (applies to model 3 only)
  --port          listen port (default 8765)
  --models        comma-separated model IDs to load (default: "1,2,3")

The module is import-safe: nothing is loaded or bound until create_server()
or main() is called.
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
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

WEIGHT_NAMES = {
    1: "baseline_vgg_phase4_v3_int8.pth",
    2: "spacenet_v1_phase4_v3_int8.pth",
    3: "spacenet_v2_phase4_v3_int8.pth",
}
WEIGHT_NAME = WEIGHT_NAMES[3]  # backward compat

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

# --- Model 1 layer metadata ---
MODEL1_LAYER_WHERE = {
    "conv1_1": "electronic", "conv1_2": "optical", "conv2_1": "optical",
    "conv2_2": "optical", "conv3_1": "optical", "conv3_2": "optical",
    "fc1": "optical", "fc2": "optical",
}
MODEL1_LAYER_SPECS = {
    "conv1_1": "Conv2d 3→32 3×3 + BN + ReLU",
    "conv1_2": "Conv2d 32→32 3×3 + BN + ReLU + MaxPool2d",
    "conv2_1": "Conv2d 32→64 3×3 + BN + ReLU",
    "conv2_2": "Conv2d 64→64 3×3 + BN + ReLU + MaxPool2d",
    "conv3_1": "Conv2d 64→128 3×3 + BN + ReLU",
    "conv3_2": "Conv2d 128→128 3×3 + BN + ReLU + MaxPool2d",
    "fc1": "Linear 8192→256 + ReLU",
    "fc2": "Linear 256→10",
}

# Model 2 reuses same LAYER_WHERE / LAYER_SPECS as Model 3 (same architecture)
MODEL2_LAYER_WHERE = LAYER_WHERE
MODEL2_LAYER_SPECS = {
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


# Model 2 is architecturally identical to Model 3 (different weights only)
OpticSpaceNetV1 = OpticSpaceNetStudent


class BaselineVGG(nn.Module):
    """Model 1: Flat VGG (6 Conv 3×3 + 2 Linear), bias=False, BN retained."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


MODEL_CLASSES = {
    1: BaselineVGG,
    2: OpticSpaceNetV1,
    3: OpticSpaceNetStudent,
}

# Model 1 variant A: only conv1_1 stays electronic
MODEL1_ELECTRONIC_LAYERS = {"conv1_1"}


def _find_weight(weight_name, weight_path=None):
    """Weight search order: explicit arg > $OPTIC_WEIGHT > alongside this file
    (deployed layout) > repo weights/ directory."""
    candidates = [
        weight_path,
        os.environ.get("OPTIC_WEIGHT") if weight_name == WEIGHT_NAME else None,
        os.path.join(_HERE, weight_name),
        os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir,
                                     "weights", weight_name)),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"weight file {weight_name!r} not found, tried: {candidates}")


def _load_weight_into(model, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    own = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(filtered, strict=False)


class SingleModelContext:
    """Engine + optical model for one model variant."""

    def __init__(self, model_id, engine, weight_path=None):
        self.model_id = model_id
        weight_name = WEIGHT_NAMES[model_id]
        self.weight_path = _find_weight(weight_name, weight_path)
        self.engine = engine
        model_cls = MODEL_CLASSES[model_id]
        model = model_cls()
        _load_weight_into(model, self.weight_path)
        if model_id == 1:
            self.model = build_optical_model(
                model, self.engine, pad_to_8=True, input_bit=8, weight_bit=8,
                keep_first_conv_electronic=True)
        else:
            self.model = build_optical_model(
                model, self.engine, pad_to_8=True, input_bit=8, weight_bit=8,
                keep_first_conv_electronic=True)
        self.model.eval()


class ModelContext:
    """Multi-model context: shares a single OpticalEngine across models."""

    def __init__(self, weight_path=None, fake=None, model_ids=None):
        if fake is None:
            fake = os.environ.get("OPTIC_FAKE") == "1"
        if model_ids is None:
            model_ids = [3]  # backward compat: only model 3
        self.engine = OpticalEngine(use_real=not fake, verbose=False)
        self.engine_label = "gazelle-osim" if self.engine.use_real else "fake-optical"
        self.started_at = time.time()
        self.models = {}
        for mid in model_ids:
            wp = weight_path if mid == 3 else None
            ctx = SingleModelContext(mid, self.engine, weight_path=wp)
            self.models[mid] = ctx
            print(f"[optic_server] loaded model {mid}: "
                  f"{os.path.basename(ctx.weight_path)}", flush=True)
        # backward compat attrs
        if 3 in self.models:
            self.model = self.models[3].model
            self.weight_path = self.models[3].weight_path
        elif self.models:
            first = next(iter(self.models.values()))
            self.model = first.model
            self.weight_path = first.weight_path


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


@torch.no_grad()
def forward_traced_model1(model, x):
    """Segmented forward for Model 1 (BaselineVGG): 6 conv groups + 2 fc."""
    layers = []

    def _trace(name, fn, inp):
        t0 = time.perf_counter()
        out = fn(inp)
        layers.append({"name": name, "act": out,
                       "latency_s": time.perf_counter() - t0})
        return out

    h = _trace("conv1_1", lambda t: torch.relu(model.bn1_1(model.conv1_1(t))), x)
    h = _trace("conv1_2", lambda t: model.pool1(torch.relu(model.bn1_2(model.conv1_2(t)))), h)
    h = _trace("conv2_1", lambda t: torch.relu(model.bn2_1(model.conv2_1(t))), h)
    h = _trace("conv2_2", lambda t: model.pool2(torch.relu(model.bn2_2(model.conv2_2(t)))), h)
    h = _trace("conv3_1", lambda t: torch.relu(model.bn3_1(model.conv3_1(t))), h)
    h = _trace("conv3_2", lambda t: model.pool3(torch.relu(model.bn3_2(model.conv3_2(t)))), h)
    h = model.flatten(h)
    h = _trace("fc1", lambda t: torch.relu(model.fc1(t)), h)
    logits = _trace("fc2", lambda t: model.fc2(t), h)
    return {"logits": logits, "layers": layers}


def _encode_act(act_tensor):
    arr = act_tensor.detach().cpu().numpy()[0].astype(np.float16)
    buf = io.BytesIO()
    np.savez(buf, act=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _get_layer_meta(model_id):
    """Return (layer_where, layer_specs) for a given model_id."""
    if model_id == 1:
        return MODEL1_LAYER_WHERE, MODEL1_LAYER_SPECS
    elif model_id == 2:
        return MODEL2_LAYER_WHERE, MODEL2_LAYER_SPECS
    else:
        return LAYER_WHERE, LAYER_SPECS


def run_inference(ctx, image_bytes, model_id=3):
    """Full request path: decode → traced forward → PathResult dict."""
    if model_id not in ctx.models:
        raise ValueError(f"model {model_id} not loaded")
    single = ctx.models[model_id]
    img_tensor = decode_image(image_bytes)

    if model_id == 1:
        traced = forward_traced_model1(single.model, img_tensor)
    else:
        traced = forward_traced(single.model, img_tensor)

    logits = traced["logits"][0]
    probs = torch.softmax(logits, dim=0)
    order = torch.argsort(probs, descending=True)
    prob_dict = {CLASSES[i]: round(float(probs[i]), 6) for i in order.tolist()}

    layer_where, layer_specs = _get_layer_meta(model_id)
    layers = [{
        "name": l["name"],
        "where": layer_where[l["name"]],
        "spec": layer_specs[l["name"]],
        "shape": list(l["act"].shape[1:]),
        "latency_s": round(l["latency_s"], 6),
        "act_b64": _encode_act(l["act"]),
    } for l in traced["layers"]]

    return {
        "engine": ctx.engine_label,
        "model_id": model_id,
        "pred": CLASSES[int(logits.argmax())],
        "probs": prob_dict,
        "latency_total_s": round(sum(l["latency_s"] for l in layers), 6),
        "layers": layers,
    }




def run_matmul(ctx, payload, weight_cache, matmul_lock):
    """Gazelle-compatible integer MVM for M9/M10 and selectable Model 3."""
    import hashlib

    if "weight_b64" in payload:
        shape = tuple(int(v) for v in payload.get("weight_shape", []))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("weight_shape must be [k,n]")
        raw = base64.b64decode(payload["weight_b64"], validate=True)
        weight = np.frombuffer(raw, dtype=np.int8)
        if weight.size != shape[0] * shape[1]:
            raise ValueError("weight byte count does not match weight_shape")
        weight = np.ascontiguousarray(weight.reshape(shape))
        weight_id = hashlib.md5(weight.tobytes()).hexdigest()
        with matmul_lock:
            weight_cache[weight_id] = weight
        return {"weight_id": weight_id, "cached": True}

    weight_id = str(payload.get("weight_id", ""))
    with matmul_lock:
        weight = weight_cache.get(weight_id)
    if weight is None:
        raise ValueError("unknown weight_id; upload weight_b64 first")

    act_shape = tuple(int(v) for v in payload.get("act_shape", []))
    if len(act_shape) != 2 or min(act_shape) <= 0:
        raise ValueError("act_shape must be [m,k]")
    act_raw = base64.b64decode(payload["act_b64"], validate=True)
    act = np.frombuffer(act_raw, dtype=np.uint8)
    if act.size != act_shape[0] * act_shape[1]:
        raise ValueError("activation byte count does not match act_shape")
    act = np.ascontiguousarray(act.reshape(act_shape))
    if act.shape[1] != weight.shape[0]:
        raise ValueError(
            "matmul k mismatch: %d vs %d" %
            (act.shape[1], weight.shape[0]))

    with matmul_lock:
        if ctx.engine.use_real:
            x3 = act.astype(np.int32, copy=False)[None, :, :]
            w3 = weight.astype(np.int32, copy=False)[None, :, :]
            raw_result = ctx.engine._real_model(x3, w3, inputType="uint8")
            if torch.is_tensor(raw_result):
                out = raw_result.detach().cpu().numpy()
            else:
                out = np.asarray(raw_result)
            if out.ndim == 3:
                out = out[0]
        else:
            out = act.astype(np.float64) @ weight.astype(np.float64)
    out = np.asarray(out, dtype=np.float64).reshape(
        act.shape[0], weight.shape[1])
    return {"shape": list(out.shape), "data": out.ravel().tolist()}

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

    def _parse_query(self):
        """Parse query string from self.path → dict."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        return parsed.path, parse_qs(parsed.query)

    def do_GET(self):
        path, qs = self._parse_query()
        if path == "/health":
            ctx = self.server.ctx
            self._send_json(200, {
                "status": "ok",
                "engine": ctx.engine_label,
                "models": sorted(ctx.models.keys()),
                "matmul": True,
                "uptime_s": round(time.time() - ctx.started_at, 3),
            })
        else:
            self._send_json(404, {"error": f"unknown route {path}"})

    def do_POST(self):
        path, qs = self._parse_query()
        if path == "/matmul":
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > 64 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                result = run_matmul(
                    self.server.ctx, payload, self.server.weight_cache,
                    self.server.matmul_lock)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"error": "matmul failed: %s" % exc})
            else:
                self._send_json(200, result)
            return
        if path != "/infer":
            self._send_json(404, {"error": f"unknown route {path}"})
            return
        # Parse model ID from ?model=N (default 3 for backward compat)
        model_id = int(qs.get("model", ["3"])[0])
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            image_bytes = base64.b64decode(payload["image_b64"])
        except Exception as e:
            self._send_json(400, {"error": f"bad request body: {e}"})
            return
        try:
            result = run_inference(self.server.ctx, image_bytes,
                                  model_id=model_id)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"inference failed: {e}"})
        else:
            self._send_json(200, result)

    def log_message(self, fmt, *args):  # keep container logs quiet
        pass


def create_server(host="0.0.0.0", port=8765, weight_path=None, fake=None,
                  model_ids=None):
    """Build a ThreadingHTTPServer with model(s) loaded once."""
    ctx = ModelContext(weight_path=weight_path, fake=fake, model_ids=model_ids)
    server = ThreadingHTTPServer((host, port), OpticHandler)
    server.ctx = ctx
    server.weight_cache = {}
    server.matmul_lock = threading.RLock()
    print(f"[optic_server] engine={ctx.engine_label} "
          f"models={sorted(ctx.models.keys())} "
          f"listening on {host}:{server.server_address[1]}", flush=True)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="optical inference server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--weight", default=None)
    parser.add_argument("--models", default="1,2,3",
                        help="comma-separated model IDs to load (default: 1,2,3)")
    args = parser.parse_args(argv)
    model_ids = [int(x.strip()) for x in args.models.split(",")]
    server = create_server(host=args.host, port=args.port,
                           weight_path=args.weight, model_ids=model_ids)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
