"""Local inference paths for multi-model comparison: FP32 + fake-optical for Model 1/2/3.

Extends inference_local.py to support all three models with fake-optical fallback.
Each model is a lazy singleton loaded from its respective weight file.
"""
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _pathsetup  # noqa: E402,F401

from optic_layers import OpticalEngine, build_optical_model  # noqa: E402
from demo.server.inference_local import CLASSES, preprocess  # noqa: E402

import torch.nn as nn  # noqa: E402


# ============================================================
#  Model Definitions (mirrors training scripts)
# ============================================================

class BaselineVGG(nn.Module):
    """Model 1: Flat VGG (6 Conv 3×3 + 2 Linear), bias=False."""
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


class OpticSpaceNetV1(nn.Module):
    """Model 2: SpaceNet V1, same arch as Model 3."""
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


# Model 3 reuses OpticSpaceNetV1 architecture (different weights)
OpticSpaceNetStudent = OpticSpaceNetV1


# ============================================================
#  Weight paths
# ============================================================

WEIGHT_PATHS = {
    1: os.path.join(REPO_ROOT, "weights", "baseline_vgg_phase4_v3_int8.pth"),
    2: os.path.join(REPO_ROOT, "weights", "spacenet_v1_phase4_v3_int8.pth"),
    3: os.path.join(REPO_ROOT, "weights", "spacenet_v2_phase4_v3_int8.pth"),
}

MODEL_CLASSES = {
    1: BaselineVGG,
    2: OpticSpaceNetV1,
    3: OpticSpaceNetStudent,
}


# ============================================================
#  Lazy singletons
# ============================================================

_fake_models = {}  # model_id -> (model, engine)


def _load_weight_into(model, weight_path):
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    own = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(filtered, strict=False)


def get_fake_model(model_id):
    """Lazy singleton: fake-optical model for given model_id."""
    if model_id not in _fake_models:
        engine = OpticalEngine(use_real=False, verbose=False)
        model_cls = MODEL_CLASSES[model_id]
        model = model_cls()
        _load_weight_into(model, WEIGHT_PATHS[model_id])
        optical = build_optical_model(
            model, engine, pad_to_8=True, input_bit=8, weight_bit=8,
            keep_first_conv_electronic=True)
        optical.eval()
        _fake_models[model_id] = (optical, engine)
    return _fake_models[model_id]


# ============================================================
#  Layer metadata per model
# ============================================================

MODEL1_LAYER_NAMES = ["conv1_1", "conv1_2", "conv2_1", "conv2_2",
                      "conv3_1", "conv3_2", "fc1", "fc2"]
MODEL1_LAYER_WHERE = {
    "conv1_1": "electronic", "conv1_2": "optical", "conv2_1": "optical",
    "conv2_2": "optical", "conv3_1": "optical", "conv3_2": "optical",
    "fc1": "optical", "fc2": "optical",
}
MODEL1_LAYER_SPECS = {
    "conv1_1": "Conv2d 3→32 3×3 + BN + ReLU",
    "conv1_2": "Conv2d 32→32 3×3 + BN + ReLU + Pool",
    "conv2_1": "Conv2d 32→64 3×3 + BN + ReLU",
    "conv2_2": "Conv2d 64→64 3×3 + BN + ReLU + Pool",
    "conv3_1": "Conv2d 64→128 3×3 + BN + ReLU",
    "conv3_2": "Conv2d 128→128 3×3 + BN + ReLU + Pool",
    "fc1": "Linear 8192→256 + ReLU",
    "fc2": "Linear 256→10",
}

SPACENET_LAYER_NAMES = ["stem", "stage1", "stage2", "stage3", "fc1", "fc2"]
SPACENET_LAYER_WHERE = {
    "stem": "electronic", "stage1": "optical", "stage2": "optical",
    "stage3": "optical", "fc1": "optical", "fc2": "optical",
}
SPACENET_LAYER_SPECS = {
    "stem": "Conv2d 3→8 1×1 + BN + ReLU",
    "stage1": "Conv2d 8→16 2×2/s2 + BN + ReLU + Pool",
    "stage2": "Conv2d 16→32 2×2/s2 + BN + ReLU",
    "stage3": "Conv2d 32→16 1×1 + BN + ReLU",
    "fc1": "Linear 1024→256 + ReLU",
    "fc2": "Linear 256→10",
}


def get_layer_meta(model_id):
    if model_id == 1:
        return MODEL1_LAYER_NAMES, MODEL1_LAYER_WHERE, MODEL1_LAYER_SPECS
    return SPACENET_LAYER_NAMES, SPACENET_LAYER_WHERE, SPACENET_LAYER_SPECS


# ============================================================
#  Traced forward (per-layer latency)
# ============================================================

@torch.no_grad()
def forward_traced_model1(model, x):
    """Segmented forward for Model 1."""
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
    logits = _trace("fc2", model.fc2, h)
    return {"logits": logits, "layers": layers}


@torch.no_grad()
def forward_traced_spacenet(model, x):
    """Segmented forward for Model 2/3 (SpaceNet)."""
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


# ============================================================
#  Inference entry point
# ============================================================

def infer_fake(model_id, img_tensor):
    """Run fake-optical inference for a single model, return simplified result."""
    model, _engine = get_fake_model(model_id)

    t0 = time.perf_counter()
    if model_id == 1:
        traced = forward_traced_model1(model, img_tensor)
    else:
        traced = forward_traced_spacenet(model, img_tensor)
    total_latency = time.perf_counter() - t0

    logits = traced["logits"][0]
    probs = torch.softmax(logits, dim=0)
    order = torch.argsort(probs, descending=True)
    prob_dict = {CLASSES[i]: round(float(probs[i]), 6) for i in order.tolist()}

    layer_names, layer_where, layer_specs = get_layer_meta(model_id)
    layers_out = [{
        "name": l["name"],
        "where": layer_where[l["name"]],
        "spec": layer_specs[l["name"]],
        "shape": list(l["act"].shape[1:]),
        "latency_s": round(l["latency_s"], 6),
    } for l in traced["layers"]]

    return {
        "engine": "fake-optical",
        "model_id": model_id,
        "pred": CLASSES[int(logits.argmax())],
        "probs": prob_dict,
        "latency_total_s": round(total_latency, 6),
        "layers": layers_out,
    }
