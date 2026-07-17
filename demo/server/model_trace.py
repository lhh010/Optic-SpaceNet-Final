"""Model definition + segmented traced forward for the optics demo.

The model is a copy of ``OpticSpaceNetStudent`` from
``src/scripts/optic_inference_kd.py`` (repo convention: demo code duplicates the
definition instead of importing the training script, which has side effects).

``forward_traced(model, x)`` runs the network in 6 segments
(stem, stage1, stage2, stage3, fc1, fc2) and returns each segment's output
tensor + wall-clock latency plus the final logits.  It works unchanged for the
plain FP32 model and for the optical model produced by
``optic_layers.build_optical_model`` (OpticConv2d/OpticLinear replacements keep
the same Sequential structure), so hook points stay strictly aligned between
the two paths.

fc1 = output of classifier's Linear(1024->256) + ReLU; fc2 = final logits.
Activation shapes for a 64x64 input:
  stem (8,64,64) -> stage1 (16,16,16) -> stage2 (32,8,8) -> stage3 (16,8,8)
  -> fc1 (256,) -> fc2 (10,)
"""
import time

import torch
import torch.nn as nn

# EuroSAT classes in ImageFolder (alphabetical) order.
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# api.md Layer contract: stem stays electronic, the other 5 segments run
# optically after build_optical_model(keep_first_conv_electronic=True).
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

# Expected single-sample activation shapes (C,H,W) / (N,) for 64×64 input.
EXPECTED_SHAPES = {
    "stem": (8, 64, 64), "stage1": (16, 16, 16), "stage2": (32, 8, 8),
    "stage3": (16, 8, 8), "fc1": (256,), "fc2": (10,),
}


class OpticSpaceNetStudent(nn.Module):
    """Model 3 KD Student (copy of src/scripts/optic_inference_kd.py)."""

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


def load_student_weights(model, weight_path):
    """Load a state dict with shape filtering + strict=False.

    Entries whose key is unknown or whose shape does not match the model are
    dropped (tolerates QAT checkpoints carrying extra quantization buffers).
    Returns {"loaded", "missing", "unexpected"}.
    """
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    own = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in own and tuple(own[k].shape) == tuple(v.shape)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {"loaded": len(filtered), "missing": list(missing),
            "unexpected": list(unexpected)}


def build_student(weight_path=None, num_classes=10):
    """Fresh OpticSpaceNetStudent in eval mode, optionally weight-loaded."""
    model = OpticSpaceNetStudent(num_classes)
    if weight_path is not None:
        load_student_weights(model, weight_path)
    model.eval()
    return model


@torch.no_grad()
def forward_traced(model, x):
    """Run ``model`` segment by segment, capturing activations and latencies.

    Returns {"logits": Tensor(1,10), "layers": [layer, ...]} where each layer
    is {"name", "where", "spec", "shape": tuple, "latency_s": float,
    "act": Tensor(1, ...)}.  Only the 6 compute segments are timed; BN/ReLU/
    pooling are folded into their conv segment's latency.
    """
    model.eval()
    layers = []

    def _trace(name, fn, inp):
        t0 = time.perf_counter()
        out = fn(inp)
        layers.append({
            "name": name,
            "where": LAYER_WHERE[name],
            "spec": LAYER_SPECS[name],
            "shape": tuple(out.shape[1:]),
            "latency_s": time.perf_counter() - t0,
            "act": out,
        })
        return out

    h = _trace("stem", model.stem, x)
    h = _trace("stage1", model.stage1, h)
    h = _trace("stage2", model.stage2, h)
    h = _trace("stage3", model.stage3, h)

    # classifier: Flatten, Linear(1024→256), ReLU, Dropout, Linear(256→10)
    # fc1 = Linear+ReLU output; fc2 = Dropout+Linear output (final logits).
    cls = list(model.classifier.children())
    h = _trace("fc1", lambda t: cls[2](cls[1](cls[0](t))), h)
    logits = _trace("fc2", lambda t: cls[4](cls[3](t)), h)

    return {"logits": logits, "layers": layers}
